#!/usr/bin/env python3
"""Smoke-test for semantic/distributional geometry evaluation.

Runs 3 behaviors x 5 steps x 3 conditions (F-C+D, Base I-GCG, Slot I-GCG),
generates 10 attacked + 10 refused samples per behavior-condition pair,
then computes all metrics except MAUVE (needs more samples).

Usage
-----
    python scripts/geometry_eval_smoke.py --model_path /workspace/models/Qwen2-7B-Instruct

Runtime: ~15 min on A100.
"""
from __future__ import annotations

import argparse
import copy
import datetime
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from llm_attacks.minimal_gcg.opt_utils import load_model_and_tokenizer
from llm_attacks.minimal_gcg.string_utils import SuffixManager, load_conversation_template
from llm_attacks.minimal_gcg.slot_utils import make_hybrid_ce_refusal_loss_fn
from robust_gcg.attack_harness import run_attack_with_model
from robust_gcg.eval_utils import TEST_PREFIXES, _is_jailbroken
from robust_gcg.semantic_eval import (
    evaluate_behavior,
    load_gold_completions,
)
from scripts.robust_gcg_F_slot_kmerge import run_slot_attack_with_model

# re-use helpers from fcd_attack_report
from scripts.fcd_attack_report import (
    CONTENT_KEYWORDS,
    STRICT_REFUSAL_PHRASES,
    TEMPLATE_NAME,
    _build_refusal_ids,
    generate_extended,
    make_arca_hook,
    make_content_early_stop,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEED = 235711
SMOKE_BIDS = [3, 8, 32]
SMOKE_STEPS = 5
N_GEN_ATTACKED = 10
N_GEN_REFUSED = 10
EMBED_MODEL = "all-MiniLM-L6-v2"
DEFAULT_CONFIG = "data/cyber_behaviors_v2_all40.json"
DEFAULT_GOLD = "data/gold_harmful_completions_smoke.json"
DEFAULT_LAM = 0.2

SMOKE_SLOT_PARAMS: Dict[str, Any] = {
    "num_steps": SMOKE_STEPS,
    "search_width": 64,
    "top_k": 64,
    "num_adv_tokens": 20,
    "attention_temp": 8.0,
    "kmerge_k": 3,
    "eval_batch_size": 16,
    "use_prefix_cache": True,
    "adv_string_init": "!",
    "seed": SEED,
}

SMOKE_BASE_PARAMS: Dict[str, Any] = {
    "num_steps": SMOKE_STEPS,
    "batch_size": 32,
    "top_k": 32,
    "seed": SEED,
    "n_flips": 1,
    # 64 candidates in one forward-pass OOMs Qwen2-7B-fp16 on a 24GB GPU and
    # silently spills to shared memory on Windows (WDDM), which stalls runs
    # by orders of magnitude. Measured on this box: eval_batch_size=16 was
    # still at 97% VRAM and paged to shared RAM (~1900 s/step). Dropped to 4.
    "eval_batch_size": 4,
}


def _argmin_clean_loss(new_adv_suffix, clean_losses, **kw):
    """Trivial candidate selector for base I-GCG."""
    return new_adv_suffix[clean_losses.argmin()], 0.0


@torch.no_grad()
def generate_samples(
    model, tokenizer, prompt_text: str, n: int,
    max_new_tokens: int = 256, temperature: float = 0.7,
) -> List[str]:
    """Generate *n* text completions from a fully-formatted prompt."""
    gen_cfg = copy.deepcopy(model.generation_config)
    gen_cfg.max_new_tokens = max_new_tokens
    gen_cfg.do_sample = True
    gen_cfg.temperature = temperature
    gen_cfg.top_p = 0.95

    results = []
    for _ in range(n):
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        out = model.generate(
            **inputs, generation_config=gen_cfg,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        text = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True,
        ).strip()
        results.append(text)
    return results


def generate_refused_samples(
    model, tokenizer, behavior: Dict, n: int, template_name: str,
) -> List[str]:
    """Generate refusals by prompting the aligned model with no adversarial suffix."""
    conv = load_conversation_template(template_name)
    sm = SuffixManager(
        tokenizer=tokenizer, conv_template=conv,
        instruction=behavior["behaviour"],
        target=behavior["target"], adv_string="",
    )
    input_ids = sm.get_input_ids(adv_string="").to(model.device)

    samples = []
    for _ in range(n):
        text = generate_extended(
            model, tokenizer, input_ids, sm._assistant_role_slice,
            max_new_tokens=256, temperature=0.7, do_sample=True,
        )
        samples.append(text)
    return samples


def run_condition_fcd(model, tokenizer, behavior, bid, params, refusal_ids_list, lam):
    """Run F-C+D attack for one behavior."""
    hooks = {
        "on_step_end": make_arca_hook(update_interval=50, max_target_tokens=30),
        "early_stop_fn": make_content_early_stop(bid),
        "custom_loss_fn": make_hybrid_ce_refusal_loss_fn(refusal_ids_list, lam=lam),
    }
    slot_params = {
        **params,
        "id": bid,
        "behaviors_config": DEFAULT_CONFIG,
    }
    return run_slot_attack_with_model(
        model=model, tokenizer=tokenizer, params=slot_params,
        skip_smoothllm=True, skip_plots=True, hooks=hooks,
    )


def run_condition_base_igcg(model, tokenizer, bid, params):
    """Run base I-GCG attack for one behavior."""
    base_params = {
        **params,
        "id": bid,
        "behaviors_config": DEFAULT_CONFIG,
    }
    return run_attack_with_model(
        method_name="base_igcg",
        select_candidate=_argmin_clean_loss,
        model=model, tokenizer=tokenizer,
        params=base_params,
        skip_smoothllm=True, skip_plots=True,
    )


def run_condition_slot_igcg(model, tokenizer, bid, params):
    """Run Slot I-GCG (no hooks — vanilla SlotGCG + k-merge)."""
    slot_params = {
        **params,
        "id": bid,
        "behaviors_config": DEFAULT_CONFIG,
    }
    return run_slot_attack_with_model(
        model=model, tokenizer=tokenizer, params=slot_params,
        skip_smoothllm=True, skip_plots=True, hooks=None,
    )


def extract_attacked_prompt(summary: Dict, behavior: Dict) -> str | None:
    """Get the full interleaved/suffixed prompt from the attack summary."""
    if "final_interleaved_prompt" in summary:
        return summary["final_interleaved_prompt"]
    if "final_suffix" in summary:
        return behavior["behaviour"] + " " + summary["final_suffix"]
    return None


def main():
    parser = argparse.ArgumentParser(description="Geometry eval — smoke test")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--gold_path", type=str, default=DEFAULT_GOLD)
    parser.add_argument("--embed_model", type=str, default=EMBED_MODEL)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path(args.output_dir or f"output/geometry_eval/smoke/{ts}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Geometry Eval — Smoke Test ===")
    print(f"  BIDs:      {SMOKE_BIDS}")
    print(f"  Steps:     {SMOKE_STEPS}")
    print(f"  N-gen:     {N_GEN_ATTACKED} attacked, {N_GEN_REFUSED} refused")
    print(f"  Output:    {output_dir}")
    print()

    print("Loading gold completions...")
    gold = load_gold_completions(args.gold_path)

    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer(
        args.model_path, low_cpu_mem_usage=True,
        use_cache=False, device=args.device,
    )
    refusal_ids_list = _build_refusal_ids(tokenizer)

    with open(DEFAULT_CONFIG, encoding="utf-8") as f:
        all_behaviors = json.load(f)

    CONDITIONS = ["fcd", "base_igcg", "slot_igcg"]
    all_results: Dict[str, Any] = {
        "metadata": {
            "timestamp": ts,
            "model_path": args.model_path,
            "embed_model": args.embed_model,
            "seed": args.seed,
            "smoke_bids": SMOKE_BIDS,
            "n_steps": SMOKE_STEPS,
            "n_gen_attacked": N_GEN_ATTACKED,
            "n_gen_refused": N_GEN_REFUSED,
        },
        "per_behavior": {},
    }

    for bid in SMOKE_BIDS:
        behavior = all_behaviors[bid - 1]
        bid_key = str(bid)
        cooperative_texts = gold.get(bid_key, [])
        if not cooperative_texts:
            print(f"  !! BID {bid}: no gold completions found, skipping")
            continue

        print(f"\n{'='*60}")
        print(f"  BID {bid}: {behavior['behaviour'][:70]}...")
        print(f"{'='*60}")

        # Generate refused samples (shared across conditions)
        print(f"  Generating {N_GEN_REFUSED} refused samples...")
        refused_texts = generate_refused_samples(
            model, tokenizer, behavior, N_GEN_REFUSED, TEMPLATE_NAME,
        )

        bid_results: Dict[str, Any] = {
            "instruction": behavior["behaviour"],
            "n_refused_samples": len(refused_texts),
            "n_cooperative_samples": len(cooperative_texts),
            "conditions": {},
        }

        for condition in CONDITIONS:
            print(f"\n  --- Condition: {condition} ---")
            t0 = time.time()

            output_path = str(output_dir / f"{condition}/id_{bid}")
            params = {
                **(SMOKE_SLOT_PARAMS if condition != "base_igcg" else SMOKE_BASE_PARAMS),
                "output_path": output_path,
            }

            try:
                if condition == "fcd":
                    summary = run_condition_fcd(
                        model, tokenizer, behavior, bid, params,
                        refusal_ids_list, DEFAULT_LAM,
                    )
                elif condition == "base_igcg":
                    summary = run_condition_base_igcg(model, tokenizer, bid, params)
                else:
                    summary = run_condition_slot_igcg(model, tokenizer, bid, params)
            except Exception as e:
                print(f"  !! Attack failed for {condition} BID {bid}: {e}")
                bid_results["conditions"][condition] = {"error": str(e)}
                continue

            elapsed = time.time() - t0
            converged = summary.get("converged", False)
            print(f"  Attack: {'CONVERGED' if converged else 'NOT converged'} "
                  f"in {summary.get('total_steps', 0)} steps ({elapsed:.0f}s)")

            # Generate attacked samples
            prompt = extract_attacked_prompt(summary, behavior)
            if prompt:
                conv = load_conversation_template(TEMPLATE_NAME)
                sm = SuffixManager(
                    tokenizer=tokenizer, conv_template=conv,
                    instruction=prompt, target=behavior["target"], adv_string="",
                )
                input_ids = sm.get_input_ids(adv_string="").to(model.device)
                attacked_texts = [
                    generate_extended(
                        model, tokenizer, input_ids, sm._assistant_role_slice,
                        max_new_tokens=256, temperature=0.7, do_sample=True,
                    )
                    for _ in range(N_GEN_ATTACKED)
                ]
            else:
                gen_str = summary.get("gen_str_short", summary.get("gen_str", ""))
                attacked_texts = [gen_str] if gen_str else []

            if len(attacked_texts) < 3:
                print(f"  !! Only {len(attacked_texts)} attacked samples, skipping metrics")
                bid_results["conditions"][condition] = {
                    "converged": converged,
                    "n_attacked_samples": len(attacked_texts),
                    "error": "insufficient samples",
                }
                continue

            print(f"  Computing metrics ({len(attacked_texts)} atk, "
                  f"{len(refused_texts)} ref, {len(cooperative_texts)} coop)...")

            metrics = evaluate_behavior(
                behavior_id=bid,
                attacked_texts=attacked_texts,
                refused_texts=refused_texts,
                cooperative_texts=cooperative_texts[:100],
                embed_model=args.embed_model,
                compute_mauve=False,
                compute_sr=False,
            )

            bid_results["conditions"][condition] = {
                "converged": converged,
                "total_steps": summary.get("total_steps", 0),
                "n_attacked_samples": len(attacked_texts),
                "metrics": metrics,
                "attacked_texts_sample": attacked_texts[:3],
            }

            gc.collect()
            torch.cuda.empty_cache()

        all_results["per_behavior"][bid_key] = bid_results

    # Write report
    report_path = output_dir / "geometry_report.json"
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n=== Report written to {report_path} ===")

    # Print summary table
    print(f"\n{'BID':>4}  {'Condition':<12}  {'Conv':>4}  {'SemASR':>7}  "
          f"{'GeoASR':>7}  {'FBD(coop)':>10}  {'MMD(coop)':>10}")
    print("-" * 72)
    for bid_key, br in all_results["per_behavior"].items():
        for cond, cr in br.get("conditions", {}).items():
            m = cr.get("metrics", {})
            sem = m.get("semantic_asr", {}).get("mean", float("nan"))
            geo = m.get("geometric_asr", {}).get("geometric_asr", float("nan"))
            fbd = m.get("fbd_atk_coop", float("nan"))
            mmd = m.get("mmd_atk_coop", float("nan"))
            conv = "Y" if cr.get("converged") else "N"
            print(f"{bid_key:>4}  {cond:<12}  {conv:>4}  {sem:>7.3f}  "
                  f"{geo:>7.3f}  {fbd:>10.2f}  {mmd:>10.4f}")


if __name__ == "__main__":
    main()
