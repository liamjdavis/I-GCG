#!/usr/bin/env python3
"""Full semantic/distributional geometry evaluation on all 40 behaviors.

Runs 40 behaviors x (up to 500 steps) x 3 conditions, generates 50 attacked
+ 50 refused samples per behavior-condition pair, then computes all metrics
including MAUVE and optional strongREJECT.

Usage
-----
    # Full run (~36 h on A100):
    python scripts/geometry_eval_full.py --model_path /workspace/models/Qwen2-7B-Instruct

    # Quick run (5 BIDs, 200 steps):
    python scripts/geometry_eval_full.py --model_path $MODEL_PATH --quick

    # Dry run (1 BID, 5 steps):
    python scripts/geometry_eval_full.py --model_path $MODEL_PATH --dry_run

    # Skip conditions already completed in prior run:
    python scripts/geometry_eval_full.py --model_path $MODEL_PATH --resume
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
from typing import Any, Dict, List, Optional

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
from scripts.fcd_attack_report import (
    CONTENT_KEYWORDS,
    STRICT_REFUSAL_PHRASES,
    TEMPLATE_NAME,
    V2_ALL40_CONFIG,
    _build_refusal_ids,
    generate_extended,
    make_arca_hook,
    make_content_early_stop,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEED = 235711
DEFAULT_GOLD = "data/gold_harmful_completions.json"
DEFAULT_LAM = 0.2

FULL_SLOT_PARAMS: Dict[str, Any] = {
    "num_steps": 500,
    "search_width": 256,
    "top_k": 256,
    "num_adv_tokens": 20,
    "attention_temp": 8.0,
    "kmerge_k": 7,
    "eval_batch_size": 64,
    "use_prefix_cache": True,
    "adv_string_init": "!",
    "seed": SEED,
}

FULL_BASE_PARAMS: Dict[str, Any] = {
    "num_steps": 500,
    "batch_size": 256,
    "top_k": 256,
    "seed": SEED,
    "n_flips": 1,
}

QUICK_SLOT_PARAMS: Dict[str, Any] = {**FULL_SLOT_PARAMS, "num_steps": 200, "search_width": 256}
QUICK_BASE_PARAMS: Dict[str, Any] = {**FULL_BASE_PARAMS, "num_steps": 200}

DRY_SLOT_PARAMS: Dict[str, Any] = {
    "num_steps": 5, "search_width": 64, "top_k": 64,
    "num_adv_tokens": 20, "attention_temp": 8.0, "kmerge_k": 3,
    "eval_batch_size": 16, "use_prefix_cache": True,
    "adv_string_init": "!", "seed": SEED,
}
DRY_BASE_PARAMS: Dict[str, Any] = {
    "num_steps": 5, "batch_size": 64, "top_k": 64, "seed": SEED, "n_flips": 1,
}

OOM_FALLBACK_SEARCH_WIDTH = 128
N_GEN_ATTACKED = 50
N_GEN_REFUSED = 50
CONDITIONS = ["fcd", "base_igcg", "slot_igcg"]
EMBED_MODEL = "all-MiniLM-L6-v2"


def _argmin_clean_loss(new_adv_suffix, clean_losses, **kw):
    return new_adv_suffix[clean_losses.argmin()], 0.0


def generate_refused_samples(model, tokenizer, behavior, n, template_name):
    conv = load_conversation_template(template_name)
    sm = SuffixManager(
        tokenizer=tokenizer, conv_template=conv,
        instruction=behavior["behaviour"],
        target=behavior["target"], adv_string="",
    )
    input_ids = sm.get_input_ids(adv_string="").to(model.device)
    return [
        generate_extended(
            model, tokenizer, input_ids, sm._assistant_role_slice,
            max_new_tokens=256, temperature=0.7, do_sample=True,
        )
        for _ in range(n)
    ]


def run_attack(condition, model, tokenizer, bid, behavior, params,
               refusal_ids_list, lam, output_path):
    """Run one attack condition with OOM fallback."""
    params = {**params, "output_path": output_path}

    for attempt in range(2):
        try:
            if condition == "fcd":
                hooks = {
                    "on_step_end": make_arca_hook(update_interval=50, max_target_tokens=30),
                    "early_stop_fn": make_content_early_stop(bid),
                    "custom_loss_fn": make_hybrid_ce_refusal_loss_fn(refusal_ids_list, lam=lam),
                }
                p = {**params, "id": bid, "behaviors_config": V2_ALL40_CONFIG}
                return run_slot_attack_with_model(
                    model=model, tokenizer=tokenizer, params=p,
                    skip_smoothllm=True, skip_plots=True, hooks=hooks,
                )
            elif condition == "base_igcg":
                p = {**params, "id": bid, "behaviors_config": V2_ALL40_CONFIG}
                return run_attack_with_model(
                    method_name="base_igcg",
                    select_candidate=_argmin_clean_loss,
                    model=model, tokenizer=tokenizer, params=p,
                    skip_smoothllm=True, skip_plots=True,
                )
            else:  # slot_igcg
                p = {**params, "id": bid, "behaviors_config": V2_ALL40_CONFIG}
                return run_slot_attack_with_model(
                    model=model, tokenizer=tokenizer, params=p,
                    skip_smoothllm=True, skip_plots=True, hooks=None,
                )
        except torch.cuda.OutOfMemoryError:
            if attempt == 0:
                print(f"  !! OOM, retrying with reduced search width")
                gc.collect()
                torch.cuda.empty_cache()
                params = {
                    **params,
                    "search_width": OOM_FALLBACK_SEARCH_WIDTH,
                    "eval_batch_size": 16, "kmerge_k": 3,
                    "batch_size": OOM_FALLBACK_SEARCH_WIDTH,
                }
            else:
                raise

    return None


def extract_attacked_prompt(summary, behavior):
    if "final_interleaved_prompt" in summary:
        return summary["final_interleaved_prompt"]
    if "final_suffix" in summary:
        return behavior["behaviour"] + " " + summary["final_suffix"]
    return None


def load_checkpoint(output_dir: Path) -> Optional[Dict]:
    report_path = output_dir / "geometry_report.json"
    if report_path.exists():
        with open(report_path) as f:
            return json.load(f)
    return None


def main():
    parser = argparse.ArgumentParser(description="Geometry eval — full experiment")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--gold_path", type=str, default=DEFAULT_GOLD)
    parser.add_argument("--embed_model", type=str, default=EMBED_MODEL)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--quick", action="store_true", help="5 BIDs, 200 steps")
    parser.add_argument("--dry_run", action="store_true", help="1 BID, 5 steps")
    parser.add_argument("--resume", action="store_true", help="Skip completed conditions")
    parser.add_argument("--enable_strongreject", action="store_true")
    parser.add_argument("--n_gen_attacked", type=int, default=N_GEN_ATTACKED)
    parser.add_argument("--n_gen_refused", type=int, default=N_GEN_REFUSED)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.dry_run:
        behavior_ids = [3]
        slot_params = DRY_SLOT_PARAMS.copy()
        base_params = DRY_BASE_PARAMS.copy()
        n_attacked = 5
        n_refused = 5
        tag = "dry"
    elif args.quick:
        behavior_ids = [3, 8, 15, 25, 32]
        slot_params = QUICK_SLOT_PARAMS.copy()
        base_params = QUICK_BASE_PARAMS.copy()
        n_attacked = 20
        n_refused = 20
        tag = "quick"
    else:
        behavior_ids = list(range(1, 41))
        slot_params = FULL_SLOT_PARAMS.copy()
        base_params = FULL_BASE_PARAMS.copy()
        n_attacked = args.n_gen_attacked
        n_refused = args.n_gen_refused
        tag = "full"

    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path(args.output_dir or f"output/geometry_eval/{tag}/{ts}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Geometry Eval — {tag.upper()} ===")
    print(f"  BIDs:       {len(behavior_ids)}")
    print(f"  Steps:      {slot_params['num_steps']}")
    print(f"  N-gen:      {n_attacked} attacked, {n_refused} refused")
    print(f"  Conditions: {CONDITIONS}")
    print(f"  MAUVE:      {'yes' if n_attacked >= 20 else 'no (too few samples)'}")
    print(f"  strongREJECT: {'yes' if args.enable_strongreject else 'no'}")
    print(f"  Output:     {output_dir}")
    print()

    print("Loading gold completions...")
    gold = load_gold_completions(args.gold_path)

    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer(
        args.model_path, low_cpu_mem_usage=True,
        use_cache=False, device=args.device,
    )
    refusal_ids_list = _build_refusal_ids(tokenizer)

    with open(V2_ALL40_CONFIG) as f:
        all_behaviors = json.load(f)

    # Resume support
    prior = load_checkpoint(output_dir) if args.resume else None
    prior_beh = prior.get("per_behavior", {}) if prior else {}

    all_results: Dict[str, Any] = {
        "metadata": {
            "timestamp": ts,
            "tag": tag,
            "model_path": args.model_path,
            "embed_model": args.embed_model,
            "seed": args.seed,
            "behavior_ids": behavior_ids,
            "n_steps": slot_params["num_steps"],
            "n_gen_attacked": n_attacked,
            "n_gen_refused": n_refused,
            "enable_strongreject": args.enable_strongreject,
        },
        "per_behavior": dict(prior_beh),
    }

    t_all = time.time()

    for bid in behavior_ids:
        behavior = all_behaviors[bid - 1]
        bid_key = str(bid)
        cooperative_texts = gold.get(bid_key, [])
        if not cooperative_texts:
            print(f"\n  !! BID {bid}: no gold completions, skipping")
            continue

        print(f"\n{'='*72}")
        print(f"  BID {bid}: {behavior['behaviour'][:65]}...")
        print(f"{'='*72}")

        # Reuse refused samples across conditions
        print(f"  Generating {n_refused} refused samples...")
        refused_texts = generate_refused_samples(
            model, tokenizer, behavior, n_refused, TEMPLATE_NAME,
        )

        bid_results = all_results["per_behavior"].get(bid_key, {
            "instruction": behavior["behaviour"],
            "n_refused_samples": len(refused_texts),
            "n_cooperative_samples": len(cooperative_texts),
            "conditions": {},
        })

        for condition in CONDITIONS:
            if condition in bid_results.get("conditions", {}) and args.resume:
                existing = bid_results["conditions"][condition]
                if existing.get("metrics"):
                    print(f"  [{condition}] already done, skipping (resume)")
                    continue

            print(f"\n  --- {condition.upper()} ---")
            t0 = time.time()

            opath = str(output_dir / f"{condition}/id_{bid}")
            params = slot_params if condition != "base_igcg" else base_params

            try:
                summary = run_attack(
                    condition, model, tokenizer, bid, behavior, params,
                    refusal_ids_list, DEFAULT_LAM, opath,
                )
            except Exception as e:
                print(f"  !! Attack failed: {e}")
                bid_results.setdefault("conditions", {})[condition] = {"error": str(e)}
                continue

            if summary is None:
                bid_results.setdefault("conditions", {})[condition] = {"error": "OOM"}
                continue

            elapsed = time.time() - t0
            converged = summary.get("converged", False)
            print(f"  Attack: {'CONVERGED' if converged else 'NOT converged'} "
                  f"in {summary.get('total_steps', 0)} steps ({elapsed:.0f}s)")

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
                    for _ in range(n_attacked)
                ]
            else:
                gen_str = summary.get("gen_str_short", summary.get("gen_str", ""))
                attacked_texts = [gen_str] if gen_str else []

            if len(attacked_texts) < 3:
                print(f"  !! Only {len(attacked_texts)} attacked samples, skipping metrics")
                bid_results.setdefault("conditions", {})[condition] = {
                    "converged": converged,
                    "n_attacked_samples": len(attacked_texts),
                    "error": "insufficient samples",
                }
                continue

            use_mauve = n_attacked >= 20 and len(cooperative_texts) >= 20
            print(f"  Computing metrics ({len(attacked_texts)} atk, "
                  f"{len(refused_texts)} ref, {len(cooperative_texts)} coop)"
                  f"{' +MAUVE' if use_mauve else ''}"
                  f"{' +strongREJECT' if args.enable_strongreject else ''}...")

            metrics = evaluate_behavior(
                behavior_id=bid,
                attacked_texts=attacked_texts,
                refused_texts=refused_texts,
                cooperative_texts=cooperative_texts[:100],
                embed_model=args.embed_model,
                compute_mauve=use_mauve,
                compute_sr=args.enable_strongreject,
                behavior_instruction=behavior["behaviour"],
                device_id=args.device,
            )

            bid_results.setdefault("conditions", {})[condition] = {
                "converged": converged,
                "total_steps": summary.get("total_steps", 0),
                "n_attacked_samples": len(attacked_texts),
                "metrics": metrics,
                "attacked_texts_sample": attacked_texts[:3],
            }

            gc.collect()
            torch.cuda.empty_cache()

        all_results["per_behavior"][bid_key] = bid_results

        # Checkpoint after each behavior
        report_path = output_dir / "geometry_report.json"
        with open(report_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    total_time = time.time() - t_all
    all_results["metadata"]["total_wall_time_s"] = total_time

    report_path = output_dir / "geometry_report.json"
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'='*72}")
    print(f"  DONE — {total_time:.0f}s total ({total_time/3600:.1f}h)")
    print(f"  Report: {report_path}")
    print(f"{'='*72}")

    # Summary table
    print(f"\n{'BID':>4}  {'Cond':<12}  {'Conv':>4}  {'SemASR':>7}  "
          f"{'GeoASR':>7}  {'FBD_c':>10}  {'MMD_c':>10}  {'FR_c':>8}")
    print("-" * 80)
    for bid_key in sorted(all_results["per_behavior"], key=lambda x: int(x)):
        br = all_results["per_behavior"][bid_key]
        for cond, cr in br.get("conditions", {}).items():
            m = cr.get("metrics", {})
            sem = m.get("semantic_asr", {}).get("mean", float("nan"))
            geo = m.get("geometric_asr", {}).get("geometric_asr", float("nan"))
            fbd = m.get("fbd_atk_coop", float("nan"))
            mmd = m.get("mmd_atk_coop", float("nan"))
            fr = m.get("fisher_rao_atk_coop", float("nan"))
            conv = "Y" if cr.get("converged") else "N"
            print(f"{bid_key:>4}  {cond:<12}  {conv:>4}  {sem:>7.3f}  "
                  f"{geo:>7.3f}  {fbd:>10.2f}  {mmd:>10.4f}  {fr:>8.2f}")


if __name__ == "__main__":
    main()
