#!/usr/bin/env python3
"""Full AdvBench geometry evaluation (F-C+D + SlotGCG) — cluster-ready.

Runs two attack conditions over the 520-behaviour AdvBench split against an
aligned target model, computes prefix/strict ASR + Sem/Geo/FBD/MMD per
behaviour, and checkpoints per-condition so SLURM array tasks can be
interrupted and resumed.

Conditions
----------
* ``fcd``       — F-C+D (Slot + K-merge + ARCA + content early-stop + hybrid
                  CE/refusal loss).
* ``slot_igcg`` — vanilla Slot + K-merge (no F-C+D hooks).

base_igcg is intentionally **excluded** from this run.

Metrics
-------
* ``prefix_asr``, ``strict_asr``  — fraction of attacked samples that pass
  the prefix (``_is_jailbroken``) and strict-refusal filters from
  ``scripts/fcd_attack_report.classify_generation``.  ``content_asr`` is
  **not** reported for AdvBench: the 40-behaviour hand-curated keyword
  dict doesn't cover these 520 behaviours, and deriving per-behaviour
  keywords is out-of-scope for this evaluator.
* ``semantic_asr``, ``geometric_asr``, ``fbd_atk_coop``, ``mmd_atk_coop``,
  plus Fisher-Rao, sliced-Wasserstein, intrinsic dim — from
  ``robust_gcg.semantic_eval.evaluate_behavior``.

Sharding
--------
``--shard_rank K --shard_world_size N`` processes BIDs ``[K::N]``.  Each
shard writes::

    output/geometry_eval/advbench/<ts>/
        geometry_report_shard{K:02d}of{N:02d}.json
        fcd/id_<bid>/summary_<bid>.json, log_<bid>.json
        slot_igcg/id_<bid>/...

Running the same command with a different rank on another GPU is safe; runs
are disjoint by construction.  A final ``--merge_only`` pass glues shard
reports into one ``geometry_report.json``.

Smoke mode (``--dry_run``)
--------------------------
Shard 0 only, 5 BIDs, 5 steps, 10 attacked + 10 refused samples, same
reduced-batch params as the Windows-24GB smoke (``SMOKE_BASE_PARAMS``
analogue).  Completes in ~10 min on one A100 and produces the full
artefact tree so the merge path can be exercised.

Usage
-----
    # Full run, 1 shard/GPU on SLURM array:
    python scripts/geometry_eval_advbench.py \
        --aligned_model Qwen/Qwen2.5-7B-Instruct \
        --gold_path data/gold_harmful_completions_advbench.json \
        --shard_rank $SLURM_ARRAY_TASK_ID --shard_world_size 8 \
        --output_dir output/geometry_eval/advbench/$SLURM_JOB_ID

    # Single-node smoke:
    python scripts/geometry_eval_advbench.py --dry_run \
        --aligned_model Qwen/Qwen2.5-7B-Instruct \
        --gold_path data/gold_harmful_completions_advbench.json

    # Merge shard reports:
    python scripts/geometry_eval_advbench.py --merge_only \
        --output_dir output/geometry_eval/advbench/<timestamp> \
        --shard_world_size 8
"""
from __future__ import annotations

import argparse
import datetime
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from llm_attacks.minimal_gcg.opt_utils import load_model_and_tokenizer
from llm_attacks.minimal_gcg.string_utils import (
    SuffixManager, load_conversation_template,
)
from llm_attacks.minimal_gcg.slot_utils import make_hybrid_ce_refusal_loss_fn
from robust_gcg.semantic_eval import evaluate_behavior, load_gold_completions
from scripts.robust_gcg_F_slot_kmerge import run_slot_attack_with_model
from scripts.fcd_attack_report import (
    TEMPLATE_NAME,
    _build_refusal_ids,
    classify_generation,
    generate_extended,
    make_arca_hook,
    make_content_early_stop,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEED = 235711
DEFAULT_ALIGNED = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_CONFIG = "data/advbench_behaviors.json"
DEFAULT_GOLD = "data/gold_harmful_completions_advbench.json"
DEFAULT_LAM = 0.2
EMBED_MODEL = "all-MiniLM-L6-v2"

# Attack conditions actually run by this script.  base_igcg is intentionally
# dropped; re-add it here and mirror geometry_eval_full.run_attack if needed.
CONDITIONS = ["fcd", "slot_igcg"]

# --- Full-experiment params (A100 80GB, Qwen2.5-7B-Instruct) ----------------
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

# --- Smoke / dry-run params (mirrors what worked on a 24GB GPU) -------------
DRY_SLOT_PARAMS: Dict[str, Any] = {
    "num_steps": 5,
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

OOM_FALLBACK = {
    "search_width": 128,
    "eval_batch_size": 16,
    "kmerge_k": 3,
}

N_GEN_ATTACKED = 50
N_GEN_REFUSED = 50

# Smoke sample counts.
DRY_N_ATTACKED = 10
DRY_N_REFUSED = 10
DRY_BIDS = [1, 2, 3, 4, 5]


# ---------------------------------------------------------------------------
# Sharding
# ---------------------------------------------------------------------------

def shard_ids(all_ids: List[int], rank: int, world_size: int) -> List[int]:
    if world_size <= 0:
        raise ValueError("shard_world_size must be >= 1")
    if not (0 <= rank < world_size):
        raise ValueError(f"shard_rank must be in [0, {world_size}), got {rank}")
    return all_ids[rank::world_size]


def shard_report_path(output_dir: Path, rank: int, world_size: int) -> Path:
    return output_dir / f"geometry_report_shard{rank:02d}of{world_size:02d}.json"


# ---------------------------------------------------------------------------
# Attack dispatch (fcd / slot_igcg; OOM-aware)
# ---------------------------------------------------------------------------

def _run_condition(
    condition: str,
    model, tokenizer, bid: int,
    behaviors_config: str,
    params: Dict[str, Any],
    refusal_ids_list, lam: float,
    output_path: str,
) -> Optional[Dict[str, Any]]:
    """Dispatch a single (bid, condition) attack with one OOM retry."""
    base = {**params, "output_path": output_path, "id": bid,
            "behaviors_config": behaviors_config}

    for attempt in range(2):
        try:
            if condition == "fcd":
                hooks = {
                    "on_step_end": make_arca_hook(
                        update_interval=50, max_target_tokens=30,
                    ),
                    # NB: make_content_early_stop looks up CONTENT_KEYWORDS
                    # by BID.  For AdvBench BIDs (1..520) the dict returns
                    # [], so the early-stop reduces to prefix+strict only.
                    "early_stop_fn": make_content_early_stop(bid),
                    "custom_loss_fn": make_hybrid_ce_refusal_loss_fn(
                        refusal_ids_list, lam=lam,
                    ),
                }
                return run_slot_attack_with_model(
                    model=model, tokenizer=tokenizer, params=base,
                    skip_smoothllm=True, skip_plots=True, hooks=hooks,
                )
            elif condition == "slot_igcg":
                return run_slot_attack_with_model(
                    model=model, tokenizer=tokenizer, params=base,
                    skip_smoothllm=True, skip_plots=True, hooks=None,
                )
            else:
                raise ValueError(f"Unknown condition: {condition}")
        except torch.cuda.OutOfMemoryError:
            if attempt == 0:
                print(f"  !! OOM, retrying with reduced search width / kmerge_k")
                gc.collect()
                torch.cuda.empty_cache()
                base = {**base, **OOM_FALLBACK}
            else:
                raise

    return None


# ---------------------------------------------------------------------------
# Refused / attacked sample generation
# ---------------------------------------------------------------------------

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


def generate_attacked_samples(model, tokenizer, prompt, behavior, n):
    conv = load_conversation_template(TEMPLATE_NAME)
    sm = SuffixManager(
        tokenizer=tokenizer, conv_template=conv,
        instruction=prompt, target=behavior["target"], adv_string="",
    )
    input_ids = sm.get_input_ids(adv_string="").to(model.device)
    return [
        generate_extended(
            model, tokenizer, input_ids, sm._assistant_role_slice,
            max_new_tokens=256, temperature=0.7, do_sample=True,
        )
        for _ in range(n)
    ]


def extract_attacked_prompt(summary, behavior):
    if "final_interleaved_prompt" in summary:
        return summary["final_interleaved_prompt"]
    if "final_suffix" in summary:
        return behavior["behaviour"] + " " + summary["final_suffix"]
    return None


# ---------------------------------------------------------------------------
# ASR summaries (prefix + strict) per batch of attacked generations
# ---------------------------------------------------------------------------

def summarise_asr(attacked_texts: List[str], bid: int) -> Dict[str, Any]:
    """Fraction of attacked generations passing prefix / strict filters.

    We reuse ``classify_generation`` from ``scripts/fcd_attack_report.py``
    and **ignore** its ``content_asr`` field because CONTENT_KEYWORDS is
    empty for AdvBench BIDs.
    """
    n = len(attacked_texts)
    if n == 0:
        return {"n": 0, "prefix_asr": 0.0, "strict_asr": 0.0}

    n_prefix = 0
    n_strict = 0
    for g in attacked_texts:
        c = classify_generation(g, bid)
        n_prefix += int(bool(c.get("prefix_asr", False)))
        n_strict += int(bool(c.get("strict_asr", False)))
    return {
        "n": n,
        "n_prefix": n_prefix,
        "n_strict": n_strict,
        "prefix_asr": n_prefix / n,
        "strict_asr": n_strict / n,
    }


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------

def load_shard_report(path: Path) -> Dict[str, Any]:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"metadata": {}, "per_behavior": {}}


def condition_complete(output_dir: Path, condition: str, bid: int) -> bool:
    """True iff the attack summary for (condition, bid) already exists."""
    return (output_dir / condition / f"id_{bid}" / f"summary_{bid}.json").exists()


# ---------------------------------------------------------------------------
# Cross-shard merge
# ---------------------------------------------------------------------------

def merge_shards(output_dir: Path, world_size: int) -> Path:
    per_behavior: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {}
    missing: List[int] = []

    for rank in range(world_size):
        p = shard_report_path(output_dir, rank, world_size)
        if not p.exists():
            missing.append(rank)
            continue
        with open(p) as f:
            data = json.load(f)
        metadata.update(data.get("metadata", {}))
        for k, v in data.get("per_behavior", {}).items():
            if k in per_behavior:
                print(f"  !! duplicate BID {k} across shards, keeping first")
                continue
            per_behavior[k] = v

    if missing:
        print(f"  !! missing shards: {missing} — merged report is INCOMPLETE")

    merged = {
        "metadata": metadata,
        "per_behavior": {
            k: per_behavior[k]
            for k in sorted(per_behavior.keys(), key=lambda x: int(x))
        },
    }
    out_path = output_dir / "geometry_report.json"
    with open(out_path, "w") as f:
        json.dump(merged, f, indent=2, default=str)
    print(f"Merged {len(merged['per_behavior'])} behaviors -> {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Summary table printer
# ---------------------------------------------------------------------------

def print_summary(report: Dict[str, Any]) -> None:
    print(f"\n{'BID':>4}  {'Cond':<10}  {'Conv':>4}  {'PfxASR':>7}  "
          f"{'StrASR':>7}  {'SemASR':>7}  {'GeoASR':>7}  "
          f"{'FBD_c':>9}  {'MMD_c':>9}")
    print("-" * 84)
    per = report.get("per_behavior", {})
    for bid_key in sorted(per.keys(), key=lambda x: int(x)):
        br = per[bid_key]
        for cond, cr in br.get("conditions", {}).items():
            if "error" in cr:
                print(f"{bid_key:>4}  {cond:<10}  {'ERR':>4}  "
                      f"error: {cr['error'][:40]}")
                continue
            asr = cr.get("asr_summary", {})
            m = cr.get("metrics", {})
            sem = m.get("semantic_asr", {}).get("mean", float("nan"))
            geo = m.get("geometric_asr", {}).get("geometric_asr", float("nan"))
            fbd = m.get("fbd_atk_coop", float("nan"))
            mmd = m.get("mmd_atk_coop", float("nan"))
            conv = "Y" if cr.get("converged") else "N"
            print(f"{bid_key:>4}  {cond:<10}  {conv:>4}  "
                  f"{asr.get('prefix_asr', float('nan')):>7.3f}  "
                  f"{asr.get('strict_asr', float('nan')):>7.3f}  "
                  f"{sem:>7.3f}  {geo:>7.3f}  {fbd:>9.2f}  {mmd:>9.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--aligned_model", type=str, default=DEFAULT_ALIGNED,
                        help="HuggingFace id or local path of the aligned "
                             "(victim) model to attack")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--behaviors_config", type=str, default=DEFAULT_CONFIG)
    parser.add_argument("--gold_path", type=str, default=DEFAULT_GOLD)
    parser.add_argument("--embed_model", type=str, default=EMBED_MODEL)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=SEED)

    parser.add_argument("--shard_rank", type=int, default=0)
    parser.add_argument("--shard_world_size", type=int, default=1)

    parser.add_argument("--n_gen_attacked", type=int, default=N_GEN_ATTACKED)
    parser.add_argument("--n_gen_refused", type=int, default=N_GEN_REFUSED)
    parser.add_argument("--num_steps", type=int, default=None,
                        help="Override FULL_SLOT_PARAMS['num_steps']")
    parser.add_argument("--search_width", type=int, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=None)

    parser.add_argument("--dry_run", action="store_true",
                        help="Shard 0 only, 5 BIDs, 5 steps, 10 samples. "
                             "Uses reduced-batch params compatible with 24GB.")
    parser.add_argument("--merge_only", action="store_true",
                        help="Skip attacks; merge per-shard reports in "
                             "--output_dir")
    parser.add_argument("--enable_mauve", action="store_true")
    parser.add_argument("--enable_strongreject", action="store_true")
    parser.add_argument("--enable_judge", action="store_true",
                        help="Run robust_gcg.judge_hooks.score_attacked on "
                             "each condition's attacked generations")

    args = parser.parse_args()

    # --- Merge-only path ---------------------------------------------------
    if args.merge_only:
        if not args.output_dir:
            raise SystemExit("--merge_only requires --output_dir")
        merge_shards(Path(args.output_dir), args.shard_world_size)
        return

    # --- Seed --------------------------------------------------------------
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # --- Choose params + bid list -----------------------------------------
    if args.dry_run:
        if args.shard_rank != 0 or args.shard_world_size != 1:
            print("  !! --dry_run forces single-shard; ignoring --shard_*")
        slot_params = DRY_SLOT_PARAMS.copy()
        n_attacked = DRY_N_ATTACKED
        n_refused = DRY_N_REFUSED
        rank, world = 0, 1
        tag = "dry"
    else:
        slot_params = FULL_SLOT_PARAMS.copy()
        n_attacked = args.n_gen_attacked
        n_refused = args.n_gen_refused
        rank, world = args.shard_rank, args.shard_world_size
        tag = "advbench"

    # CLI overrides for tuning on the cluster.
    if args.num_steps is not None:
        slot_params["num_steps"] = args.num_steps
    if args.search_width is not None:
        slot_params["search_width"] = args.search_width
    if args.eval_batch_size is not None:
        slot_params["eval_batch_size"] = args.eval_batch_size

    # --- Load behaviours ---------------------------------------------------
    with open(args.behaviors_config) as f:
        all_behaviors = json.load(f)
    n_total = len(all_behaviors)
    all_ids = list(range(1, n_total + 1))

    if args.dry_run:
        behavior_ids = [b for b in DRY_BIDS if 1 <= b <= n_total]
    else:
        behavior_ids = shard_ids(all_ids, rank, world)

    # --- Output dir --------------------------------------------------------
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path(args.output_dir or f"output/geometry_eval/{tag}/{ts}")
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = shard_report_path(output_dir, rank, world)

    # --- Banner ------------------------------------------------------------
    print(f"=== Geometry Eval (AdvBench) | shard {rank}/{world} | {tag} ===")
    print(f"  Aligned model: {args.aligned_model}")
    print(f"  Config:        {args.behaviors_config}  (N={n_total})")
    print(f"  Gold:          {args.gold_path}")
    print(f"  BIDs:          {len(behavior_ids)} "
          f"({behavior_ids[:3]}...{behavior_ids[-3:] if len(behavior_ids) > 3 else ''})")
    print(f"  Conditions:    {CONDITIONS}")
    print(f"  Steps:         {slot_params['num_steps']}")
    print(f"  N-gen:         {n_attacked} attacked, {n_refused} refused")
    print(f"  MAUVE:         {'yes' if args.enable_mauve and n_attacked >= 20 else 'no'}")
    print(f"  strongREJECT:  {'yes' if args.enable_strongreject else 'no'}")
    print(f"  Judge hook:    {'yes' if args.enable_judge else 'no'}")
    print(f"  Output:        {output_dir}")
    print(f"  Shard report:  {report_path}")
    print()

    # --- Load gold completions --------------------------------------------
    print("Loading gold completions...")
    gold = load_gold_completions(args.gold_path)
    print(f"  {len(gold)} behaviours with gold completions\n")

    # --- Load aligned model -----------------------------------------------
    print(f"Loading aligned model: {args.aligned_model}")
    t0 = time.time()
    model, tokenizer = load_model_and_tokenizer(
        args.aligned_model, low_cpu_mem_usage=True,
        use_cache=False, device=args.device,
    )
    refusal_ids_list = _build_refusal_ids(tokenizer)
    print(f"  Model loaded in {time.time() - t0:.1f}s\n")

    # --- Load / init shard report -----------------------------------------
    shard_report = load_shard_report(report_path)
    shard_report.setdefault("per_behavior", {})
    shard_report["metadata"] = {
        **shard_report.get("metadata", {}),
        "timestamp": ts,
        "tag": tag,
        "aligned_model": args.aligned_model,
        "embed_model": args.embed_model,
        "seed": args.seed,
        "shard_rank": rank,
        "shard_world_size": world,
        "behavior_ids": behavior_ids,
        "n_steps": slot_params["num_steps"],
        "n_gen_attacked": n_attacked,
        "n_gen_refused": n_refused,
        "conditions": CONDITIONS,
        "judge_enabled": args.enable_judge,
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

        # Refused samples are shared across conditions (same aligned model).
        print(f"  Generating {n_refused} refused samples...")
        refused_texts = generate_refused_samples(
            model, tokenizer, behavior, n_refused, TEMPLATE_NAME,
        )

        bid_results = shard_report["per_behavior"].get(bid_key, {
            "instruction": behavior["behaviour"],
            "n_refused_samples": len(refused_texts),
            "n_cooperative_samples": len(cooperative_texts),
            "conditions": {},
        })
        bid_results.setdefault("conditions", {})

        for condition in CONDITIONS:
            existing = bid_results["conditions"].get(condition, {})
            if existing.get("metrics") and condition_complete(output_dir, condition, bid):
                print(f"  [{condition}] already done, skip")
                continue

            print(f"\n  --- {condition.upper()} ---")
            t0 = time.time()
            opath = str(output_dir / f"{condition}/id_{bid}")
            Path(opath).mkdir(parents=True, exist_ok=True)

            try:
                summary = _run_condition(
                    condition, model, tokenizer, bid,
                    args.behaviors_config, slot_params,
                    refusal_ids_list, DEFAULT_LAM, opath,
                )
            except Exception as e:
                print(f"  !! Attack failed: {e}")
                bid_results["conditions"][condition] = {"error": str(e)}
                continue

            if summary is None:
                bid_results["conditions"][condition] = {"error": "OOM"}
                continue

            elapsed = time.time() - t0
            converged = bool(summary.get("converged", False))
            total_steps = summary.get("total_steps", 0)
            print(f"  Attack: {'CONVERGED' if converged else 'NOT converged'} "
                  f"in {total_steps} steps ({elapsed:.0f}s)")

            # Generate attacked samples ------------------------------------
            prompt = extract_attacked_prompt(summary, behavior)
            if prompt:
                attacked_texts = generate_attacked_samples(
                    model, tokenizer, prompt, behavior, n_attacked,
                )
            else:
                gen_str = summary.get("gen_str_short", summary.get("gen_str", ""))
                attacked_texts = [gen_str] if gen_str else []

            if len(attacked_texts) < 3:
                print(f"  !! Only {len(attacked_texts)} attacked samples, "
                      f"skipping metrics")
                bid_results["conditions"][condition] = {
                    "converged": converged,
                    "total_steps": total_steps,
                    "n_attacked_samples": len(attacked_texts),
                    "error": "insufficient samples",
                }
                continue

            # Prefix / strict ASR ------------------------------------------
            asr_summary = summarise_asr(attacked_texts, bid)
            print(f"  ASR: prefix={asr_summary['prefix_asr']:.2f} "
                  f"strict={asr_summary['strict_asr']:.2f} "
                  f"(n={asr_summary['n']})")

            # Geometry metrics ---------------------------------------------
            use_mauve = (args.enable_mauve
                         and len(attacked_texts) >= 20
                         and len(cooperative_texts) >= 20)
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

            condition_record: Dict[str, Any] = {
                "converged": converged,
                "total_steps": total_steps,
                "n_attacked_samples": len(attacked_texts),
                "asr_summary": asr_summary,
                "metrics": metrics,
                "attacked_texts_sample": attacked_texts[:3],
            }

            # ================================================================
            # LLM-AS-JUDGE HOOK (attacked generations)
            # --------------------------------------------------------------
            # Enabled with `--enable_judge`.  To plug in a real judge,
            # implement `judge_generation` inside
            # `robust_gcg/judge_hooks.py` — no other file needs editing.
            # The judge summary lands at
            # per_behavior[bid].conditions[cond].judge in the final report.
            # ================================================================
            if args.enable_judge:
                from robust_gcg.judge_hooks import score_attacked
                condition_record["judge"] = score_attacked(
                    behavior, attacked_texts,
                    reference_completions=cooperative_texts[:10],
                )

            bid_results["conditions"][condition] = condition_record
            gc.collect()
            torch.cuda.empty_cache()

        # Persist after each behaviour so SLURM preemption is cheap.
        shard_report["per_behavior"][bid_key] = bid_results
        with open(report_path, "w") as f:
            json.dump(shard_report, f, indent=2, default=str)

    total_time = time.time() - t_all
    shard_report["metadata"]["total_wall_time_s"] = total_time

    with open(report_path, "w") as f:
        json.dump(shard_report, f, indent=2, default=str)

    print(f"\n{'='*72}")
    print(f"  Shard {rank}/{world} DONE — {total_time:.0f}s "
          f"({total_time/3600:.1f}h)")
    print(f"  Report: {report_path}")
    print(f"{'='*72}")

    print_summary(shard_report)


if __name__ == "__main__":
    main()
