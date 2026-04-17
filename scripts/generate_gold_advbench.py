#!/usr/bin/env python3
"""Generate gold (cooperative) completions for the full AdvBench dataset.

Shard-aware wrapper around the single-node implementation in
``scripts/generate_gold_completions.py``.  Each shard handles a strided
subset of the 520 AdvBench behaviours, loads one copy of the abliterated
Qwen2.5 model, writes its own checkpointable JSON, and a final
``--merge_only`` pass concatenates shards into one canonical file.

Intended cluster use
--------------------
SLURM array job (one task per GPU).  See
``scripts/slurm/advbench_gold.sbatch`` for a concrete template.  Nothing
about the script requires SLURM — you can also launch it yourself with
``CUDA_VISIBLE_DEVICES=K`` per process.

Usage
-----
    # One shard (called per SLURM array task, rank in [0, world_size)):
    python scripts/generate_gold_advbench.py \
        --shard_rank 0 --shard_world_size 8 \
        --behaviors_config data/advbench_behaviors.json \
        --output data/gold_harmful_completions_advbench.json

    # Merge per-shard JSONs into one canonical file:
    python scripts/generate_gold_advbench.py --merge_only \
        --shard_world_size 8 \
        --output data/gold_harmful_completions_advbench.json

Runtime (1x A100 80GB, bf16, 100 samples/behavior, max_new_tokens=512):
~10 s/behavior -> ~65 behaviors / shard on 8 GPUs -> ~11 min/shard.
"""
from __future__ import annotations

import argparse
import datetime
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

# Reuse the building blocks from the existing single-device script so the
# two code paths stay in lock-step on generation knobs.
from scripts.generate_gold_completions import (
    GENERATION_BATCH,
    REFUSAL_WARN_THRESHOLD,
    generate_for_behavior,
)

DEFAULT_MODEL = "huihui-ai/Qwen2.5-7B-Instruct-abliterated-v3"
DEFAULT_CONFIG = "data/advbench_behaviors.json"
DEFAULT_OUTPUT = "data/gold_harmful_completions_advbench.json"
DEFAULT_N = 100
DEFAULT_MAX_TOKENS = 512
DEFAULT_TEMP = 0.8
DEFAULT_TOP_P = 0.95
DEFAULT_SEED = 235711


# ---------------------------------------------------------------------------
# Sharding + paths
# ---------------------------------------------------------------------------

def shard_ids(n_total: int, rank: int, world_size: int) -> List[int]:
    """Strided shard of 1..n_total for better load balance across ranks."""
    if world_size <= 0:
        raise ValueError("shard_world_size must be >= 1")
    if not (0 <= rank < world_size):
        raise ValueError(f"shard_rank must be in [0, {world_size}), got {rank}")
    return list(range(1, n_total + 1))[rank::world_size]


def shard_output_path(base: str, rank: int, world_size: int) -> Path:
    """Per-shard output alongside the canonical merged path."""
    p = Path(base)
    stem = p.stem
    return p.with_name(f"{stem}.shard{rank:02d}of{world_size:02d}{p.suffix}")


# ---------------------------------------------------------------------------
# Merge path
# ---------------------------------------------------------------------------

def merge_shards(output_base: str, world_size: int) -> Path:
    """Concatenate per-shard JSONs -> one canonical file."""
    out_path = Path(output_base)
    merged_behaviors: Dict[str, Any] = {}
    merged_metadata: Dict[str, Any] = {}
    missing: List[int] = []

    for rank in range(world_size):
        p = shard_output_path(output_base, rank, world_size)
        if not p.exists():
            missing.append(rank)
            continue
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        merged_metadata.update(data.get("metadata", {}))
        for bid_key, entry in data.get("behaviors", {}).items():
            if bid_key in merged_behaviors:
                # Shards are disjoint by construction; warn on overlap.
                print(f"  !! duplicate BID {bid_key} in shard {rank}, keeping first")
                continue
            merged_behaviors[bid_key] = entry

    if missing:
        print(f"  !! missing shards: {missing} — merged output is INCOMPLETE")

    # Sort by int BID for readability.
    ordered = {
        k: merged_behaviors[k]
        for k in sorted(merged_behaviors.keys(), key=lambda x: int(x))
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {"metadata": merged_metadata, "behaviors": ordered},
            f, indent=2, ensure_ascii=False,
        )
    print(f"Merged {len(ordered)} behaviors across {world_size - len(missing)}/"
          f"{world_size} shards -> {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Generation loop (one shard)
# ---------------------------------------------------------------------------

def run_shard(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    with open(args.behaviors_config, encoding="utf-8") as f:
        all_behaviors = json.load(f)
    n_total = len(all_behaviors)

    bids = shard_ids(n_total, args.shard_rank, args.shard_world_size)
    if args.limit:
        bids = bids[: args.limit]

    out_path = shard_output_path(args.output, args.shard_rank, args.shard_world_size)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"=== Gold Completions (AdvBench) | shard {args.shard_rank}"
          f"/{args.shard_world_size} ===")
    print(f"  Model:       {args.abliterated_model}")
    print(f"  Config:      {args.behaviors_config}  (N={n_total})")
    print(f"  Shard BIDs:  {len(bids)} ({bids[:3]}...{bids[-3:]})")
    print(f"  Samples/beh: {args.n_samples}")
    print(f"  Max tokens:  {args.max_new_tokens}")
    print(f"  Temp/top_p:  {args.temperature}/{args.top_p}")
    print(f"  Judge:       {'ON' if args.enable_judge else 'off'}")
    print(f"  Output:      {out_path}")
    print()

    # ---- Load model ------------------------------------------------------
    print("Loading model...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.abliterated_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.abliterated_model,
        torch_dtype="auto",
        device_map={"": f"cuda:{args.device}"},
    )
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print(f"  Model loaded in {time.time() - t0:.1f}s\n")

    # ---- Resume from previous shard run if any --------------------------
    existing: Dict[str, Any] = {}
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            existing = json.load(f)
        print(f"  Resuming: {len(existing.get('behaviors', {}))} behaviors already done\n")

    metadata = {
        "model": args.abliterated_model,
        "timestamp": datetime.datetime.now().isoformat(),
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "n_target": args.n_samples,
        "shard_rank": args.shard_rank,
        "shard_world_size": args.shard_world_size,
        "behaviors_config": args.behaviors_config,
        "judge_enabled": args.enable_judge,
    }

    behaviors_out: Dict[str, Any] = existing.get("behaviors", {})
    t_all = time.time()

    for bid in bids:
        bid_key = str(bid)
        if bid_key in behaviors_out:
            n_existing = len(behaviors_out[bid_key].get("completions", []))
            if n_existing >= args.n_samples:
                print(f"  BID {bid:>3d}: {n_existing} completions, skip")
                continue

        behavior = all_behaviors[bid - 1]
        instruction = behavior["behaviour"]
        print(f"  BID {bid:>3d}: {instruction[:70]}...")

        t0 = time.time()
        completions, n_refused = generate_for_behavior(
            model, tokenizer, instruction,
            args.n_samples, args.max_new_tokens,
            args.temperature, args.top_p,
        )
        elapsed = time.time() - t0

        refusal_rate = n_refused / max(n_refused + len(completions), 1)
        tag = ""
        if refusal_rate > REFUSAL_WARN_THRESHOLD:
            tag = f"  !! HIGH REFUSAL: {refusal_rate:.0%}"
        if len(completions) < args.n_samples:
            tag += f"  !! SHORT: {len(completions)}/{args.n_samples}"

        mean_len = sum(len(c) for c in completions) / max(len(completions), 1)
        print(f"           {len(completions)} accepted, {n_refused} refused, "
              f"mean_len={mean_len:.0f} chars, {elapsed:.1f}s{tag}")

        entry: Dict[str, Any] = {
            "instruction": instruction,
            "n_generated": len(completions),
            "n_refused_discarded": n_refused,
            "completions": completions,
        }

        # ====================================================================
        # LLM-AS-JUDGE HOOK (abliterated / gold completions)
        # -------------------------------------------------------------------
        # Enabled with `--enable_judge`.  To plug in a real judge, implement
        # `judge_generation` inside `robust_gcg/judge_hooks.py` — no other
        # file needs editing.  This block attaches per-behaviour judge
        # results to the gold-completions JSON so downstream analysis can
        # compare judge scores between the "cooperative" (abliterated) and
        # "attacked" (aligned-model + adversarial suffix) distributions.
        # ====================================================================
        if args.enable_judge and completions:
            from robust_gcg.judge_hooks import score_abliterated
            entry["judge"] = score_abliterated(behavior, completions)

        behaviors_out[bid_key] = entry

        # Checkpoint after each behaviour so any shard restart is O(1) work.
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"metadata": metadata, "behaviors": behaviors_out},
                      f, indent=2, ensure_ascii=False)

        gc.collect()
        torch.cuda.empty_cache()

    total_time = time.time() - t_all
    n_done = sum(
        1 for v in behaviors_out.values()
        if len(v.get("completions", [])) >= args.n_samples
    )
    n_completions = sum(len(v.get("completions", [])) for v in behaviors_out.values())

    print(f"\n=== Shard {args.shard_rank}/{args.shard_world_size} done ===")
    print(f"  Behaviors complete: {n_done}/{len(bids)}")
    print(f"  Total completions:  {n_completions}")
    print(f"  Time:               {total_time:.0f}s ({total_time/60:.1f}m)")
    print(f"  Output:             {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--abliterated_model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--behaviors_config", type=str, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT,
                        help="Canonical merged-output path; per-shard files "
                             "are named <stem>.shardKKofNN<ext>")
    parser.add_argument("--n_samples", type=int, default=DEFAULT_N)
    parser.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMP)
    parser.add_argument("--top_p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", type=int, default=0,
                        help="Local CUDA device index inside this process")
    parser.add_argument("--shard_rank", type=int, default=0)
    parser.add_argument("--shard_world_size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0,
                        help="If >0, only process the first N BIDs in this "
                             "shard (smoke / debugging)")
    parser.add_argument("--enable_judge", action="store_true",
                        help="Run robust_gcg.judge_hooks.score_abliterated "
                             "on each behaviour's completions")
    parser.add_argument("--merge_only", action="store_true",
                        help="Skip generation; just merge per-shard files")
    return parser


def main():
    args = build_parser().parse_args()

    if args.merge_only:
        merge_shards(args.output, args.shard_world_size)
        return

    run_shard(args)


if __name__ == "__main__":
    main()
