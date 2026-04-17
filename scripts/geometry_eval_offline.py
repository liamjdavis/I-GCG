#!/usr/bin/env python3
"""Offline re-scoring of existing experiment_report.json with geometry metrics.

Re-reads the F-C+D results already on disk, generates refused references
from the aligned model, loads gold completions, and computes all semantic
+ distributional metrics — no attack GPU time needed.

Usage
-----
    # Default (reads deliverable/experiment_report.json):
    python scripts/geometry_eval_offline.py --model_path /workspace/models/Qwen2-7B-Instruct

    # Custom paths:
    python scripts/geometry_eval_offline.py \
        --model_path $MODEL_PATH \
        --report deliverable/experiment_report.json \
        --gold_path data/gold_harmful_completions.json

Runtime: ~30 min on A100 (embedding + metric computation only).
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
from robust_gcg.eval_utils import TEST_PREFIXES
from robust_gcg.semantic_eval import (
    evaluate_behavior,
    load_gold_completions,
)
from scripts.fcd_attack_report import (
    TEMPLATE_NAME,
    generate_extended,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEED = 235711
DEFAULT_REPORT = "deliverable/experiment_report.json"
DEFAULT_GOLD = "data/gold_harmful_completions.json"
DEFAULT_CONFIG = "data/cyber_behaviors_v2_all40.json"
EMBED_MODEL = "all-MiniLM-L6-v2"
N_REFUSED_SAMPLES = 50


def generate_refused_samples(model, tokenizer, behavior, n, template_name):
    """Generate N refused (clean-prompt) samples from the aligned model."""
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


def main():
    parser = argparse.ArgumentParser(description="Offline geometry re-scoring")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--report", type=str, default=DEFAULT_REPORT)
    parser.add_argument("--gold_path", type=str, default=DEFAULT_GOLD)
    parser.add_argument("--behaviors_config", type=str, default=DEFAULT_CONFIG)
    parser.add_argument("--embed_model", type=str, default=EMBED_MODEL)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n_refused", type=int, default=N_REFUSED_SAMPLES)
    parser.add_argument("--enable_mauve", action="store_true")
    parser.add_argument("--enable_strongreject", action="store_true")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path(args.output_dir or f"output/geometry_eval/offline/{ts}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Geometry Eval — Offline Re-scoring ===")
    print(f"  Source report: {args.report}")
    print(f"  Gold path:     {args.gold_path}")
    print(f"  Embed model:   {args.embed_model}")
    print(f"  Output:        {output_dir}")
    print()

    # Load experiment report
    print("Loading experiment report...")
    with open(args.report) as f:
        report = json.load(f)

    verifications = {v["behavior_id"]: v for v in report.get("verification", [])}
    per_behavior_results = {r["behavior_id"]: r for r in report.get("per_behavior_results", [])}

    print(f"  {len(verifications)} verified behaviors in report")

    # Load gold completions
    print("Loading gold completions...")
    gold = load_gold_completions(args.gold_path)

    # Load behaviors config
    with open(args.behaviors_config) as f:
        all_behaviors = json.load(f)

    # Determine which BIDs have both verification data and gold completions
    valid_bids = []
    for bid, v in verifications.items():
        bid_key = str(bid)
        if bid_key in gold and gold[bid_key]:
            valid_bids.append(bid)
        else:
            print(f"  !! BID {bid}: no gold completions, will skip")

    print(f"  {len(valid_bids)} behaviors with both verification + gold data")

    # Load model (only needed for generating refused samples)
    need_model = True
    refused_cache_path = output_dir / "refused_cache.json"
    refused_cache: Dict[str, List[str]] = {}
    if refused_cache_path.exists():
        with open(refused_cache_path) as f:
            refused_cache = json.load(f)
        uncached = [b for b in valid_bids if str(b) not in refused_cache]
        if not uncached:
            need_model = False
            print(f"  All refused samples cached, skipping model load")

    model, tokenizer = None, None
    if need_model:
        print("Loading model for refused-sample generation...")
        model, tokenizer = load_model_and_tokenizer(
            args.model_path, low_cpu_mem_usage=True,
            use_cache=False, device=args.device,
        )

    # Generate refused samples for each behavior (cached)
    for bid in valid_bids:
        bid_key = str(bid)
        if bid_key in refused_cache:
            continue

        behavior = all_behaviors[bid - 1]
        print(f"  Generating {args.n_refused} refused samples for BID {bid}...")
        refused_cache[bid_key] = generate_refused_samples(
            model, tokenizer, behavior, args.n_refused, TEMPLATE_NAME,
        )
        # Checkpoint
        with open(refused_cache_path, "w") as f:
            json.dump(refused_cache, f, indent=2, ensure_ascii=False)

    # Free model memory
    if model is not None:
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    # Compute metrics
    geometry_results: Dict[str, Any] = {
        "metadata": {
            "timestamp": ts,
            "source_report": args.report,
            "embed_model": args.embed_model,
            "seed": args.seed,
            "n_refused": args.n_refused,
            "enable_mauve": args.enable_mauve,
            "enable_strongreject": args.enable_strongreject,
        },
        "per_behavior": {},
    }

    t_all = time.time()

    for bid in valid_bids:
        bid_key = str(bid)
        v = verifications[bid]
        p1 = per_behavior_results.get(bid, {})
        behavior = all_behaviors[bid - 1]

        # Collect attacked texts from the verification data
        attacked_texts = []
        greedy = v.get("greedy_gen", "")
        if greedy:
            attacked_texts.append(greedy)
        attacked_texts.extend(v.get("sampled_gens", []))

        if len(attacked_texts) < 2:
            print(f"  BID {bid}: too few attacked samples ({len(attacked_texts)}), skipping")
            continue

        refused_texts = refused_cache.get(bid_key, [])
        cooperative_texts = gold.get(bid_key, [])

        print(f"  BID {bid:>2}: {len(attacked_texts)} atk, "
              f"{len(refused_texts)} ref, {len(cooperative_texts)} coop")

        use_mauve = (args.enable_mauve
                     and len(attacked_texts) >= 20
                     and len(cooperative_texts) >= 20)

        metrics = evaluate_behavior(
            behavior_id=bid,
            attacked_texts=attacked_texts,
            refused_texts=refused_texts,
            cooperative_texts=cooperative_texts[:100],
            embed_model=args.embed_model,
            compute_mauve=use_mauve,
            compute_sr=args.enable_strongreject,
            behavior_instruction=behavior["behaviour"],
        )

        geometry_results["per_behavior"][bid_key] = {
            "instruction": behavior["behaviour"],
            "converged": p1.get("converged", False),
            "n_attacked": len(attacked_texts),
            "n_refused": len(refused_texts),
            "n_cooperative": len(cooperative_texts),
            "greedy_cls": v.get("greedy_cls", {}),
            "metrics": metrics,
        }

    total_time = time.time() - t_all
    geometry_results["metadata"]["total_wall_time_s"] = total_time

    # Write report
    report_path = output_dir / "geometry_report.json"
    with open(report_path, "w") as f:
        json.dump(geometry_results, f, indent=2, default=str)

    print(f"\n=== DONE in {total_time:.0f}s ===")
    print(f"  Report: {report_path}")

    # Summary table
    print(f"\n{'BID':>4}  {'Conv':>4}  {'OldASR':>7}  {'SemASR':>7}  "
          f"{'GeoASR':>7}  {'FBD_c':>10}  {'MMD_c':>10}  {'FR_c':>8}  {'SW_c':>8}")
    print("-" * 88)
    for bid_key in sorted(geometry_results["per_behavior"], key=lambda x: int(x)):
        br = geometry_results["per_behavior"][bid_key]
        m = br.get("metrics", {})
        sem = m.get("semantic_asr", {}).get("mean", float("nan"))
        geo = m.get("geometric_asr", {}).get("geometric_asr", float("nan"))
        fbd = m.get("fbd_atk_coop", float("nan"))
        mmd = m.get("mmd_atk_coop", float("nan"))
        fr = m.get("fisher_rao_atk_coop", float("nan"))
        sw = m.get("sw_atk_coop", float("nan"))
        conv = "Y" if br.get("converged") else "N"
        old_asr = "CONT" if br.get("greedy_cls", {}).get("content_asr") else \
                  "STRCT" if br.get("greedy_cls", {}).get("strict_asr") else \
                  "PRFX" if br.get("greedy_cls", {}).get("prefix_asr") else "FAIL"
        print(f"{bid_key:>4}  {conv:>4}  {old_asr:>7}  {sem:>7.3f}  "
              f"{geo:>7.3f}  {fbd:>10.2f}  {mmd:>10.4f}  {fr:>8.2f}  {sw:>8.4f}")

    # Cross-metric correlation summary
    all_sem = []
    all_geo = []
    for br in geometry_results["per_behavior"].values():
        m = br.get("metrics", {})
        s = m.get("semantic_asr", {}).get("mean")
        g = m.get("geometric_asr", {}).get("geometric_asr")
        if s is not None and g is not None:
            all_sem.append(s)
            all_geo.append(g)

    if len(all_sem) >= 5:
        corr = np.corrcoef(all_sem, all_geo)[0, 1]
        print(f"\n  Pearson(SemanticASR, GeometricASR) = {corr:.3f}  (N={len(all_sem)})")


if __name__ == "__main__":
    main()
