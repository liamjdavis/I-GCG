#!/usr/bin/env python3
"""Build an AdvBench behaviours JSON in the attack-pipeline schema.

Reads ``data/harmful_behaviors.csv`` (AdvBench, 520 rows with columns
``goal,target``) and writes ``data/advbench_behaviors.json`` containing the
same per-behaviour keys the attack code expects in
``data/cyber_behaviors_v2_all40.json``:

    {"id", "behaviour", "target", "adv_init_suffix", "step", "batch_size", "top_k"}

This is a one-shot preprocessing step.  Run it once before
``scripts/generate_gold_advbench.py`` / ``scripts/geometry_eval_advbench.py``.

Usage
-----
    python scripts/build_advbench_json.py
    python scripts/build_advbench_json.py --input data/harmful_behaviors.csv \
                                          --output data/advbench_behaviors.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from data.build_cyber_dataset import load_advbench_csv

DEFAULT_INPUT = "data/harmful_behaviors.csv"
DEFAULT_OUTPUT = "data/advbench_behaviors.json"

# Keep defaults aligned with FULL_SLOT_PARAMS / FULL_BASE_PARAMS in
# scripts/geometry_eval_full.py.  CLI flags on the attack script can still
# override per-run without touching the JSON.
DEFAULT_ADV_INIT = ("! " * 20).strip()
DEFAULT_STEP = 500
DEFAULT_BATCH_SIZE = 256
DEFAULT_TOP_K = 256


def build_entries(
    csv_path: str,
    adv_init: str = DEFAULT_ADV_INIT,
    step: int = DEFAULT_STEP,
    batch_size: int = DEFAULT_BATCH_SIZE,
    top_k: int = DEFAULT_TOP_K,
) -> list[dict]:
    """Load AdvBench CSV and emit attack-schema dicts, 1-indexed by row."""
    rows = load_advbench_csv(csv_path)
    entries: list[dict] = []
    for i, r in enumerate(rows, start=1):
        entries.append({
            "id": i,
            "behaviour": r["behaviour"],
            "target": r["target"],
            "adv_init_suffix": adv_init,
            "step": step,
            "batch_size": batch_size,
            "top_k": top_k,
            "source": r.get("source", "advbench"),
        })
    return entries


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--adv_init_suffix", type=str, default=DEFAULT_ADV_INIT)
    parser.add_argument("--step", type=int, default=DEFAULT_STEP)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K)
    args = parser.parse_args()

    entries = build_entries(
        args.input,
        adv_init=args.adv_init_suffix,
        step=args.step,
        batch_size=args.batch_size,
        top_k=args.top_k,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(entries)} AdvBench behaviours -> {out_path}")
    if entries:
        print(f"  First:  [{entries[0]['id']}] {entries[0]['behaviour'][:80]}")
        print(f"  Last:   [{entries[-1]['id']}] {entries[-1]['behaviour'][:80]}")


if __name__ == "__main__":
    main()
