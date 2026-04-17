#!/usr/bin/env python3
"""Smoke test for gold completions generation.

Generates 10 completions for 9 diverse behaviors to validate the pipeline
before committing to the full 40×100 run.

BIDs chosen to span: network hacking (3), ransomware (7), password cracking (8),
phishing (14), DDoS (20), self-driving car (24), election hacking (28),
password cracking duplicate/drift case (32), network theft (37).

Usage
-----
    python scripts/generate_gold_smoke.py
    python scripts/generate_gold_smoke.py --device 0
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from scripts.generate_gold_completions import main as gold_main

SMOKE_BIDS = "3,7,8,14,20,24,28,32,37"
SMOKE_N = 10
SMOKE_OUTPUT = "data/gold_harmful_completions_smoke.json"


if __name__ == "__main__":
    sys.argv = [
        "generate_gold_smoke",
        "--behavior_ids", SMOKE_BIDS,
        "--n_samples", str(SMOKE_N),
        "--output", SMOKE_OUTPUT,
        "--max_new_tokens", "256",
    ] + sys.argv[1:]
    gold_main()
