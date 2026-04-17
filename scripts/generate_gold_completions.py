#!/usr/bin/env python3
"""Generate cooperative (non-refused) completions from an abliterated model.

Loads an abliterated Qwen variant, prompts it with each behavior's raw
instruction, and collects N completions per behavior.  The output JSON
serves as the "cooperative distribution" anchor for SemanticASR,
GeometricASR, FBD, and other distributional metrics.

Usage
-----
    # Full run (all 40 behaviors, 100 samples each, ~2-3 h on A100):
    python scripts/generate_gold_completions.py

    # Smoke test (3 behaviors, 10 samples, ~5 min):
    python scripts/generate_gold_completions.py --behavior_ids 3,8,32 --n_samples 10

    # Custom model:
    python scripts/generate_gold_completions.py \
        --abliterated_model huihui-ai/Qwen2.5-Coder-7B-Instruct-abliterated
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

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from robust_gcg.eval_utils import TEST_PREFIXES

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "huihui-ai/Qwen2.5-7B-Instruct-abliterated-v3"
DEFAULT_CONFIG = "data/cyber_behaviors_v2_all40.json"
DEFAULT_OUTPUT = "data/gold_harmful_completions.json"
DEFAULT_N = 100
DEFAULT_MAX_TOKENS = 512
DEFAULT_TEMP = 0.8
DEFAULT_TOP_P = 0.95
DEFAULT_SEED = 235711
GENERATION_BATCH = 8
REFUSAL_WARN_THRESHOLD = 0.20
MAX_RETRIES_PER_BEHAVIOR = 25


def parse_behavior_ids(raw: str, n_total: int) -> list[int]:
    """Parse '1-40' or '3,8,32' into a list of ints."""
    parts = raw.replace(" ", "").split(",")
    ids: list[int] = []
    for p in parts:
        if "-" in p:
            lo, hi = p.split("-", 1)
            ids.extend(range(int(lo), int(hi) + 1))
        else:
            ids.append(int(p))
    return [i for i in ids if 1 <= i <= n_total]


def build_chat_prompt(tokenizer, instruction: str) -> str:
    """Format a single user turn using the model's chat template."""
    messages = [{"role": "user", "content": instruction}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )


def is_refusal(text: str) -> bool:
    return any(p in text for p in TEST_PREFIXES)


@torch.no_grad()
def generate_batch(
    model, tokenizer, prompt: str, n: int,
    max_new_tokens: int, temperature: float, top_p: float,
) -> list[str]:
    """Generate *n* completions for a single prompt, batched."""
    inputs = tokenizer(
        [prompt] * n, return_tensors="pt", padding=True, truncation=False,
    ).to(model.device)
    gen_cfg = copy.deepcopy(model.generation_config)
    gen_cfg.max_new_tokens = max_new_tokens
    gen_cfg.do_sample = True
    gen_cfg.temperature = temperature
    gen_cfg.top_p = top_p

    outputs = model.generate(
        **inputs, generation_config=gen_cfg,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    prompt_len = inputs["input_ids"].shape[1]
    completions = [
        tokenizer.decode(o[prompt_len:], skip_special_tokens=True).strip()
        for o in outputs
    ]
    return completions


def generate_for_behavior(
    model, tokenizer, instruction: str,
    n_target: int, max_new_tokens: int,
    temperature: float, top_p: float,
) -> tuple[list[str], int]:
    """Collect *n_target* non-refused completions, returning texts + discard count."""
    prompt = build_chat_prompt(tokenizer, instruction)
    accepted: list[str] = []
    n_refused = 0

    for retry in range(MAX_RETRIES_PER_BEHAVIOR):
        remaining = n_target - len(accepted)
        if remaining <= 0:
            break
        batch_n = min(remaining + 4, GENERATION_BATCH)
        batch = generate_batch(
            model, tokenizer, prompt, batch_n,
            max_new_tokens, temperature, top_p,
        )
        for text in batch:
            if len(accepted) >= n_target:
                break
            if is_refusal(text) or len(text.strip()) < 10:
                n_refused += 1
                continue
            accepted.append(text)

        gc.collect()
        torch.cuda.empty_cache()

    return accepted[:n_target], n_refused


def main():
    parser = argparse.ArgumentParser(
        description="Generate gold harmful completions from an abliterated model",
    )
    parser.add_argument(
        "--abliterated_model", type=str, default=DEFAULT_MODEL,
        help="HuggingFace model ID for the abliterated model",
    )
    parser.add_argument("--behaviors_config", type=str, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--behavior_ids", type=str, default=None,
        help="Comma-separated or range (e.g. '1-40', '3,8,32'). Default: all.",
    )
    parser.add_argument("--n_samples", type=int, default=DEFAULT_N)
    parser.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMP)
    parser.add_argument("--top_p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    with open(args.behaviors_config, encoding="utf-8") as f:
        all_behaviors = json.load(f)

    n_total = len(all_behaviors)
    if args.behavior_ids is None:
        behavior_ids = list(range(1, n_total + 1))
    else:
        behavior_ids = parse_behavior_ids(args.behavior_ids, n_total)

    print(f"=== Gold Completions Generator ===")
    print(f"  Model:       {args.abliterated_model}")
    print(f"  Behaviors:   {len(behavior_ids)} ({min(behavior_ids)}-{max(behavior_ids)})")
    print(f"  Samples/beh: {args.n_samples}")
    print(f"  Max tokens:  {args.max_new_tokens}")
    print(f"  Temperature: {args.temperature}, top_p: {args.top_p}")
    print(f"  Seed:        {args.seed}")
    print(f"  Output:      {args.output}")
    print()

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

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing partial results if any
    existing: Dict[str, Any] = {}
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            existing = json.load(f)
        print(f"  Found existing output with {len(existing.get('behaviors', {}))} behaviors")

    metadata = {
        "model": args.abliterated_model,
        "timestamp": datetime.datetime.now().isoformat(),
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "n_target": args.n_samples,
    }

    behaviors_out = existing.get("behaviors", {})
    t_all = time.time()

    for bid in behavior_ids:
        bid_key = str(bid)
        if bid_key in behaviors_out:
            n_existing = len(behaviors_out[bid_key].get("completions", []))
            if n_existing >= args.n_samples:
                print(f"  BID {bid:>2d}: already have {n_existing} completions, skipping")
                continue

        behavior = all_behaviors[bid - 1]
        instruction = behavior["behaviour"]
        print(f"  BID {bid:>2d}: {instruction[:70]}...")

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
            tag = f"  !! HIGH REFUSAL RATE: {refusal_rate:.0%}"
        if len(completions) < args.n_samples:
            tag += f"  !! SHORT: only {len(completions)}/{args.n_samples}"

        mean_len = sum(len(c) for c in completions) / max(len(completions), 1)
        print(f"          {len(completions)} accepted, {n_refused} refused, "
              f"mean_len={mean_len:.0f} chars, {elapsed:.1f}s{tag}")

        behaviors_out[bid_key] = {
            "instruction": instruction,
            "n_generated": len(completions),
            "n_refused_discarded": n_refused,
            "completions": completions,
        }

        # Checkpoint after each behavior
        report = {"metadata": metadata, "behaviors": behaviors_out}
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    total_time = time.time() - t_all
    n_behaviors_done = sum(
        1 for k in behaviors_out
        if len(behaviors_out[k].get("completions", [])) >= args.n_samples
    )
    n_total_completions = sum(
        len(v.get("completions", [])) for v in behaviors_out.values()
    )

    print(f"\n=== Summary ===")
    print(f"  Behaviors completed: {n_behaviors_done}/{len(behavior_ids)}")
    print(f"  Total completions:   {n_total_completions}")
    print(f"  Total time:          {total_time:.0f}s ({total_time/60:.1f}m)")
    print(f"  Output:              {output_path}")


if __name__ == "__main__":
    main()
