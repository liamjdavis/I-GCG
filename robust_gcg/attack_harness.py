"""Shared attack-loop harness for all four robust GCG scripts.

Each script only needs to implement a ``select_candidate`` function and pass
it to :func:`run_attack`.  Everything else — argument parsing, model loading,
the GCG optimisation loop, checkpointing, the SmoothLLM sweep, and plotting —
lives here.
"""

from __future__ import annotations

import argparse
import copy
import datetime
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# ----- repo-local imports --------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from llm_attacks.minimal_gcg.opt_utils import (
    get_filtered_cands,
    get_logits,
    load_model_and_tokenizer,
    sample_control,
    target_loss,
    token_gradients,
)
from llm_attacks.minimal_gcg.string_utils import SuffixManager, load_conversation_template
from llm_attacks import get_nonascii_toks

from robust_gcg.eval_utils import (
    ExperimentLogger,
    RobustEvaluator,
    plot_run_results,
    run_smoothllm_sweep,
    TEST_PREFIXES,
)
from robust_gcg.perturbation import apply_perturbation

# SmoothLLM imports
_smooth_llm_path = os.path.join(str(_REPO_ROOT), "smooth-llm")
if _smooth_llm_path not in sys.path:
    sys.path.insert(0, _smooth_llm_path)


# ---------------------------------------------------------------------------
# Argument parser (shared across all scripts)
# ---------------------------------------------------------------------------

def build_parser(method_name: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"Robust GCG — {method_name}")
    # Model
    parser.add_argument("--model_path", type=str, default="/workspace/models/Qwen2-7B-Instruct")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--template_name", type=str, default="qwen-7b-chat")
    # Behaviour
    parser.add_argument("--id", type=int, default=1)
    parser.add_argument("--behaviors_config", type=str, default="data/cyber_behaviors.json")
    # GCG hyper-params
    parser.add_argument("--batch_size", type=int, default=None, help="Override config batch_size")
    parser.add_argument("--top_k", type=int, default=None, help="Override config top_k")
    parser.add_argument("--num_steps", type=int, default=None, help="Override config step count")
    # Robust selection
    parser.add_argument("--pert_type", type=str, default="RandomSwapPerturbation")
    parser.add_argument("--pert_pct", type=float, default=10)
    parser.add_argument("--n_pert_samples", type=int, default=5, help="M perturbation samples for robust eval")
    parser.add_argument("--robust_topk", type=int, default=16, help="Candidates promoted to tier-2")
    parser.add_argument("--warm_start_steps", type=int, default=50)
    parser.add_argument("--n_flips", type=int, default=1,
                        help="Token positions flipped per candidate (1=standard GCG)")
    parser.add_argument("--eval_batch_size", type=int, default=64,
                        help="Micro-batch size for candidate forward pass "
                             "(reduce if VRAM spills on <=24GB GPUs)")
    # Scaffold
    parser.add_argument("--use_scaffold", action="store_true")
    # Token robustness (Script B/D)
    parser.add_argument("--token_robustness_path", type=str, default=None)
    parser.add_argument("--token_neighborhoods_path", type=str, default=None)
    parser.add_argument("--robustness_threshold", type=float, default=0.3)
    # SmoothLLM final eval
    parser.add_argument("--smoothllm_pert_type", type=str, default="RandomSwapPerturbation")
    parser.add_argument("--smoothllm_pert_pct", type=int, default=10)
    parser.add_argument("--smoothllm_num_copies", type=int, default=10)
    # Output / resume
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--plot_only", action="store_true")
    # Seed
    parser.add_argument("--seed", type=int, default=235711)
    return parser


# ---------------------------------------------------------------------------
# Model + SmoothLLM wrappers
# ---------------------------------------------------------------------------

class WrappedLLM:
    """Adapter matching the interface SmoothLLM expects."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "left"

    def __call__(self, batch, max_new_tokens=100):
        inputs = self.tokenizer(batch, padding=True, truncation=False, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.model.device)
        attn_mask = inputs["attention_mask"].to(self.model.device)
        try:
            outputs = self.model.generate(
                input_ids, attention_mask=attn_mask, max_new_tokens=max_new_tokens
            )
        except RuntimeError:
            return []
        batch_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        gen_start = [
            len(self.tokenizer.decode(input_ids[i], skip_special_tokens=True))
            for i in range(len(input_ids))
        ]
        return [out[gen_start[i]:] for i, out in enumerate(batch_outputs)]


def make_smooth_defense_factory(wrapped_model):
    from lib.defenses import SmoothLLM

    def factory(pert_type, pert_pct, num_copies):
        return SmoothLLM(
            target_model=wrapped_model,
            pert_type=pert_type,
            pert_pct=pert_pct,
            num_copies=num_copies,
        )
    return factory


# ---------------------------------------------------------------------------
# CandidateSelector protocol
# ---------------------------------------------------------------------------
# Each script implements a callable with this signature:
#
#   def select_candidate(
#       new_adv_suffix: list[str],       # B candidate suffix strings
#       clean_losses: torch.Tensor,       # (B,)  clean loss per candidate
#       input_ids: torch.Tensor,          # current full-sequence token ids
#       suffix_manager: SuffixManager,
#       model, tokenizer,
#       step: int,
#       args: argparse.Namespace,
#       **kwargs,
#   ) -> tuple[str, float]:              # (best_suffix, robust_loss)

CandidateSelector = Callable[..., Tuple[str, float]]


# ---------------------------------------------------------------------------
# The main attack loop
# ---------------------------------------------------------------------------

def run_attack(
    method_name: str,
    select_candidate: CandidateSelector,
    extra_parser_setup: Callable[[argparse.ArgumentParser], None] | None = None,
    extra_init: Callable | None = None,
):
    """Entry-point called by each attack script."""

    parser = build_parser(method_name)
    if extra_parser_setup:
        extra_parser_setup(parser)
    args = parser.parse_args()

    # --- Seed ---
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # --- Output dir ---
    if args.output_path is None:
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        args.output_path = f"output/robust_eval/{method_name}/{ts}"

    # --- Plot-only mode ---
    if args.plot_only:
        plot_run_results(args.output_path)
        return

    # --- CUDA ---
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    device = "cuda"

    # --- Load behaviour config ---
    with open(args.behaviors_config) as f:
        all_behaviors = json.load(f)

    behavior = all_behaviors[args.id - 1]
    user_prompt = behavior.get("behaviour_scaffolded") if args.use_scaffold else behavior["behaviour"]
    if user_prompt is None:
        user_prompt = behavior["behaviour"]
    target = behavior["target"]
    adv_string_init = behavior["adv_init_suffix"]
    num_steps = args.num_steps or behavior["step"]
    batch_size = args.batch_size or behavior["batch_size"]
    topk = args.top_k or behavior["top_k"]

    print(f"[{method_name}] behaviour #{args.id}: {user_prompt[:80]}…")

    # --- Load model ---
    model, tokenizer = load_model_and_tokenizer(
        args.model_path, low_cpu_mem_usage=True, use_cache=False, device=device
    )
    conv_template = load_conversation_template(args.template_name)

    suffix_manager = SuffixManager(
        tokenizer=tokenizer,
        conv_template=conv_template,
        instruction=user_prompt,
        target=target,
        adv_string=adv_string_init,
    )

    allow_non_ascii = False
    not_allowed_tokens = get_nonascii_toks(tokenizer) if not allow_non_ascii else None

    evaluator = RobustEvaluator(model, tokenizer)
    logger = ExperimentLogger(method_name, args.output_path)

    # SmoothLLM factory (deferred — only built when sweep actually runs)
    smooth_factory = None

    # Optional per-script init (e.g. load token neighborhoods)
    extra_state: dict = {}
    if extra_init:
        extra_state = extra_init(args, model, tokenizer, suffix_manager) or {}

    # --- Resume from checkpoint ---
    existing = logger.load_checkpoint(args.id)
    start_step = 0
    adv_suffix = adv_string_init
    if existing:
        start_step = existing[-1]["step"] + 1
        adv_suffix = existing[-1].get("adv_suffix", adv_string_init)
        print(f"  Resuming from step {start_step}")

    # --- GCG loop ---
    t0_total = time.time()
    for step_i in range(start_step, num_steps):
        t0 = time.time()

        # 1. Encode
        input_ids = suffix_manager.get_input_ids(adv_string=adv_suffix).to(device)

        # 2. Coordinate gradient
        coordinate_grad = token_gradients(
            model, input_ids,
            suffix_manager._control_slice,
            suffix_manager._target_slice,
            suffix_manager._loss_slice,
        )

        # 3. Sample + filter candidates
        with torch.no_grad():
            adv_suffix_tokens = input_ids[suffix_manager._control_slice].to(device)
            new_adv_suffix_toks = sample_control(
                adv_suffix_tokens, coordinate_grad, batch_size,
                topk=topk, temp=1, not_allowed_tokens=not_allowed_tokens,
                n_flips=getattr(args, "n_flips", 1),
            )
            new_adv_suffix = get_filtered_cands(
                tokenizer, new_adv_suffix_toks,
                filter_cand=True, curr_control=adv_suffix,
            )

            # Clean losses for all candidates
            logits, ids = get_logits(
                model=model, tokenizer=tokenizer,
                input_ids=input_ids,
                control_slice=suffix_manager._control_slice,
                test_controls=new_adv_suffix,
                return_ids=True,
                batch_size=getattr(args, "eval_batch_size", 64),
            )
            clean_losses = target_loss(logits, ids, suffix_manager._target_slice)

            # --- candidate selection (script-specific or warm-start) ---
            if step_i < args.warm_start_steps:
                # Standard argmin during warm-start
                best_idx = clean_losses.argmin()
                best_suffix = new_adv_suffix[best_idx]
                robust_loss = 0.0
            else:
                best_suffix, robust_loss = select_candidate(
                    new_adv_suffix=new_adv_suffix,
                    clean_losses=clean_losses,
                    input_ids=input_ids,
                    suffix_manager=suffix_manager,
                    model=model,
                    tokenizer=tokenizer,
                    step=step_i,
                    args=args,
                    **extra_state,
                )

            adv_suffix = best_suffix
            current_loss = clean_losses.min().item()

            # --- evaluate ---
            jb, gen_str, eval_loss = evaluator.evaluate_clean(suffix_manager, adv_suffix)

        wall = time.time() - t0
        entry = {
            "step": step_i,
            "clean_loss": current_loss,
            "robust_loss": robust_loss,
            "clean_asr": 1.0 if jb else 0.0,
            "robust_survival_rate": 0.0,
            "adv_suffix": adv_suffix,
            "gen_str": gen_str,
            "wall_time": wall,
        }
        logger.log_step(args.id, entry)

        if step_i % 10 == 0:
            logger.flush_steps(args.id)
            print(
                f"  step {step_i:>4d}  loss={current_loss:.4f}  "
                f"robust_loss={robust_loss:.4f}  jailbroken={jb}  "
                f"{wall:.1f}s"
            )

        if jb:
            print(f"  SUCCESS at step {step_i}: {gen_str[:80]}")
            logger.flush_steps(args.id)
            break

        del coordinate_grad, adv_suffix_tokens
        gc.collect()
        torch.cuda.empty_cache()

    logger.flush_steps(args.id)
    total_time = time.time() - t0_total

    # --- Final SmoothLLM sweep ---
    sweep: Dict[str, Any] = {}
    if not getattr(args, "skip_smoothllm", False):
        print("  Running SmoothLLM evaluation sweep…")
        wrapped_model = WrappedLLM(model, tokenizer)
        smooth_factory = make_smooth_defense_factory(wrapped_model)
        sweep = run_smoothllm_sweep(
            evaluator=evaluator,
            smooth_defense_factory=smooth_factory,
            suffix_manager=suffix_manager,
            adv_suffix=adv_suffix,
            user_prompt=user_prompt,
        )

    summary = {
        "behavior_id": args.id,
        "total_steps": step_i + 1,
        "converged": jb,
        "final_clean_asr": 1.0 if jb else 0.0,
        "final_suffix": adv_suffix,
        "smoothllm_sweep": sweep,
        "total_wall_time": total_time,
    }
    logger.log_summary(args.id, summary)

    # --- Plots ---
    plot_run_results(logger.log_dir)

    # --- One-liner ---
    smooth_results = {k: v["jailbroken"] for k, v in sweep.items()}
    print(f"  DONE  method={method_name}  id={args.id}  steps={step_i+1}  "
          f"clean_jb={jb}  smoothllm={smooth_results}  time={total_time:.0f}s")


# ---------------------------------------------------------------------------
# Programmatic entry-point (no CLI parsing, accepts pre-loaded model)
# ---------------------------------------------------------------------------

_DEFAULT_PARAMS: Dict[str, Any] = {
    "model_path": "/workspace/models/Qwen2-7B-Instruct",
    "device": 0,
    "template_name": "qwen-7b-chat",
    "id": 1,
    "behaviors_config": "data/cyber_behaviors.json",
    "batch_size": None,
    "top_k": None,
    "num_steps": None,
    "pert_type": "RandomSwapPerturbation",
    "pert_pct": 10,
    "n_pert_samples": 5,
    "robust_topk": 16,
    "warm_start_steps": 50,
    "use_scaffold": False,
    "token_robustness_path": None,
    "token_neighborhoods_path": None,
    "robustness_threshold": 0.3,
    "smoothllm_pert_type": "RandomSwapPerturbation",
    "smoothllm_pert_pct": 10,
    "smoothllm_num_copies": 10,
    "output_path": None,
    "plot_only": False,
    "seed": 235711,
    "buffer_token_id": None,
    "n_flips": 1,
    "eval_batch_size": 64,
}


def run_attack_with_model(
    method_name: str,
    select_candidate: CandidateSelector,
    model,
    tokenizer,
    params: Dict[str, Any],
    extra_init: Callable | None = None,
    skip_smoothllm: bool = False,
    skip_plots: bool = False,
) -> Dict[str, Any]:
    """Run a robust GCG attack with an already-loaded model.

    Parameters
    ----------
    params : dict
        Overrides for the default hyper-parameters (same keys as CLI args).
    skip_smoothllm : bool
        If True, skip the final SmoothLLM evaluation sweep.
    skip_plots : bool
        If True, skip plot generation.

    Returns
    -------
    dict
        Summary dict with keys: behavior_id, total_steps, converged,
        final_clean_asr, final_suffix, smoothllm_sweep, total_wall_time.
    """
    merged = {**_DEFAULT_PARAMS, **params}
    args = argparse.Namespace(**merged)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.output_path is None:
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        args.output_path = f"output/robust_eval/{method_name}/{ts}"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(args.behaviors_config) as f:
        all_behaviors = json.load(f)

    behavior = all_behaviors[args.id - 1]
    user_prompt = (
        behavior.get("behaviour_scaffolded") if args.use_scaffold else behavior["behaviour"]
    )
    if user_prompt is None:
        user_prompt = behavior["behaviour"]
    target = behavior["target"]
    adv_string_init = behavior["adv_init_suffix"]
    num_steps = args.num_steps or behavior["step"]
    batch_size = args.batch_size or behavior["batch_size"]
    topk = args.top_k or behavior["top_k"]

    print(f"[{method_name}] behaviour #{args.id}: {user_prompt[:80]}…")

    conv_template = load_conversation_template(args.template_name)
    suffix_manager = SuffixManager(
        tokenizer=tokenizer,
        conv_template=conv_template,
        instruction=user_prompt,
        target=target,
        adv_string=adv_string_init,
    )

    allow_non_ascii = False
    not_allowed_tokens = get_nonascii_toks(tokenizer) if not allow_non_ascii else None

    evaluator = RobustEvaluator(model, tokenizer)
    logger = ExperimentLogger(method_name, args.output_path)

    smooth_factory = None

    extra_state: dict = {}
    if extra_init:
        extra_state = extra_init(args, model, tokenizer, suffix_manager) or {}

    existing = logger.load_checkpoint(args.id)
    start_step = 0
    adv_suffix = adv_string_init
    if existing:
        start_step = existing[-1]["step"] + 1
        adv_suffix = existing[-1].get("adv_suffix", adv_string_init)
        print(f"  Resuming from step {start_step}")

    jb = False
    step_i = start_step
    t0_total = time.time()
    for step_i in range(start_step, num_steps):
        t0 = time.time()

        input_ids = suffix_manager.get_input_ids(adv_string=adv_suffix).to(device)

        coordinate_grad = token_gradients(
            model, input_ids,
            suffix_manager._control_slice,
            suffix_manager._target_slice,
            suffix_manager._loss_slice,
        )

        with torch.no_grad():
            adv_suffix_tokens = input_ids[suffix_manager._control_slice].to(device)
            new_adv_suffix_toks = sample_control(
                adv_suffix_tokens, coordinate_grad, batch_size,
                topk=topk, temp=1, not_allowed_tokens=not_allowed_tokens,
                n_flips=getattr(args, "n_flips", 1),
            )
            new_adv_suffix = get_filtered_cands(
                tokenizer, new_adv_suffix_toks,
                filter_cand=True, curr_control=adv_suffix,
            )

            logits, ids = get_logits(
                model=model, tokenizer=tokenizer,
                input_ids=input_ids,
                control_slice=suffix_manager._control_slice,
                test_controls=new_adv_suffix,
                return_ids=True,
                batch_size=getattr(args, "eval_batch_size", 64),
            )
            clean_losses = target_loss(logits, ids, suffix_manager._target_slice)

            if step_i < args.warm_start_steps:
                best_idx = clean_losses.argmin()
                best_suffix = new_adv_suffix[best_idx]
                robust_loss = 0.0
            else:
                best_suffix, robust_loss = select_candidate(
                    new_adv_suffix=new_adv_suffix,
                    clean_losses=clean_losses,
                    input_ids=input_ids,
                    suffix_manager=suffix_manager,
                    model=model,
                    tokenizer=tokenizer,
                    step=step_i,
                    args=args,
                    **extra_state,
                )

            adv_suffix = best_suffix
            current_loss = clean_losses.min().item()
            jb, gen_str, eval_loss = evaluator.evaluate_clean(suffix_manager, adv_suffix)

        wall = time.time() - t0
        entry = {
            "step": step_i,
            "clean_loss": current_loss,
            "robust_loss": robust_loss,
            "clean_asr": 1.0 if jb else 0.0,
            "robust_survival_rate": 0.0,
            "adv_suffix": adv_suffix,
            "gen_str": gen_str,
            "wall_time": wall,
        }
        logger.log_step(args.id, entry)

        if step_i % 10 == 0:
            logger.flush_steps(args.id)
            print(
                f"  step {step_i:>4d}  loss={current_loss:.4f}  "
                f"robust_loss={robust_loss:.4f}  jailbroken={jb}  "
                f"{wall:.1f}s"
            )

        if jb:
            print(f"  SUCCESS at step {step_i}: {gen_str[:80]}")
            logger.flush_steps(args.id)
            break

        del coordinate_grad, adv_suffix_tokens
        gc.collect()
        torch.cuda.empty_cache()

    logger.flush_steps(args.id)
    total_time = time.time() - t0_total

    sweep: Dict[str, Any] = {}
    if not skip_smoothllm:
        print("  Running SmoothLLM evaluation sweep…")
        wrapped_model = WrappedLLM(model, tokenizer)
        smooth_factory = make_smooth_defense_factory(wrapped_model)
        sweep = run_smoothllm_sweep(
            evaluator=evaluator,
            smooth_defense_factory=smooth_factory,
            suffix_manager=suffix_manager,
            adv_suffix=adv_suffix,
            user_prompt=user_prompt,
        )

    summary = {
        "behavior_id": args.id,
        "total_steps": step_i + 1,
        "converged": jb,
        "final_clean_asr": 1.0 if jb else 0.0,
        "final_suffix": adv_suffix,
        "smoothllm_sweep": sweep,
        "total_wall_time": total_time,
    }
    logger.log_summary(args.id, summary)

    if not skip_plots:
        plot_run_results(logger.log_dir)

    print(
        f"  DONE  method={method_name}  id={args.id}  steps={step_i+1}  "
        f"clean_jb={jb}  time={total_time:.0f}s"
    )
    return summary
