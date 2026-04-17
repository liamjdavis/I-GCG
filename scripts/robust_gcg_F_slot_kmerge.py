#!/usr/bin/env python3
"""Method F -- SlotGCG positional insertion with K-merge candidate selection.

Uses attention-based Vulnerable Slot Scores (VSS) to distribute adversarial
tokens across the most vulnerable positions *within* the behaviour prompt,
rather than appending a suffix.  After the one-shot VSS probe, runs standard
GCG-style coordinate-gradient optimisation with I-GCG K-merge candidate
selection, then evaluates against SmoothLLM.

Reference:
    Jeong et al., "SlotGCG: Exploiting the Positional Vulnerability in LLMs
    for Jailbreak Attacks", ICLR 2026.
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
from typing import Any, Dict

import numpy as np
import torch

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from llm_attacks.minimal_gcg.opt_utils import load_model_and_tokenizer
from llm_attacks.minimal_gcg.string_utils import SuffixManager, load_conversation_template
from llm_attacks.minimal_gcg.slot_utils import (
    compute_vss,
    get_nonascii_toks,
    insert_optim_embed_pos,
    interleave_behavior_and_controls,
    slot_candidates_loss,
    slot_sample_control,
    slot_token_gradients,
)

from robust_gcg.eval_utils import (
    ExperimentLogger,
    RobustEvaluator,
    plot_run_results,
    run_smoothllm_sweep,
)
from robust_gcg.attack_harness import WrappedLLM, make_smooth_defense_factory

_smooth_llm_path = os.path.join(str(_REPO), "smooth-llm")
if _smooth_llm_path not in sys.path:
    sys.path.insert(0, _smooth_llm_path)


# ---------------------------------------------------------------------------
# Default params (mirrors attack_harness._DEFAULT_PARAMS + slot additions)
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
    "seed": 235711,
    "output_path": None,
    # SlotGCG-specific
    "num_adv_tokens": 20,
    "attention_temp": 8.0,
    "kmerge_k": 7,
    "use_prefix_cache": True,
    "adv_string_init": "!",
    "search_width": None,
    "eval_batch_size": 64,
    # SmoothLLM
    "skip_smoothllm": False,
}


# ---------------------------------------------------------------------------
# K-merge for slot-based adversarial tokens
# ---------------------------------------------------------------------------

def _kmerge_select(
    optim_ids: torch.Tensor,
    candidate_optim_ids: torch.Tensor,
    candidate_losses: torch.Tensor,
    K: int,
    model,
    embed_layer,
    behavior_ids: torch.Tensor,
    after_embeds: torch.Tensor,
    target_embeds: torch.Tensor,
    target_ids: torch.Tensor,
    optim_pos: torch.Tensor,
    eval_batch_size: int,
    prefix_cache=None,
    before_embeds=None,
    return_all: bool = False,
    loss_fn=None,
) -> tuple:
    """Take top-K candidates by loss, iteratively merge, return best.

    When *return_all* is True the full ``(merged_stack, losses)`` are
    appended to the return tuple so callers can re-rank.
    """
    K = min(K, candidate_optim_ids.shape[0])
    _, sorted_idx = torch.sort(candidate_losses)
    top_k_idx = sorted_idx[:K]

    base_ids = optim_ids.squeeze(0).clone()
    merged_ids = base_ids.clone()

    all_merged: list[torch.Tensor] = []
    for k in range(K):
        cand = candidate_optim_ids[top_k_idx[k]]
        n = min(len(base_ids), len(cand))
        for pos in range(n):
            if base_ids[pos] != cand[pos]:
                merged_ids[pos] = cand[pos]
        all_merged.append(merged_ids.clone())

    merged_stack = torch.stack(all_merged, dim=0)

    merged_interleaved = interleave_behavior_and_controls(
        behavior_ids.expand(K, -1),
        merged_stack,
        optim_pos.unsqueeze(0).expand(K, -1),
    )

    losses = slot_candidates_loss(
        model,
        embed_layer,
        merged_interleaved,
        after_embeds,
        target_embeds,
        target_ids,
        batch_size=eval_batch_size,
        prefix_cache=prefix_cache,
        before_embeds=before_embeds,
        loss_fn=loss_fn,
    )

    best_idx = losses.argmin().item()
    result = (merged_stack[best_idx].unsqueeze(0), losses[best_idx].item())
    if return_all:
        return result + (merged_stack, losses)
    return result


# ---------------------------------------------------------------------------
# SlotAttackRunner
# ---------------------------------------------------------------------------

class SlotAttackRunner:
    """Runs SlotGCG + K-merge optimisation with SmoothLLM evaluation.

    Optional *hooks* dict may contain:

    - ``target_override`` (str): replacement target string (for F-A).
    - ``on_candidate_selected`` (callable): called after K-merge with
      ``(runner, step, best_optim, kmerge_loss, all_merged, all_losses)``
      and must return ``(optim_ids, loss)`` (for F-B).
    - ``on_step_end`` (callable): called at the end of each step with
      ``(runner, step, optim_ids, gen_str, jb)`` (for F-C).
    - ``early_stop_fn`` (callable): ``(runner, step, jb, gen_str) -> bool``
      replaces the default ``jb``-based early stop (for content-ASR stop).
    - ``custom_loss_fn`` (callable): pluggable loss function passed to
      ``slot_token_gradients`` / ``slot_candidates_loss`` (for F-D).
    """

    def __init__(self, args: argparse.Namespace, model, tokenizer, hooks=None):
        hooks = hooks or {}
        self._target_override = hooks.get("target_override")
        self._on_candidate_selected = hooks.get("on_candidate_selected")
        self._on_step_end = hooks.get("on_step_end")
        self._early_stop_fn = hooks.get("early_stop_fn")
        self._custom_loss_fn = hooks.get("custom_loss_fn")

        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device

        # Load behaviour config
        with open(args.behaviors_config) as f:
            all_behaviors = json.load(f)
        behavior = all_behaviors[args.id - 1]

        self.user_prompt = behavior["behaviour"]
        self.target = self._target_override or behavior["target"]
        self.num_steps = args.num_steps or behavior["step"]
        self.search_width = args.search_width or behavior.get("batch_size", 256)
        self.topk = args.top_k or behavior.get("top_k", 256)
        self.num_adv_tokens = args.num_adv_tokens
        self.kmerge_k = args.kmerge_k

        # Conversation template
        self.conv_template = load_conversation_template(args.template_name)
        template_str = self._get_template_string()
        self.template_before, self.template_after = template_str.split("{instruction}")

        # Embedding layer
        self.embed_layer = model.get_input_embeddings()
        vocab_size = self.embed_layer.weight.shape[0]
        self.vocab_embeds = self.embed_layer(
            torch.arange(vocab_size, device=self.device)
        ).detach()

        # Tokenise segments
        self.behavior_ids = torch.tensor(
            tokenizer(self.user_prompt, padding=False, add_special_tokens=False).input_ids,
            device=self.device,
        ).unsqueeze(0)
        self.before_ids = torch.tensor(
            tokenizer(self.template_before, padding=False).input_ids,
            device=self.device,
        ).unsqueeze(0)
        self.after_ids = torch.tensor(
            tokenizer(self.template_after, padding=False, add_special_tokens=False).input_ids,
            device=self.device,
        ).unsqueeze(0)
        self.target_ids = torch.tensor(
            tokenizer(self.target, padding=False, add_special_tokens=False).input_ids,
            device=self.device,
        ).unsqueeze(0)

        self.before_embeds = self.embed_layer(self.before_ids).to(model.dtype)
        self.behavior_embeds = self.embed_layer(self.behavior_ids).to(model.dtype)
        self.after_embeds = self.embed_layer(self.after_ids).to(model.dtype)
        self.target_embeds = self.embed_layer(self.target_ids).to(model.dtype)

        # Prefix cache
        self.prefix_cache = None
        if args.use_prefix_cache:
            model.config.use_cache = True
            with torch.no_grad():
                out = model(inputs_embeds=self.before_embeds, use_cache=True)
                self.prefix_cache = out.past_key_values

        # Non-ASCII token filter
        self.not_allowed_tokens = get_nonascii_toks(tokenizer, device=str(self.device))
        self.not_allowed_tokens = torch.unique(self.not_allowed_tokens)

        # Compute VSS and get optimisation positions
        print(f"  Computing Vulnerable Slot Scores (temp={args.attention_temp})…")
        t_vss = time.time()
        self.optim_pos, self.attention_probs = compute_vss(
            model,
            tokenizer,
            self.template_before,
            self.template_after,
            self.user_prompt,
            self.target,
            num_adv_tokens=self.num_adv_tokens,
            attention_temp=args.attention_temp,
            use_prefix_cache=args.use_prefix_cache,
        )
        self.vss_time = time.time() - t_vss
        print(f"  VSS done in {self.vss_time:.2f}s  positions={self.optim_pos.tolist()}")

        # Initialise adversarial token ids
        init_tok = tokenizer(
            " " + args.adv_string_init,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(self.device)["input_ids"]
        if init_tok.shape[1] != 1:
            init_tok = tokenizer(
                args.adv_string_init,
                return_tensors="pt",
                add_special_tokens=False,
            ).to(self.device)["input_ids"]
        self.optim_ids = init_tok.expand(-1, self.num_adv_tokens)

        # Evaluator and logger
        self.evaluator = RobustEvaluator(model, tokenizer)
        method_name = "F_slot_kmerge"
        self.method_name = method_name
        self.logger = ExperimentLogger(method_name, args.output_path)

    def _get_template_string(self) -> str:
        """Extract the template string with ``{instruction}`` placeholder."""
        self.conv_template.append_message(self.conv_template.roles[0], "{instruction}")
        self.conv_template.append_message(self.conv_template.roles[1], None)
        template = self.conv_template.get_prompt()
        self.conv_template.messages = []
        return template

    def _decode_interleaved(self, optim_ids: torch.Tensor) -> str:
        """Decode optimised adversarial tokens interleaved with behaviour."""
        interleaved = interleave_behavior_and_controls(
            self.behavior_ids,
            optim_ids,
            self.optim_pos.unsqueeze(0),
        )
        return self.tokenizer.decode(interleaved[0], skip_special_tokens=True)

    def _make_eval_suffix_manager(self, interleaved_prompt: str) -> SuffixManager:
        """Build a SuffixManager with the interleaved prompt baked in.

        The adversarial tokens are already embedded in *interleaved_prompt*,
        so we use an empty ``adv_string`` and pass the full interleaved text
        as the instruction.  This lets the existing evaluation infrastructure
        work unchanged.
        """
        conv = load_conversation_template(self.args.template_name)
        return SuffixManager(
            tokenizer=self.tokenizer,
            conv_template=conv,
            instruction=interleaved_prompt,
            target=self.target,
            adv_string="",
        )

    def run(self) -> Dict[str, Any]:
        args = self.args
        model = self.model
        tokenizer = self.tokenizer
        device = self.device

        existing = self.logger.load_checkpoint(args.id)
        start_step = 0
        if existing:
            start_step = existing[-1]["step"] + 1
            saved_ids = existing[-1].get("optim_ids")
            if saved_ids is not None:
                self.optim_ids = torch.tensor(saved_ids, device=device).unsqueeze(0)
            print(f"  Resuming from step {start_step}")

        optim_ids = self.optim_ids
        jb = False
        step_i = start_step
        t0_total = time.time()

        for step_i in range(start_step, self.num_steps):
            t0 = time.time()
            try:
                # 1. Gradient computation
                grad, loss_val = slot_token_gradients(
                    model,
                    self.embed_layer,
                    self.vocab_embeds,
                    self.behavior_embeds,
                    self.after_embeds,
                    self.target_embeds,
                    self.target_ids,
                    optim_ids,
                    self.optim_pos,
                    prefix_cache=self.prefix_cache,
                    before_embeds=None if self.prefix_cache else self.before_embeds,
                    loss_fn=self._custom_loss_fn,
                )

                # 2. Sample candidates
                with torch.no_grad():
                    candidate_optim_ids, token_val, token_pos = slot_sample_control(
                        optim_ids.squeeze(0),
                        grad,
                        self.search_width,
                        topk=self.topk,
                        not_allowed_tokens=self.not_allowed_tokens,
                    )

                    # 3. Filter via decode-reencode roundtrip
                    candidate_interleaved = interleave_behavior_and_controls(
                        self.behavior_ids.expand(self.search_width, -1),
                        candidate_optim_ids,
                        self.optim_pos.unsqueeze(0).expand(self.search_width, -1),
                    )

                    decoded_texts = tokenizer.batch_decode(candidate_interleaved)
                    valid_mask = []
                    for j in range(len(decoded_texts)):
                        reencoded = tokenizer(
                            decoded_texts[j],
                            return_tensors="pt",
                            add_special_tokens=False,
                        ).to(device)["input_ids"][0]
                        valid_mask.append(torch.equal(reencoded, candidate_interleaved[j]))

                    valid_idx = [j for j, v in enumerate(valid_mask) if v]
                    if not valid_idx:
                        valid_idx = list(range(len(decoded_texts)))
                    valid_idx_t = torch.tensor(valid_idx, device=device)
                    filtered_optim = candidate_optim_ids[valid_idx_t]
                    filtered_interleaved = candidate_interleaved[valid_idx_t]

                    # 4. Evaluate candidate losses
                    losses = slot_candidates_loss(
                        model,
                        self.embed_layer,
                        filtered_interleaved,
                        self.after_embeds,
                        self.target_embeds,
                        self.target_ids,
                        batch_size=args.eval_batch_size,
                        prefix_cache=self.prefix_cache,
                        before_embeds=None if self.prefix_cache else self.before_embeds,
                        loss_fn=self._custom_loss_fn,
                    )

                    # 5. K-merge candidate selection
                    need_all = self._on_candidate_selected is not None
                    kmerge_result = _kmerge_select(
                        optim_ids,
                        filtered_optim,
                        losses,
                        self.kmerge_k,
                        model,
                        self.embed_layer,
                        self.behavior_ids,
                        self.after_embeds,
                        self.target_embeds,
                        self.target_ids,
                        self.optim_pos,
                        args.eval_batch_size,
                        prefix_cache=self.prefix_cache,
                        before_embeds=None if self.prefix_cache else self.before_embeds,
                        return_all=need_all,
                        loss_fn=self._custom_loss_fn,
                    )

                    if need_all:
                        best_optim, kmerge_loss, all_merged, all_merged_losses = kmerge_result
                        best_optim, kmerge_loss = self._on_candidate_selected(
                            self, step_i, best_optim, kmerge_loss,
                            all_merged, all_merged_losses,
                        )
                    else:
                        best_optim, kmerge_loss = kmerge_result

                    optim_ids = best_optim
                    current_loss = losses.min().item()

                    # 6. Evaluate clean ASR
                    interleaved_str = self._decode_interleaved(optim_ids)
                    eval_sm = self._make_eval_suffix_manager(interleaved_str)
                    jb, gen_str, eval_loss = self.evaluator.evaluate_clean(eval_sm, "")

                # Hook: on_step_end (e.g. F-C target update)
                if self._on_step_end is not None:
                    self._on_step_end(self, step_i, optim_ids, gen_str, jb)

                wall = time.time() - t0
                entry = {
                    "step": step_i,
                    "clean_loss": current_loss,
                    "robust_loss": kmerge_loss,
                    "clean_asr": 1.0 if jb else 0.0,
                    "robust_survival_rate": 0.0,
                    "adv_suffix": interleaved_str,
                    "optim_ids": optim_ids.squeeze(0).tolist(),
                    "gen_str": gen_str,
                    "wall_time": wall,
                }
                self.logger.log_step(args.id, entry)

                if step_i % 10 == 0:
                    self.logger.flush_steps(args.id)
                    print(
                        f"  step {step_i:>4d}  loss={current_loss:.4f}  "
                        f"kmerge_loss={kmerge_loss:.4f}  jailbroken={jb}  "
                        f"{wall:.1f}s"
                    )

                should_stop = (
                    self._early_stop_fn(self, step_i, jb, gen_str)
                    if self._early_stop_fn is not None
                    else jb
                )
                if should_stop:
                    print(f"  SUCCESS at step {step_i}: {gen_str[:80]}")
                    self.logger.flush_steps(args.id)
                    break

                del grad
                gc.collect()
                torch.cuda.empty_cache()

            except torch.cuda.OutOfMemoryError:
                self.logger.flush_steps(args.id)
                raise

        self.logger.flush_steps(args.id)
        total_time = time.time() - t0_total

        # Final SmoothLLM sweep
        sweep: Dict[str, Any] = {}
        if not getattr(args, "skip_smoothllm", False):
            print("  Running SmoothLLM evaluation sweep…")
            interleaved_str = self._decode_interleaved(optim_ids)
            eval_sm = self._make_eval_suffix_manager(interleaved_str)

            wrapped_model = WrappedLLM(model, tokenizer)
            smooth_factory = make_smooth_defense_factory(wrapped_model)
            sweep = run_smoothllm_sweep(
                evaluator=self.evaluator,
                smooth_defense_factory=smooth_factory,
                suffix_manager=eval_sm,
                adv_suffix="",
                user_prompt=interleaved_str,
            )

        summary = {
            "behavior_id": args.id,
            "total_steps": step_i + 1,
            "converged": jb,
            "final_clean_asr": 1.0 if jb else 0.0,
            "final_interleaved_prompt": self._decode_interleaved(optim_ids),
            "final_optim_ids": optim_ids.squeeze(0).tolist(),
            "optim_pos": self.optim_pos.tolist(),
            "vss_time": self.vss_time,
            "smoothllm_sweep": sweep,
            "total_wall_time": total_time,
        }
        self.logger.log_summary(args.id, summary)

        if not getattr(args, "skip_plots", False):
            try:
                plot_run_results(self.logger.log_dir)
            except ImportError as e:
                print(f"  [plot] skipped — {e}")

        smooth_results = {k: v["jailbroken"] for k, v in sweep.items()}
        print(
            f"  DONE  method={self.method_name}  id={args.id}  steps={step_i + 1}  "
            f"clean_jb={jb}  smoothllm={smooth_results}  time={total_time:.0f}s"
        )
        return summary


# ---------------------------------------------------------------------------
# Programmatic entry point (for fast_robust_eval.py)
# ---------------------------------------------------------------------------

def run_slot_attack_with_model(
    model,
    tokenizer,
    params: Dict[str, Any],
    skip_smoothllm: bool = False,
    skip_plots: bool = False,
    hooks: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Run a SlotGCG + K-merge attack with a pre-loaded model.

    Parameters
    ----------
    params : dict
        Overrides for ``_DEFAULT_PARAMS``.
    hooks : dict, optional
        Callback hooks passed to :class:`SlotAttackRunner`.  See its
        docstring for the recognised keys.
    """
    merged = {**_DEFAULT_PARAMS, **params}
    if skip_smoothllm:
        merged["skip_smoothllm"] = True
    if skip_plots:
        merged["skip_plots"] = True
    args = argparse.Namespace(**merged)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.output_path is None:
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        args.output_path = f"output/robust_eval/F_slot_kmerge/{ts}"

    runner = SlotAttackRunner(args, model, tokenizer, hooks=hooks)
    return runner.run()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def build_slot_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser with both shared and slot-specific args."""
    from robust_gcg.attack_harness import build_parser

    parser = build_parser("F_slot_kmerge")
    parser.add_argument(
        "--num_adv_tokens", type=int, default=20,
        help="Total adversarial tokens distributed across slots.",
    )
    parser.add_argument(
        "--attention_temp", type=float, default=8.0,
        help="Softmax temperature for VSS computation.",
    )
    parser.add_argument(
        "--kmerge_k", type=int, default=7,
        help="Number of top candidates to merge (K in I-GCG).",
    )
    parser.add_argument(
        "--use_prefix_cache", action="store_true", default=True,
        help="Cache KV states for static template prefix.",
    )
    parser.add_argument(
        "--no_prefix_cache", action="store_false", dest="use_prefix_cache",
        help="Disable prefix KV cache.",
    )
    parser.add_argument(
        "--adv_string_init", type=str, default="!",
        help="Initial adversarial token character.",
    )
    parser.add_argument(
        "--search_width", type=int, default=None,
        help="Number of candidates per step (defaults to config batch_size).",
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=64,
        help="Batch size for candidate loss evaluation.",
    )
    parser.add_argument(
        "--skip_smoothllm", action="store_true",
        help="Skip the final SmoothLLM evaluation sweep.",
    )
    return parser


def main():
    parser = build_slot_parser()
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.output_path is None:
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        args.output_path = f"output/robust_eval/F_slot_kmerge/{ts}"

    if args.plot_only:
        plot_run_results(args.output_path)
        return

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    device = "cuda"

    print("Loading model…")
    model, tokenizer = load_model_and_tokenizer(
        args.model_path, low_cpu_mem_usage=True, use_cache=False, device=device,
    )

    runner = SlotAttackRunner(args, model, tokenizer)
    runner.run()


if __name__ == "__main__":
    main()
