"""LLM-as-Judge hooks for the AdvBench geometry-eval pipeline.

Purpose
-------
This module is a deliberately **empty skeleton** that declares the single
interface the rest of the pipeline uses to talk to an LLM judge.  No working
judge is wired in right now.  Future agents should add an implementation
**here and only here** — no other file needs to change.

Two scripts call into this module (guarded by their own ``--enable_judge``
flag):

* ``scripts/geometry_eval_advbench.py``  -> ``score_attacked(behavior, texts)``
  runs the judge on each attacked-model generation for each behaviour and
  attaches the results to the geometry report under
  ``per_behavior[bid].conditions[cond].judge``.

* ``scripts/generate_gold_advbench.py``  -> ``score_abliterated(behavior, texts)``
  optionally judges the abliterated model's gold completions (a sanity
  check that the "cooperative" distribution really is harmful per the
  judge).  Results land under ``behaviors[bid].judge``.

How to plug in a real judge
---------------------------
Implement ``judge_generation`` below.  Minimum contract:

    def judge_generation(behavior_instruction, generation, reference_completions=None):
        return {"score": float_in_0_1, "label": "harmful"|"refused"|..., "raw": {...}}

Some concrete options, easiest -> hardest:

1. **strongREJECT** (already a dependency): delegate to
   :func:`robust_gcg.semantic_eval.compute_strongreject`.  Returns
   ``{refused, specific, convincing, score}`` — wrap into the contract.
2. **HarmBench classifier** (HuggingFace ``cais/HarmBench-Llama-2-13b-cls``):
   load once at import-time into a module-level ``_JUDGE_MODEL`` and call it
   in a ``torch.no_grad()`` block.
3. **OpenAI / Anthropic API**: use ``OPENAI_API_KEY`` / ``ANTHROPIC_API_KEY``
   from env, with exponential backoff.  ``openai`` is already in
   ``requirements.txt``.

Performance note
----------------
Both ``score_attacked`` and ``score_abliterated`` are called per-behaviour
with O(10-100) generations, so an unbatched per-string judge is fine for
AdvBench-scale (520 behaviours).  For a hot path, add a batched interface
next to ``judge_generation`` and call it instead.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Public contract
# ---------------------------------------------------------------------------

def judge_generation(
    behavior_instruction: str,
    generation: str,
    reference_completions: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Score a single (instruction, generation) pair.

    Returns
    -------
    dict with at least these keys:

    * ``score``        : float in [0, 1] (higher = more harmful) OR ``None``
    * ``label``        : short string tag (e.g. ``"harmful"``, ``"refused"``,
                          ``"not_implemented"``)
    * ``raw``          : dict with judge-specific fields (model name, logits,
                          API response id, etc.) — free-form

    The default implementation returns a sentinel so callers can detect
    that no judge is wired up yet.  DO NOT remove the sentinel without also
    implementing a real judge.
    """
    # TODO(agent): implement a real judge.  Example:
    #
    #   from robust_gcg.semantic_eval import compute_strongreject
    #   sr = compute_strongreject(behavior_instruction, generation)
    #   return {
    #       "score": sr["score"],
    #       "label": "refused" if sr["refused"] else "harmful",
    #       "raw": {"judge": "strongreject", **sr},
    #   }
    return {
        "score": None,
        "label": "not_implemented",
        "raw": {
            "behavior": behavior_instruction[:120],
            "generation_len": len(generation),
            "n_references": 0 if reference_completions is None else len(reference_completions),
        },
    }


# ---------------------------------------------------------------------------
# Pipeline-facing helpers (called by the two AdvBench scripts)
# ---------------------------------------------------------------------------

def score_attacked(
    behavior: Dict[str, Any],
    attacked_texts: List[str],
    reference_completions: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Judge a batch of attacked-model generations for one behaviour.

    Called from ``scripts/geometry_eval_advbench.py`` when ``--enable_judge``
    is set.  Returns a dict suitable for direct JSON dump:

        {"per_gen": [judge_generation(...), ...],
         "mean_score": float | None,
         "frac_harmful": float | None}
    """
    per_gen = [
        judge_generation(
            behavior["behaviour"], g,
            reference_completions=reference_completions,
        )
        for g in attacked_texts
    ]
    return _summarise(per_gen)


def score_abliterated(
    behavior: Dict[str, Any],
    gold_completions: List[str],
) -> Dict[str, Any]:
    """Judge the abliterated-model cooperative completions for one behaviour.

    Called from ``scripts/generate_gold_advbench.py`` when ``--enable_judge``
    is set.  Same return shape as :func:`score_attacked`.
    """
    per_gen = [
        judge_generation(behavior["behaviour"], g)
        for g in gold_completions
    ]
    return _summarise(per_gen)


def _summarise(per_gen: List[Dict[str, Any]]) -> Dict[str, Any]:
    scores = [r["score"] for r in per_gen if r.get("score") is not None]
    labels = [r.get("label") for r in per_gen]
    mean_score = float(sum(scores) / len(scores)) if scores else None
    frac_harmful = (
        float(sum(1 for l in labels if l == "harmful") / len(labels))
        if labels else None
    )
    return {
        "per_gen": per_gen,
        "mean_score": mean_score,
        "frac_harmful": frac_harmful,
        "n_scored": len(scores),
        "n_total": len(per_gen),
    }
