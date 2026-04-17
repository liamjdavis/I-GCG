#!/bin/bash
#SBATCH --job-name=advbench-gold
#SBATCH --array=0-7
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/advbench_gold_%A_%a.out
#SBATCH --error=logs/advbench_gold_%A_%a.err
# =============================================================================
# AdvBench gold-completions generation (abliterated Qwen2.5).
#
# One task = one GPU = one shard of the 520 AdvBench behaviours.  With
# WORLD=8 the full sweep finishes in ~15 min/shard of wall clock.
#
# Before launching:
#   1) Build the AdvBench JSON once:
#        python scripts/build_advbench_json.py
#   2) Create logs dir:
#        mkdir -p logs
#   3) Submit:
#        sbatch scripts/slurm/advbench_gold.sbatch
#   4) When all array tasks finish, merge shard JSONs:
#        python scripts/generate_gold_advbench.py --merge_only \
#            --shard_world_size 8 \
#            --output data/gold_harmful_completions_advbench.json
#
# Single-node multi-GPU alternative (no SLURM):
#   for r in 0 1 2 3 4 5 6 7; do
#     CUDA_VISIBLE_DEVICES=$r python scripts/generate_gold_advbench.py \
#         --shard_rank $r --shard_world_size 8 &
#   done; wait
# =============================================================================

set -euo pipefail

WORLD=${SLURM_ARRAY_TASK_COUNT:-8}
RANK=${SLURM_ARRAY_TASK_ID:-0}

# Point this at the Python used in your environment (miniconda3/iris, etc.).
PY=${PY:-python}

# Optional: set HF cache + model dirs (avoid per-task re-downloads).
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-0}
export TOKENIZERS_PARALLELISM=false

$PY scripts/generate_gold_advbench.py \
    --abliterated_model "huihui-ai/Qwen2.5-7B-Instruct-abliterated-v3" \
    --behaviors_config data/advbench_behaviors.json \
    --output data/gold_harmful_completions_advbench.json \
    --n_samples 100 \
    --max_new_tokens 512 \
    --temperature 0.8 --top_p 0.95 \
    --shard_rank "$RANK" \
    --shard_world_size "$WORLD" \
    --device 0
    # Uncomment to also judge the gold completions (requires a real
    # implementation in robust_gcg/judge_hooks.py):
    # --enable_judge
