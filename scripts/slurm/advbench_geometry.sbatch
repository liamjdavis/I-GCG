#!/bin/bash
#SBATCH --job-name=advbench-geom
#SBATCH --array=0-7
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=logs/advbench_geom_%A_%a.out
#SBATCH --error=logs/advbench_geom_%A_%a.err
# =============================================================================
# AdvBench geometry evaluation: F-C+D and SlotGCG on the aligned model.
#
# One task = one GPU = one shard of the 520 AdvBench behaviours x 2 conditions
# (fcd, slot_igcg).  Expect ~32 h/shard on A100 80GB at 500 steps each.
#
# Prereqs:
#   - scripts/build_advbench_json.py has been run.
#   - Gold completions exist at data/gold_harmful_completions_advbench.json
#     (produced by advbench_gold.sbatch + merge step).
#   - logs/ directory exists.
#
# Submit all shards:
#     sbatch scripts/slurm/advbench_geometry.sbatch
#
# When all tasks finish, merge the per-shard reports:
#     python scripts/geometry_eval_advbench.py --merge_only \
#         --output_dir output/geometry_eval/advbench/<SLURM_JOB_ID> \
#         --shard_world_size 8
#
# Single-node multi-GPU alternative (no SLURM):
#   TS=$(date +%Y%m%d-%H%M%S)
#   for r in 0 1 2 3 4 5 6 7; do
#     CUDA_VISIBLE_DEVICES=$r python scripts/geometry_eval_advbench.py \
#         --aligned_model Qwen/Qwen2.5-7B-Instruct \
#         --gold_path data/gold_harmful_completions_advbench.json \
#         --shard_rank $r --shard_world_size 8 \
#         --output_dir output/geometry_eval/advbench/$TS &
#   done; wait
# =============================================================================

set -euo pipefail

WORLD=${SLURM_ARRAY_TASK_COUNT:-8}
RANK=${SLURM_ARRAY_TASK_ID:-0}
JOB=${SLURM_JOB_ID:-$(date +%Y%m%d-%H%M%S)}

PY=${PY:-python}

export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-0}
export TOKENIZERS_PARALLELISM=false

# Shared output_dir across all array tasks so they write into the same run
# tree and the --merge_only pass can find every shard.  The job-id-keyed
# name makes re-submits non-colliding.
OUT_DIR="output/geometry_eval/advbench/${JOB}"

$PY scripts/geometry_eval_advbench.py \
    --aligned_model "Qwen/Qwen2.5-7B-Instruct" \
    --behaviors_config data/advbench_behaviors.json \
    --gold_path data/gold_harmful_completions_advbench.json \
    --output_dir "$OUT_DIR" \
    --shard_rank "$RANK" \
    --shard_world_size "$WORLD" \
    --n_gen_attacked 50 \
    --n_gen_refused 50 \
    --device 0
    # Optional toggles:
    # --enable_mauve           # requires mauve-text, >=20 samples per side
    # --enable_strongreject    # requires strong-reject package
    # --enable_judge           # requires robust_gcg/judge_hooks.py impl
