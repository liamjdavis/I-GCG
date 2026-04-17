# =============================================================================
# Makefile — common tasks for MLAI2026COLUMBIA on RunPod
#
# All targets assume:
#   - source start_session.sh has been run (sets RESOLVED_MODEL_PATH)
#   - Network volume mounted at /workspace
# =============================================================================

SHELL := /bin/bash
NETWORK_VOL ?= /workspace
REPO_DIR ?= $(NETWORK_VOL)/MLAI2026COLUMBIA
RESULTS_BACKUP ?= $(NETWORK_VOL)/results_backup

# Auto-resolve model path; override with: make attack-single MODEL_PATH=/path/to/model
MODEL_PATH ?= $(RESOLVED_MODEL_PATH)

# Defaults
BEHAVIOR_ID ?= 1
DEVICE ?= 0
DEFENSE ?= no_defense
CONFIG ?= behaviors_config.json
OURS_CONFIG ?= behaviors_ours_config.json
INIT_CONFIG ?= behaviors_ours_config_init.json

METHODS ?= A,B,C,D,F
CYBER_CONFIG ?= data/cyber_behaviors.json

.PHONY: help status attack-single attack-all attack-ours robust-a robust-b robust-c robust-d robust-f \
        generate-config backup-results clean-output tmux-run smoke-test quick-eval \
        thorough-D thorough-D-dry experiment-improved experiment-improved-dry download-vicuna \
        transfer-experiment transfer-experiment-dry \
        slotgcg-experiment slotgcg-experiment-dry \
        target-ablation target-ablation-quick target-ablation-dry \
        fc-scaled fc-scaled-quick fc-scaled-dry \
        fcd-report fcd-report-quick fcd-report-dry \
        gold-completions gold-completions-smoke \
        geometry-smoke geometry-full geometry-full-quick geometry-full-dry geometry-offline

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# ── Status ──────────────────────────────────────────────────────────────────

status: ## GPU utilization, VRAM, disk usage
	@echo "=== GPU ==="
	@nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu \
		--format=csv,noheader 2>/dev/null || echo "No GPU found"
	@echo ""
	@echo "=== Disk ==="
	@df -h $(NETWORK_VOL) 2>/dev/null || echo "Network volume not mounted"
	@echo ""
	@echo "=== Running Python processes ==="
	@ps aux | grep '[p]ython.*attack\|[p]ython.*robust_gcg\|[p]ython.*smooth' | head -5 || echo "None"

# ── Base GCG attack ────────────────────────────────────────────────────────

attack-single: ## Run single behavior: make attack-single BEHAVIOR_ID=1
	python igcg_upstream/attack_llm_core_base.py \
		--model_path $(MODEL_PATH) \
		--device $(DEVICE) \
		--id $(BEHAVIOR_ID) \
		--defense $(DEFENSE) \
		--behaviors_config igcg_upstream/$(CONFIG)

attack-all: ## Run all 50 behaviors serially (adapts multi-GPU script for 1 GPU)
	python igcg_upstream/run_multiple_attack_our_target.py \
		--defense $(DEFENSE) \
		--behaviors_config igcg_upstream/$(CONFIG)

# ── I-GCG (our target) ─────────────────────────────────────────────────────

attack-ours-init: ## Step 1: Generate initial suffixes with I-GCG
	python igcg_upstream/attack_llm_core_best_update_our_target.py \
		--model_path $(MODEL_PATH) \
		--behaviors_config igcg_upstream/$(OURS_CONFIG)

generate-config: ## Step 2: Build config with initialized suffixes
	python igcg_upstream/generate_our_config.py

attack-ours: ## Step 3: Run I-GCG with initialized config
	python igcg_upstream/run_multiple_attack_our_target.py \
		--behaviors_config igcg_upstream/$(INIT_CONFIG)

# ── Robust GCG variants ────────────────────────────────────────────────────

robust-a: ## Robust GCG A: suffix char perturbation
	python scripts/robust_gcg_A_suffix_charperturb.py \
		--model_path $(MODEL_PATH) \
		--device $(DEVICE) \
		--id $(BEHAVIOR_ID)

robust-b: ## Robust GCG B: token perturbation
	python scripts/robust_gcg_B_token_perturb.py \
		--model_path $(MODEL_PATH) \
		--device $(DEVICE) \
		--id $(BEHAVIOR_ID)

robust-c: ## Robust GCG C: generation eval (expensive)
	python scripts/robust_gcg_C_generation_eval.py \
		--model_path $(MODEL_PATH) \
		--device $(DEVICE) \
		--id $(BEHAVIOR_ID)

robust-d: ## Robust GCG D: inert buffer
	python scripts/robust_gcg_D_inert_buffer.py \
		--model_path $(MODEL_PATH) \
		--device $(DEVICE) \
		--id $(BEHAVIOR_ID)

robust-f: ## Robust GCG F: SlotGCG positional insertion + K-merge
	python scripts/robust_gcg_F_slot_kmerge.py \
		--model_path $(MODEL_PATH) \
		--device $(DEVICE) \
		--id $(BEHAVIOR_ID)

# ── SmoothLLM defense evaluation ───────────────────────────────────────────

attack-with-smooth: ## Run attack + SmoothLLM defense eval
	python igcg_upstream/attack_llm_core_base.py \
		--model_path $(MODEL_PATH) \
		--device $(DEVICE) \
		--id $(BEHAVIOR_ID) \
		--defense smooth_llm \
		--behaviors_config igcg_upstream/$(CONFIG)

# ── Fast evaluation (smoke / quick) ─────────────────────────────────────────

smoke-test: ## Smoke test: 5 behaviors x 100 steps (~30 min on A100)
	python scripts/fast_robust_eval.py \
		--tier smoke \
		--methods $(METHODS) \
		--model_path $(MODEL_PATH) \
		--device $(DEVICE) \
		--behaviors_config $(CYBER_CONFIG)

quick-eval: ## Quick eval: 15 behaviors x 200 steps (~3 h on A100)
	python scripts/fast_robust_eval.py \
		--tier quick \
		--methods $(METHODS) \
		--model_path $(MODEL_PATH) \
		--device $(DEVICE) \
		--behaviors_config $(CYBER_CONFIG)

# ── Thorough Method D evaluation ───────────────────────────────────────────

thorough-D: ## Thorough Method D: 15 behaviors x 3 seeds x 500 steps (~8.5 h on A100)
	python scripts/thorough_method_D_eval.py \
		--model_path $(MODEL_PATH) \
		--device $(DEVICE) \
		--behaviors_config $(CYBER_CONFIG)

thorough-D-dry: ## Dry-run thorough Method D (1 behavior, 5 steps)
	python scripts/thorough_method_D_eval.py \
		--model_path $(MODEL_PATH) \
		--device $(DEVICE) \
		--behaviors_config $(CYBER_CONFIG) \
		--dry_run

# ── Improved GCG experiment ─────────────────────────────────────────────────

experiment-improved: ## Full improved experiment (~10h on A100)
	python scripts/improved_gcg_experiment.py \
		--model_path $(MODEL_PATH) \
		--device $(DEVICE)

experiment-improved-dry: ## Dry-run improved experiment (5 steps per run)
	python scripts/improved_gcg_experiment.py \
		--model_path $(MODEL_PATH) \
		--device $(DEVICE) \
		--dry_run

download-vicuna: ## Download Vicuna-7B-v1.5 for baseline testing
	python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
		AutoTokenizer.from_pretrained('lmsys/vicuna-7b-v1.5', cache_dir='/workspace/models'); \
		AutoModelForCausalLM.from_pretrained('lmsys/vicuna-7b-v1.5', cache_dir='/workspace/models', torch_dtype=__import__('torch').float16, low_cpu_mem_usage=True)"

# ── B1 Suffix Transfer Experiment ───────────────────────────────────────────

transfer-experiment: ## Full B1 transfer experiment (~7.5h on A100)
	python scripts/transfer_experiment.py \
		--model_path $(MODEL_PATH) \
		--device $(DEVICE)

transfer-experiment-dry: ## Dry-run transfer experiment (5 steps per run)
	python scripts/transfer_experiment.py \
		--model_path $(MODEL_PATH) \
		--device $(DEVICE) \
		--dry_run

# ── SlotGCG Experiment ──────────────────────────────────────────────────────

slotgcg-experiment: ## SlotGCG on v2 cyber behaviours (~5h on A100)
	python scripts/slotgcg_experiment.py \
		--model_path $(MODEL_PATH) \
		--device $(DEVICE)

slotgcg-experiment-dry: ## Dry-run SlotGCG experiment (5 steps per run)
	python scripts/slotgcg_experiment.py \
		--model_path $(MODEL_PATH) \
		--device $(DEVICE) \
		--dry_run

# ── Target Ablation Experiment ──────────────────────────────────────────────

target-ablation: ## Full verification-gap ablation (~4 h on A100)
	python scripts/target_ablation_experiment.py \
		--model_path $(MODEL_PATH) \
		--device $(DEVICE)

target-ablation-quick: ## Quick ablation (BIDs 3/4/5, 200 steps, ~3 h)
	python scripts/target_ablation_experiment.py \
		--model_path $(MODEL_PATH) \
		--device $(DEVICE) \
		--quick

target-ablation-dry: ## Dry-run ablation (1 behaviour, 5 steps per condition)
	python scripts/target_ablation_experiment.py \
		--model_path $(MODEL_PATH) \
		--device $(DEVICE) \
		--dry_run

# ── F-C Scaled Experiment ──────────────────────────────────────────────────

fc-scaled: ## F-C on 30 behaviors x 500 steps + SmoothLLM (~10 h on A100)
	python scripts/fc_scaled_experiment.py \
		--model_path $(MODEL_PATH) \
		--device $(DEVICE)

fc-scaled-quick: ## F-C on 5 behaviors x 200 steps (~1 h)
	python scripts/fc_scaled_experiment.py \
		--model_path $(MODEL_PATH) \
		--device $(DEVICE) \
		--quick

fc-scaled-dry: ## Dry-run F-C scaled (1 behaviour, 5 steps)
	python scripts/fc_scaled_experiment.py \
		--model_path $(MODEL_PATH) \
		--device $(DEVICE) \
		--dry_run

# ── F-C+D Hybrid Loss Experiment ─────────────────────────────────────────────

fcd-scaled: ## F-C+D hybrid on 40 behaviors x 500 steps (~12 h on A100)
	python scripts/fcd_scaled_experiment.py \
		--model_path $(MODEL_PATH) \
		--device $(DEVICE)

fcd-scaled-quick: ## F-C+D hybrid on 5 behaviors x 200 steps (~1 h)
	python scripts/fcd_scaled_experiment.py \
		--model_path $(MODEL_PATH) \
		--device $(DEVICE) \
		--quick

fcd-scaled-dry: ## Dry-run F-C+D (1 behaviour, 5 steps)
	python scripts/fcd_scaled_experiment.py \
		--model_path $(MODEL_PATH) \
		--device $(DEVICE) \
		--dry_run

# ── F-C+D Attack Report ─────────────────────────────────────────────────

fcd-report: ## F-C+D on 40 behaviors + formatted .txt report (~12 h on A100)
	python scripts/fcd_attack_report.py \
		--model_path $(MODEL_PATH) \
		--device $(DEVICE)

fcd-report-quick: ## F-C+D report on 5 behaviors x 200 steps (~1 h)
	python scripts/fcd_attack_report.py \
		--model_path $(MODEL_PATH) \
		--device $(DEVICE) \
		--quick

fcd-report-dry: ## Dry-run F-C+D report (1 behaviour, 5 steps)
	python scripts/fcd_attack_report.py \
		--model_path $(MODEL_PATH) \
		--device $(DEVICE) \
		--dry_run

# ── Analysis ────────────────────────────────────────────────────────────────

analyze: ## Run result analysis and generate plots
	python scripts/analyze_results.py

# ── Utilities ───────────────────────────────────────────────────────────────

backup-results: ## Copy output/ to network volume
	@mkdir -p $(RESULTS_BACKUP)
	rsync -av --progress output/ $(RESULTS_BACKUP)/
	rsync -av --progress Our_GCG_target_len_20/ $(RESULTS_BACKUP)/Our_GCG_target_len_20/ 2>/dev/null || true
	@echo "Results backed up to $(RESULTS_BACKUP)"

clean-output: ## Remove local output (keeps network backup)
	@echo "This will delete output/ and Our_GCG_target_len_20/ locally."
	@read -p "Continue? [y/N] " ans && [ "$${ans}" = "y" ] && \
		rm -rf output/ Our_GCG_target_len_20/ && echo "Cleaned." || echo "Aborted."

tmux-run: ## Start a tmux session and run attack-all inside it
	tmux new-session -d -s gcg_attack "cd $(REPO_DIR) && source start_session.sh && make attack-all; exec bash"
	@echo "Attack running in tmux session 'gcg_attack'. Attach with: tmux attach -t gcg_attack"

vram: ## Show live VRAM usage
	@watch -n 1 nvidia-smi

# ── Gold Completions (abliterated model) ─────────────────────────────────

gold-completions: ## Generate 100 gold completions per behavior (~2-3 h on A100)
	python scripts/generate_gold_completions.py \
		--abliterated_model huihui-ai/Qwen2.5-7B-Instruct-abliterated-v3 \
		--behaviors_config data/cyber_behaviors_v2_all40.json \
		--n_samples 100 --device $(DEVICE)

gold-completions-smoke: ## Generate 10 gold completions for 9 diverse BIDs (~10 min)
	python scripts/generate_gold_smoke.py --device $(DEVICE)

# ── Geometry Evaluation ──────────────────────────────────────────────────

geometry-smoke: ## Geometry smoke test: 3 BIDs, 5 steps, 3 conditions (~15 min)
	python scripts/geometry_eval_smoke.py \
		--model_path $(MODEL_PATH) --device $(DEVICE)

geometry-full: ## Full geometry eval: 40 BIDs, 500 steps, 3 conditions (~36 h)
	python scripts/geometry_eval_full.py \
		--model_path $(MODEL_PATH) --device $(DEVICE)

geometry-full-quick: ## Quick geometry eval: 5 BIDs, 200 steps (~3 h)
	python scripts/geometry_eval_full.py \
		--model_path $(MODEL_PATH) --device $(DEVICE) --quick

geometry-full-dry: ## Dry-run geometry eval: 1 BID, 5 steps
	python scripts/geometry_eval_full.py \
		--model_path $(MODEL_PATH) --device $(DEVICE) --dry_run

geometry-offline: ## Offline re-score existing F-C+D results (~30 min)
	python scripts/geometry_eval_offline.py \
		--model_path $(MODEL_PATH) --device $(DEVICE)
