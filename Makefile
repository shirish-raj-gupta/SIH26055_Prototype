# SmartScan -- SIH 26055
#
# `make reproduce` regenerates every headline number in the README from scratch.
# Every other target is a shortcut for something in `python -m smartscan.cli`.

PY      ?= python
CLI     := $(PY) -m smartscan.cli
SEEDS   ?= 30
JOBS    ?= 1
REPORTS ?= reports

.DEFAULT_GOAL := help
.PHONY: help install install-min demo smoke test test-fast test-acceptance lint fmt \
        benchmark benchmark-medium benchmark-easy benchmark-hard grid train-predictor train-ppo train-dqn \
        train-all estimate ablate reproduce reproduce-easy clean coverage \n        dataset dataset-smoke dataset-verify publish publish-dry publish-models \n        credentials external info

help:  ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# --------------------------------------------------------------------------- #
# Setup
# --------------------------------------------------------------------------- #
install:  ## Install with ML, viz and dev extras
	$(PY) -m pip install -e ".[ml,viz,dev]"

install-min:  ## Install core only (no torch): env, analytic schedulers, benchmark
	$(PY) -m pip install -e .

# --------------------------------------------------------------------------- #
# Quick looks
# --------------------------------------------------------------------------- #
demo:  ## Launch the live dashboard in a browser (offline, one command)
	$(PY) -m streamlit run dashboard/app.py

smoke:  ## ~1 min terminal smoke run (environment, schedulers, scan-on-scan)
	$(CLI) demo --config configs/easy.yaml

info:  ## Print the resolved MEDIUM config and its hash
	$(CLI) info --config configs/medium.yaml

# --------------------------------------------------------------------------- #
# Tests and quality
# --------------------------------------------------------------------------- #
test:  ## Full test suite
	$(PY) -m pytest -q

test-fast:  ## Everything except the slow acceptance tests
	$(PY) -m pytest -q -m "not slow"

test-acceptance:  ## Only the numbered acceptance tests
	$(PY) -m pytest -q -m acceptance

coverage:  ## Coverage gate: >=80% on env/ and analysis/
	$(PY) -m pytest -q --cov=smartscan/env --cov=smartscan/analysis \
	  --cov-report=term-missing --cov-fail-under=80

lint:  ## ruff check
	$(PY) -m ruff check smartscan tests

fmt:  ## ruff format + autofix
	$(PY) -m ruff check --fix smartscan tests
	$(PY) -m ruff format smartscan tests

# --------------------------------------------------------------------------- #
# Benchmarks
# --------------------------------------------------------------------------- #
benchmark:  ## Full grid -> results.parquet, leaderboard.md/.tex and all 7 figures
	$(CLI) ablate --config configs/medium.yaml --which reward --n-seeds 8 --out $(REPORTS)/ablation.json
	$(CLI) estimate --config configs/scan_on_scan.yaml --n-seeds 6 --out $(REPORTS)/scan_on_scan.json
	$(CLI) grid --tiers easy,medium,hard --n-seeds $(SEEDS) --n-jobs $(JOBS) --out $(REPORTS)
	@echo "artefacts -> $(REPORTS)/  (results.parquet, leaderboard.md, leaderboard.tex, f1..f7)"

benchmark-medium:  ## Paired benchmark on MEDIUM only
	$(CLI) benchmark --config configs/medium.yaml --n-seeds $(SEEDS) --n-jobs $(JOBS) --out $(REPORTS)

benchmark-easy:  ## Paired benchmark on EASY
	$(CLI) benchmark --config configs/easy.yaml --n-seeds $(SEEDS) --n-jobs $(JOBS) --out $(REPORTS)

benchmark-hard:  ## Paired benchmark on HARD
	$(CLI) benchmark --config configs/hard.yaml --n-seeds $(SEEDS) --n-jobs $(JOBS) --out $(REPORTS)

estimate:  ## Scan-period estimator validation (acceptance test 4)
	$(CLI) estimate --config configs/scan_on_scan.yaml --n-seeds 10 --out $(REPORTS)/scan_on_scan.json

ablate:  ## Reward / IBW / retune / density / belief sweeps
	$(CLI) ablate --config configs/medium.yaml --which all --n-seeds 8 --out $(REPORTS)/ablation.json

# --------------------------------------------------------------------------- #
# Training (needs the `ml` extra)
# --------------------------------------------------------------------------- #
train-predictor:  ## Train the occupancy predictor (with privileged distillation)
	$(CLI) train --config configs/medium.yaml --what predictor

train-ppo:  ## Train PPO
	$(CLI) train --config configs/medium.yaml --what ppo

train-dqn:  ## Train Double-DQN
	$(CLI) train --config configs/medium.yaml --what dqn

train-all: train-predictor train-ppo train-dqn  ## Train every learned scheduler

# --------------------------------------------------------------------------- #
# Reproduction
# --------------------------------------------------------------------------- #
reproduce-easy:  ## EASY tier end to end -- the <10 min live-demo path
	$(CLI) reproduce --tiers easy --n-seeds $(SEEDS) --n-jobs $(JOBS) --out $(REPORTS)
	$(CLI) estimate --config configs/scan_on_scan.yaml --n-seeds 6 --out $(REPORTS)/scan_on_scan.json

reproduce:  ## Regenerate every headline number (analytic schedulers, all tiers)
	$(CLI) reproduce --tiers easy,medium,hard --n-seeds $(SEEDS) --n-jobs $(JOBS) --out $(REPORTS)
	$(CLI) estimate --config configs/scan_on_scan.yaml --n-seeds 10 --out $(REPORTS)/scan_on_scan.json
	$(CLI) ablate --config configs/medium.yaml --which all --n-seeds 8 --out $(REPORTS)/ablation.json
	@echo "headline numbers -> $(REPORTS)/leaderboard.md"

# --------------------------------------------------------------------------- #
# Dataset and publication
# --------------------------------------------------------------------------- #
dataset:  ## Build the full 3000-episode corpus (1000/1200/800). Takes ~40 min.
	$(PY) -c "from smartscan.data.dataset_builder import build_dataset; build_dataset('build/dataset', n_jobs=$(JOBS))"

dataset-smoke:  ## Build a tiny corpus to exercise the pipeline (~10 s)
	$(PY) -c "from smartscan.data.dataset_builder import build_dataset; build_dataset('build/smoke', counts={'easy':3,'medium':3,'hard':2})"

dataset-verify:  ## Check dataset integrity, splits and byte accounting
	$(PY) -c "import json;from smartscan.data.kaggle_io import verify_dataset;print(json.dumps(verify_dataset('build/dataset'),indent=2))"

credentials:  ## Report which credentials are configured (never prints a value)
	$(CLI) credentials

publish-dry:  ## Preflight the Kaggle upload without sending anything
	$(PY) scripts/publish_kaggle.py --dry-run

publish:  ## Create or version the PUBLIC Kaggle dataset (prompts before uploading)
	$(PY) scripts/publish_kaggle.py

publish-models:  ## Publish trained checkpoints as a second Kaggle dataset
	$(PY) scripts/publish_kaggle.py --what models

external:  ## External validation against the gated Turing dataset (needs HF_TOKEN)
	$(CLI) external --config configs/medium.yaml --n-records 4

clean:  ## Remove generated artefacts (checkpoints are kept)
	rm -rf $(REPORTS) runs/*/metrics_*.json .pytest_cache .ruff_cache build/smoke
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
