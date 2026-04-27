.PHONY: help install install-dev download-models download-datasets dataset-status dataset-validate eval-reid-baseline run-backend run-frontend run test clean docker-build docker-up docker-down lint format rust-build rust-install rust-test rust-bench rust-clean

DATA_ROOT ?= data/datasets/cattle/cattely
RUST_WORKSPACE ?= rust_core
RUST_CRATE ?= rust_core/crates/agroai_core

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install runtime dependencies
	pip install -r requirements.txt

install-dev:  ## Install dev extras (fine-tuning, notebooks)
	pip install -r requirements-dev.txt
	python -m pip install maturin
	$(MAKE) rust-install

download-models:  ## Download YOLOv8 + warm ResNet50 weights
	python scripts/download_models.py

download-datasets:  ## List verified livestock datasets (see docs/DATASETS.md)
	python scripts/download_datasets.py --list

dataset-status:  ## Show installation status for all datasets
	python scripts/dataset_status.py

dataset-validate:  ## Validate dataset payloads and fail on broken auto datasets
	python scripts/validate_datasets.py

eval-reid-baseline:  ## Run baseline ReID metrics (top-1 + ROC-AUC) on identity folders
	python scripts/eval_reid_baseline.py --data-root "$(DATA_ROOT)"

rust-build:  ## Build Rust core in release mode
	cd "$(RUST_WORKSPACE)" && cargo build --release

rust-install:  ## Build and install agroai_core into active Python environment
	cd "$(RUST_CRATE)" && python -m maturin develop --release

rust-test:  ## Run Rust unit/integration tests
	cd "$(RUST_WORKSPACE)" && cargo test --all

rust-bench:  ## Run Rust criterion benches
	cd "$(RUST_CRATE)" && cargo bench

rust-clean:  ## Clean Rust artifacts
	cd "$(RUST_WORKSPACE)" && cargo clean

run-backend:  ## Run FastAPI backend (port 8000)
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

run-frontend:  ## Run Streamlit frontend (port 8501)
	streamlit run frontend/app.py

run:  ## Run backend and frontend concurrently (use: make -j2 run)
	$(MAKE) -j2 run-backend run-frontend

test:  ## Run pytest with coverage
	pytest -v --cov=app --cov-report=term-missing

lint:  ## Lint with ruff + black --check
	ruff check app tests || true
	black --check app tests || true

format:  ## Auto-format
	black app tests
	ruff check --fix app tests

clean:  ## Remove caches, pyc, local databases
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -f *.db *.sqlite *.sqlite3
	rm -rf .pytest_cache .coverage htmlcov .ruff_cache

docker-build:  ## Build docker images
	docker compose build

docker-up:  ## Start docker stack
	docker compose up

docker-down:  ## Stop docker stack
	docker compose down
