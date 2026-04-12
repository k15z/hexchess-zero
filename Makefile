.PHONY: setup worker trainer elo dashboard status test bench lint docs-dev docs-build clean

# --- Training pipeline ---

setup: ## Install Python deps + build hexchess bindings
	uv sync --group dev
	uv run maturin develop --release -m bindings/python/Cargo.toml

worker: ## Run a self-play worker
	uv run python -m training worker

trainer: ## Run the continuous trainer
	uv run python -m training trainer

elo: ## Run the Elo rating service
	uv run python -m training elo-service

dashboard: ## Run the training dashboard
	uv run python -m training dashboard

status: ## Show training pipeline status
	uv run python -m training status

# --- Engine (Rust) ---

test: ## Run engine tests
	cargo test -p hexchess-engine

bench: ## Run move generation benchmark
	cargo bench --bench movegen

lint: ## Run clippy + format check + ruff
	cargo clippy -p hexchess-engine -- -D warnings
	cargo fmt --all --check
	uvx ruff check

# --- Documentation ---

docs-dev: ## Start documentation dev server (builds WASM first)
	cd docs && npm ci && npm run prepare-wasm:build && npm run dev

docs-build: ## Build documentation site (builds WASM first)
	cd docs && npm ci && npm run prepare-wasm:build && npm run build

# --- Docker ---

docker-up: ## Start all services via docker compose
	docker compose up --build -d

docker-down: ## Stop all services
	docker compose down

# --- Misc ---

clean: ## Remove build artifacts and caches
	rm -rf .cache/ .venv/ target/ bindings/wasm/pkg/

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'
