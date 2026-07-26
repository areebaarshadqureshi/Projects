# Karachi Real Estate Fraud Detection - Makefile
# ================================================

.PHONY: help install install-api install-dev notebook notebook-lab \
        pipeline api api-dev test test-api coverage lint format \
        download-artifacts validate-models docker-build docker-up \
        docker-up-detached docker-down docker-logs docker-restart docker-ps \
        docker-clean clean clean-notebooks clean-docker clean-all check-deps version info

# Variables
PYTHON := python
PIP := pip
JUPYTER := jupyter
UVICORN := uvicorn
DOCKER_COMPOSE := docker-compose

# Paths
API_MODULE := api.main:app
NOTEBOOKS_DIR := notebooks
MODELS_DIR := models
DATA_DIR := data

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

#==========================================
# Help Target (Default)
#==========================================

help:  ## Show this help message
	@echo ""
	@echo "$(BLUE)Karachi Real Estate Fraud Detection - Available Commands$(NC)"
	@echo "=========================================================="
	@echo ""
	@echo "$(GREEN)Setup Commands:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; /^install/ || /^download/ {printf "  $(BLUE)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Development Commands:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; /^notebook/ || /^pipeline/ || /^api/ || /^test/ || /^coverage/ || /^lint/ || /^format/ || /^validate/ {printf "  $(BLUE)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Docker Commands:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; /^docker/ {printf "  $(BLUE)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Cleaning Commands:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; /^clean/ {printf "  $(BLUE)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(YELLOW)Quick Start:$(NC)"
	@echo "  make install-api  - Install API dependencies"
	@echo "  make api          - Start the API server"
	@echo "  make docker-up    - Run with Docker"
	@echo ""
	@echo "$(YELLOW)Retrain everything from scratch:$(NC)"
	@echo "  make pipeline     - Run notebooks 01 through 06 non-interactively"
	@echo ""

#==========================================
# Setup Commands
#==========================================

install:  ## Install all dependencies (full environment: training + API)
	@echo "$(BLUE)Installing full dependencies...$(NC)"
	$(PIP) install -r requirements.txt
	@echo "$(GREEN)✓ Full dependencies installed$(NC)"

install-api:  ## Install API-only dependencies (faster, for deployment)
	@echo "$(BLUE)Installing API dependencies...$(NC)"
	$(PIP) install -r requirements-api.txt
	@echo "$(GREEN)✓ API dependencies installed$(NC)"

install-dev: install  ## Install linting/formatting/coverage tools on top of the full env
	@echo "$(BLUE)Installing development tooling...$(NC)"
	$(PIP) install ruff black pytest-cov
	@echo "$(GREEN)✓ Development tooling installed$(NC)"

download-artifacts:  ## Fetch large training-only files (raw data, extra candidate models)
	@echo "$(BLUE)Downloading training-only artifacts...$(NC)"
	bash scripts/download_artifacts.sh

#==========================================
# Development Commands
#==========================================

notebook:  ## Launch Jupyter notebook server
	@echo "$(BLUE)Starting Jupyter notebook server...$(NC)"
	@echo "$(YELLOW)Run notebooks in sequence: 01 → 02 → 03 → 04 → 05 → 06$(NC)"
	$(JUPYTER) notebook

notebook-lab:  ## Launch Jupyter Lab (modern interface)
	@echo "$(BLUE)Starting Jupyter Lab...$(NC)"
	$(JUPYTER) lab

pipeline:  ## Run the full 01→06 notebook pipeline non-interactively (retrains everything)
	@echo "$(BLUE)Running full notebook pipeline (01 → 06)...$(NC)"
	@echo "$(YELLOW)This retrains all models and regenerates all artifacts. Takes several minutes.$(NC)"
	$(JUPYTER) nbconvert --to notebook --execute --inplace $(NOTEBOOKS_DIR)/01-data_cleaning.ipynb
	$(JUPYTER) nbconvert --to notebook --execute --inplace $(NOTEBOOKS_DIR)/02-eda.ipynb
	$(JUPYTER) nbconvert --to notebook --execute --inplace $(NOTEBOOKS_DIR)/03-feature_engineering.ipynb
	$(JUPYTER) nbconvert --to notebook --execute --inplace $(NOTEBOOKS_DIR)/04-unsupervised_model.ipynb
	$(JUPYTER) nbconvert --to notebook --execute --inplace $(NOTEBOOKS_DIR)/05-supervised_models.ipynb
	$(JUPYTER) nbconvert --to notebook --execute --inplace $(NOTEBOOKS_DIR)/06-model_comparison_and_hyperparameter_tuning.ipynb
	@echo "$(GREEN)✓ Pipeline complete — models/ and data/preprocessed/ regenerated$(NC)"

api: validate-models  ## Start API server (production mode)
	@echo "$(BLUE)Starting API server...$(NC)"
	@echo "$(GREEN)Frontend:  http://localhost:8000$(NC)"
	@echo "$(GREEN)API Docs:  http://localhost:8000/docs$(NC)"
	@echo "$(GREEN)Health:    http://localhost:8000/health$(NC)"
	$(UVICORN) $(API_MODULE) --host 0.0.0.0 --port 8000

api-dev: validate-models  ## Start API server (development mode with auto-reload)
	@echo "$(BLUE)Starting API server in development mode...$(NC)"
	@echo "$(YELLOW)Auto-reload enabled - changes will restart server$(NC)"
	@echo "$(GREEN)Frontend:  http://localhost:8000$(NC)"
	@echo "$(GREEN)API Docs:  http://localhost:8000/docs$(NC)"
	$(UVICORN) $(API_MODULE) --host 0.0.0.0 --port 8000 --reload

test:  ## Run the full test suite (skips gracefully if the API server isn't running)
	@echo "$(BLUE)Running test suite...$(NC)"
	$(PYTHON) -m pytest tests/ -v

test-api:  ## Run API integration tests (requires a running server — see 'make api-dev')
	@echo "$(BLUE)Testing API endpoints...$(NC)"
	@echo "$(YELLOW)Make sure API is running (make api or make docker-up)$(NC)"
	$(PYTHON) -m pytest tests/test_api.py -v

coverage:  ## Run tests with a coverage report
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	$(PYTHON) -m pytest --cov=api --cov=src tests/ -v --cov-report=term-missing

lint:  ## Check code style with ruff
	@echo "$(BLUE)Linting...$(NC)"
	ruff check api src tests

format:  ## Auto-format code with black and ruff
	@echo "$(BLUE)Formatting...$(NC)"
	black api src tests
	ruff check --fix api src tests

validate-models:  ## Validate that model files exist
	@echo "$(BLUE)Validating model files...$(NC)"
	@if [ ! -f "$(MODELS_DIR)/fraud_detector_v1.pkl" ]; then \
		echo "$(RED)✗ Error: Model file not found at $(MODELS_DIR)/fraud_detector_v1.pkl$(NC)"; \
		echo "$(YELLOW)  Run 'make pipeline' or the notebooks 01-06 to train the model first$(NC)"; \
		exit 1; \
	fi
	@if [ ! -f "$(MODELS_DIR)/artifacts/scaler_robust_v1.pkl" ]; then \
		echo "$(RED)✗ Error: Artifacts not found in $(MODELS_DIR)/artifacts/$(NC)"; \
		echo "$(YELLOW)  Run notebook 03 to generate preprocessing artifacts$(NC)"; \
		exit 1; \
	fi
	@echo "$(GREEN)✓ Model files validated$(NC)"

#==========================================
# Docker Commands
#==========================================

docker-build:  ## Build Docker image
	@echo "$(BLUE)Building Docker image...$(NC)"
	$(DOCKER_COMPOSE) build
	@echo "$(GREEN)✓ Docker image built$(NC)"

docker-up: validate-models  ## Start services with Docker Compose
	@echo "$(BLUE)Starting Docker services...$(NC)"
	$(DOCKER_COMPOSE) up
	@echo "$(GREEN)✓ Services started$(NC)"

docker-up-detached: validate-models  ## Start services in background
	@echo "$(BLUE)Starting Docker services in background...$(NC)"
	$(DOCKER_COMPOSE) up -d
	@echo "$(GREEN)✓ Services running in background$(NC)"
	@echo "$(YELLOW)View logs with: make docker-logs$(NC)"
	@echo "$(YELLOW)Stop with: make docker-down$(NC)"

docker-down:  ## Stop and remove Docker containers
	@echo "$(BLUE)Stopping Docker services...$(NC)"
	$(DOCKER_COMPOSE) down
	@echo "$(GREEN)✓ Services stopped$(NC)"

docker-logs:  ## Show Docker container logs
	@echo "$(BLUE)Showing Docker logs...$(NC)"
	$(DOCKER_COMPOSE) logs -f

docker-restart:  ## Restart Docker services
	@echo "$(BLUE)Restarting Docker services...$(NC)"
	$(DOCKER_COMPOSE) restart
	@echo "$(GREEN)✓ Services restarted$(NC)"

docker-ps:  ## Show running Docker containers
	@echo "$(BLUE)Running containers:$(NC)"
	$(DOCKER_COMPOSE) ps

docker-clean:  ## Remove Docker containers, networks, and volumes
	@echo "$(BLUE)Cleaning Docker resources...$(NC)"
	$(DOCKER_COMPOSE) down -v --remove-orphans
	@echo "$(GREEN)✓ Docker resources cleaned$(NC)"

#==========================================
# Cleaning Commands
#==========================================

clean:  ## Clean Python cache and temporary files
	@echo "$(BLUE)Cleaning Python cache...$(NC)"
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '.pytest_cache' -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '*.egg-info' -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '.coverage' -delete
	@echo "$(GREEN)✓ Cache cleaned$(NC)"

clean-notebooks:  ## Clean notebook checkpoints
	@echo "$(BLUE)Cleaning notebook checkpoints...$(NC)"
	find . -type d -name '.ipynb_checkpoints' -exec rm -rf {} + 2>/dev/null || true
	@echo "$(GREEN)✓ Notebook checkpoints cleaned$(NC)"

clean-docker: docker-clean  ## Alias for docker-clean
	@echo "$(GREEN)✓ Docker cleaned$(NC)"

clean-all: clean clean-notebooks docker-clean  ## Clean everything (cache, notebooks, Docker)
	@echo "$(GREEN)✓ Everything cleaned$(NC)"

#==========================================
# Utility Commands
#==========================================

check-deps:  ## Check if required tools are installed
	@echo "$(BLUE)Checking dependencies...$(NC)"
	@which python > /dev/null || echo "$(RED)✗ Python not found$(NC)"
	@which pip > /dev/null || echo "$(RED)✗ pip not found$(NC)"
	@which docker > /dev/null || echo "$(YELLOW)⚠ Docker not found (optional)$(NC)"
	@which jupyter > /dev/null || echo "$(YELLOW)⚠ Jupyter not found (install with make install)$(NC)"
	@echo "$(GREEN)✓ Dependency check complete$(NC)"

version:  ## Show Python and package versions
	@echo "$(BLUE)Python version:$(NC)"
	@$(PYTHON) --version
	@echo ""
	@echo "$(BLUE)Key package versions:$(NC)"
	@$(PIP) show pandas numpy scikit-learn xgboost fastapi 2>/dev/null | grep -E "(Name|Version)" || echo "$(YELLOW)Run 'make install' first$(NC)"

info:  ## Show project information
	@echo ""
	@echo "$(BLUE)╔════════════════════════════════════════════════════╗$(NC)"
	@echo "$(BLUE)║  Karachi Real Estate Fraud Detection System       ║$(NC)"
	@echo "$(BLUE)╚════════════════════════════════════════════════════╝$(NC)"
	@echo ""
	@echo "$(GREEN)Dataset:$(NC)      46,667 Karachi listings (2018-2019)"
	@echo "$(GREEN)Model:$(NC)        XGBoost (Optuna-tuned, PR-AUC vs heuristic: 0.0716)"
	@echo "$(GREEN)Features:$(NC)     43 model features (47 engineered pre-selection)"
	@echo "$(GREEN)Threshold:$(NC)    0.5822 (CV-optimized, see README Evaluation Metrics)"
	@echo "$(GREEN)Detection:$(NC)    Ghost listings, price manipulation, geo anomalies"
	@echo ""
	@echo "$(YELLOW)Quick Start:$(NC)"
	@echo "  1. make install-api    # Install dependencies"
	@echo "  2. make api            # Start the API"
	@echo "  3. make test-api       # Test endpoints"
	@echo ""
	@echo "$(YELLOW)Or with Docker:$(NC)"
	@echo "  make docker-up         # Build and run"
	@echo ""

#==========================================
# Special Targets
#==========================================

.DEFAULT_GOAL := help

# Ensure commands run in bash shell
SHELL := /bin/bash
