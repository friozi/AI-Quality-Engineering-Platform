.PHONY: install test test-parallel test-accuracy test-reasoning test-safety ci help

# ── Developer targets ──────────────────────────────────────────────────────

install:        ## Install all Python dependencies
	pip install -r requirements.txt

test:           ## Run the full test suite (requires a running LLM server)
	pytest -v --tb=short

test-parallel:  ## Run tests in parallel with 2 workers
	pytest -n 2 -v --tb=short

test-accuracy:  ## Run accuracy evaluator tests only
	pytest -m accuracy -v --tb=short

test-reasoning: ## Run reasoning evaluator tests only
	pytest -m reasoning -v --tb=short

test-safety:    ## Run safety evaluator tests only (prompt injection + jailbreak)
	pytest -m safety -v --tb=short

# ── CI target ─────────────────────────────────────────────────────────────

ci:             ## Run offline tests only (no LLM required — for CI validation)
	pytest tests/test_phase5_reporters.py -v --tb=short

# ── Help ──────────────────────────────────────────────────────────────────

help:           ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*##"}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
