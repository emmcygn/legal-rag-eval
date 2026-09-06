.PHONY: help install install-all lint format typecheck test test-fast benchmark benchmark-structural report dashboard clean ci

PYTHON ?= python
SRC = src/scaffolder
TESTS = tests

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install package with dev dependencies
	$(PYTHON) -m pip install -e ".[dev]"

install-all:  ## Install package with all dependencies
	$(PYTHON) -m pip install -e ".[all]"

lint:  ## Run ruff linter
	$(PYTHON) -m ruff check $(SRC) $(TESTS)
	$(PYTHON) -m ruff format --check $(SRC) $(TESTS)

format:  ## Auto-format code with ruff
	$(PYTHON) -m ruff format $(SRC) $(TESTS)
	$(PYTHON) -m ruff check --fix $(SRC) $(TESTS)

typecheck:  ## Run mypy type checker
	$(PYTHON) -m mypy $(SRC)

test:  ## Run tests with coverage
	$(PYTHON) -m pytest

test-fast:  ## Run tests without coverage
	$(PYTHON) -m pytest --no-cov -x

benchmark:  ## Run the deterministic anchored-evidence benchmark
	$(PYTHON) -m scaffolder benchmark

benchmark-structural:  ## Run deprecated SDK-generated structural diagnostics
	$(PYTHON) -m scaffolder benchmark-legacy --no-embed

report:  ## Generate HTML report from benchmark results
	$(PYTHON) -m scaffolder report

dashboard:  ## Launch Streamlit dashboard
	$(PYTHON) -m streamlit run $(SRC)/dashboard/app.py

clean:  ## Remove build artifacts and caches
	rm -rf build/ dist/ *.egg-info .mypy_cache .pytest_cache .ruff_cache
	rm -rf .cache/embeddings
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

ci:  ## Run all CI checks (lint + typecheck + test)
	$(MAKE) lint
	$(MAKE) typecheck
	$(MAKE) test
