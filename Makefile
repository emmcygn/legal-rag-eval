.PHONY: help install install-all lint format typecheck test test-fast gold gold-check benchmark benchmark-structural benchmark-embed readme readme-check compare-builds report dashboard clean ci

PYTHON ?= python
SRC = src/scaffolder
TESTS = tests

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-22s\033[0m %s\n", $$1, $$2}'

install:  ## Install package with dev dependencies
	$(PYTHON) -m pip install -e ".[dev]"

install-all:  ## Install package with all dependencies
	$(PYTHON) -m pip install -e ".[all]"

lint:  ## Run ruff linter
	$(PYTHON) -m ruff check $(SRC) $(TESTS) scripts
	$(PYTHON) -m ruff format --check $(SRC) $(TESTS) scripts

format:  ## Auto-format code with ruff
	$(PYTHON) -m ruff format $(SRC) $(TESTS) scripts
	$(PYTHON) -m ruff check --fix $(SRC) $(TESTS) scripts

typecheck:  ## Run mypy type checker
	$(PYTHON) -m mypy $(SRC)

test:  ## Run tests with coverage
	$(PYTHON) -m pytest

test-fast:  ## Run tests without coverage
	$(PYTHON) -m pytest --no-cov -x

gold:  ## Re-seed the gold annotations (refuses to overwrite hand-corrected files)
	$(PYTHON) scripts/build_gold.py

gold-check:  ## Report where the committed gold annotations differ from a fresh seeding
	$(PYTHON) scripts/build_gold.py --check

benchmark: benchmark-structural  ## Alias for benchmark-structural

benchmark-structural:  ## Structural metrics against the gold annotations (no embeddings, seconds)
	$(PYTHON) -m scaffolder benchmark --json

benchmark-embed:  ## Full benchmark: chunking + embedding + retrieval + statistics (minutes)
	$(PYTHON) -m scaffolder benchmark-embed --json

readme:  ## Regenerate the README results section from the committed fixed-build run
	$(PYTHON) scripts/update_readme.py --results results/lexichunk_fixed/full_benchmark.json

readme-check:  ## Fail if the README results section no longer matches that run
	$(PYTHON) scripts/update_readme.py --check --results results/lexichunk_fixed/full_benchmark.json

compare-builds:  ## Side-by-side table for the two committed LexiChunk builds
	$(PYTHON) scripts/compare_builds.py 		--before results/lexichunk_baseline/full_benchmark.json --before-name baseline 		--after results/lexichunk_fixed/full_benchmark.json --after-name fixed

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
