.PHONY: help install install-all lint format typecheck test test-fast gold gold-check benchmark benchmark-structural benchmark-embed benchmark-legacy evals evals-smoke readme readme-check compare-builds report dashboard clean ci

PYTHON ?= python
SRC = src/legal_rag_eval
TESTS = tests
TOOLS = tools

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-22s\033[0m %s\n", $$1, $$2}'

install:  ## Install package with dev dependencies
	$(PYTHON) -m pip install -e ".[dev]"

install-all:  ## Install package with all dependencies
	$(PYTHON) -m pip install -e ".[all]"

lint:  ## Run ruff linter
	$(PYTHON) -m ruff check $(SRC) $(TESTS) $(TOOLS)
	$(PYTHON) -m ruff format --check $(SRC) $(TESTS) $(TOOLS)

format:  ## Auto-format code with ruff
	$(PYTHON) -m ruff format $(SRC) $(TESTS) $(TOOLS)
	$(PYTHON) -m ruff check --fix $(SRC) $(TESTS) $(TOOLS)

typecheck:  ## Run mypy type checker
	$(PYTHON) -m mypy $(SRC)

test:  ## Run tests with coverage
	$(PYTHON) -m pytest

test-fast:  ## Run tests without coverage
	$(PYTHON) -m pytest --no-cov -x

gold:  ## Re-seed the gold annotations (refuses to overwrite hand-corrected files)
	$(PYTHON) tools/build_gold.py

gold-check:  ## Report where the committed gold annotations differ from a fresh seeding
	$(PYTHON) tools/build_gold.py --check

benchmark:  ## Deterministic anchored-evidence benchmark (offline, seconds)
	$(PYTHON) -m legal_rag_eval benchmark

benchmark-structural:  ## Structural metrics against the gold annotations (offline, seconds)
	$(PYTHON) -m legal_rag_eval benchmark-structural --json

benchmark-embed:  ## Full benchmark: chunking + embedding + retrieval + statistics (minutes)
	$(PYTHON) -m legal_rag_eval benchmark-embed --json

benchmark-legacy:  ## Superseded LexiChunk-derived structural diagnostics (circular, provenance only)
	$(PYTHON) -m legal_rag_eval benchmark-legacy --json

evals:  ## External LEDGAR + CUAD evaluations (downloads public datasets)
	$(PYTHON) -m legal_rag_eval.evals ledgar
	$(PYTHON) -m legal_rag_eval.evals cuad

evals-smoke:  ## Offline smoke run of the external evaluations (synthetic data, meaningless numbers)
	$(PYTHON) -m legal_rag_eval.evals ledgar --smoke
	$(PYTHON) -m legal_rag_eval.evals cuad --smoke

readme:  ## Regenerate the README results section from the committed fixed-build run
	$(PYTHON) tools/update_readme.py --results results/lexichunk_fixed/full_benchmark.json

readme-check:  ## Fail if the README results section no longer matches that run
	$(PYTHON) tools/update_readme.py --check --results results/lexichunk_fixed/full_benchmark.json

compare-builds:  ## Side-by-side table for the two committed LexiChunk builds
	$(PYTHON) tools/compare_builds.py 		--before results/lexichunk_baseline/full_benchmark.json --before-name baseline 		--after results/lexichunk_fixed/full_benchmark.json --after-name fixed

report:  ## Generate HTML report from benchmark results
	$(PYTHON) -m legal_rag_eval report

dashboard:  ## Launch the Streamlit dashboard (needs the `dashboard` extra)
	$(PYTHON) -m streamlit run $(SRC)/dashboard/app.py

clean:  ## Remove build artifacts and caches
	rm -rf build/ dist/ *.egg-info .mypy_cache .pytest_cache .ruff_cache
	rm -rf .cache/embeddings
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

ci:  ## Run all CI checks (lint + typecheck + test)
	$(MAKE) lint
	$(MAKE) typecheck
	$(MAKE) test
