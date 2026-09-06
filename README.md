# Legal RAG Eval

Legal RAG Eval is a deterministic, offline benchmark for comparing chunk boundaries under a controlled retrieval and context budget. It scores retrieved source spans against independently stored evidence annotations instead of treating chunker output as gold.

The primary benchmark compares [LexiChunk](https://github.com/emmcygn/lexichunk), fixed token windows, and LangChain RecursiveCharacterTextSplitter using the same lexical cosine term-frequency ranker. It does not call an embedding model or paid API.

## Quick Start

Python 3.10–3.12 is supported.

```bash
git clone https://github.com/emmcygn/legal-rag-eval.git
cd legal-rag-eval
python -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m scaffolder benchmark
```

Windows PowerShell:

```powershell
git clone https://github.com/emmcygn/legal-rag-eval.git
Set-Location legal-rag-eval
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev]"
.venv\Scripts\python.exe -m scaffolder benchmark
```

In the examples below, Windows users should replace `.venv/bin/python` with
`.venv\Scripts\python.exe`. Plain `python` commands assume the virtual environment
is activated; otherwise use its explicit interpreter path. For Make targets,
set `PYTHON` accordingly.

The command writes `results/evidence-benchmark.json`. No API key, model download, or network access is needed after installation.

LexiChunk is not published on PyPI, so this package pins public SDK commit `397274a289e31eeb420a9c65d1e98c70db978d34` in `pyproject.toml` and `requirements-dashboard.txt`.

## What It Measures

- **Evidence recall:** fraction of independently annotated source characters covered by selected context for answerable queries.
- **Evidence precision:** fraction of selected source characters that overlap annotated evidence for answerable queries.
- **Abstention accuracy:** whether retrieval emitted no context for an unanswerable query. This measures retrieval emission only, not legal-answer correctness.
- **Context budget:** a shared count of non-whitespace token spans matching `\S+`; the report records this convention.

Evidence coverage merges duplicate or overlapping gold spans and gives no credit for the wrong document. Every selected character is charged, including duplicate or irrelevant context.

## Benchmark Scope

The bundled `synthetic-contracts-v1` challenge has three AI-authored contract-like documents and 15 questions. It includes single- and multi-evidence questions, definition/exception combinations, cross-document distractors, topical unanswerable questions, and a zero-overlap unanswerable control.

It is explicitly **not** lawyer validated, held out, customer proof, or evidence of market superiority. Results are useful for testing evaluator behavior and exposing wins, losses, and mixed trade-offs—not for making legal or product-performance claims.

LexiChunk is evaluated in boundary-only mode. Ranking and budget accounting use only the canonical `source[char_start:char_end]`. Generated ancestor headers, definition expansion, and cross-reference expansion are not evaluated in this release.

## Configuration

Copy `scaffolder.yaml.example` and pass it explicitly:

```bash
.venv/bin/python -m scaffolder benchmark --config scaffolder.yaml
.venv/bin/python -m scaffolder benchmark --corpus path/to/dataset.json --output results/custom.json
```

```yaml
strategies: [lexichunk, token_window, rcts]
retrieval_depth: 5
context_budget_tokens: 128
token_window_tokens: 64
token_window_overlap: 8
rcts_chunk_chars: 512
rcts_chunk_overlap: 50
lexichunk_max_tokens: 512
```

Invalid types, booleans used as integers, duplicate IDs, unsupported jurisdictions, bad hashes, missing files, malformed spans, and answer spans not covered by referenced evidence are rejected.

## Dataset Interface

Datasets use schema `anchored_evidence_v1` and declare:

- corpus authorship, legal-validation, held-out, and customer-proof status;
- document jurisdiction, exact SHA-256, provenance, source URL when applicable, license status, and review status;
- source-exact evidence spans with grades;
- answerability, answer spans, and one or more referenced evidence IDs.

Metadata is dataset-declared and is not independently certified by the evaluator. See the [dataset card](docs/dataset-card.md) and [methodology](docs/methodology.md) before authoring or interpreting a dataset.

## Reports and Dashboard

Reports include schema and metric versions, resolved configuration, corpus/document hashes, evaluator and SDK package-tree hashes, per-query selected source spans, all strategy aggregates, unfiltered comparisons, and a canonical payload hash.

The Streamlit dashboard is a viewer for these JSON reports and does not recompute scoring:

```bash
.venv/bin/python -m pip install -e ".[dashboard]"
.venv/bin/python -m streamlit run streamlit_app.py
```

## Legacy Diagnostics

The old SDK-generated structural reference scorer remains available only as an explicitly deprecated diagnostic:

```bash
.venv/bin/python -m scaffolder benchmark-legacy
```

The five SDK-duplicate fixtures and legacy embedding workflow are regression assets, not headline evidence. `evidence_assignment_ndcg_v1` is a versioned custom assignment metric, not classical NDCG. The legacy dashboard retrieval page is no longer routed by the hosted app.

## Development

```bash
make lint
make typecheck
make test
.venv/bin/python -m build
```

See [Contributing](CONTRIBUTING.md), [Security](SECURITY.md), [Methodology](docs/methodology.md), [Dataset Card](docs/dataset-card.md), [Limitations](docs/limitations.md), and [Legacy Diagnostics](EXTENSIBILITY.md).

## License

MIT. See `LICENSE`.
