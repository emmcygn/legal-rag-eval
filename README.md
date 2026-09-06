# Legal RAG Eval

Evaluation harness proving that clause-aware legal chunking ([LexiChunk](https://github.com/emmcygn/lexichunk)) improves RAG retrieval P@5 by **100%** over RecursiveCharacterTextSplitter with statistical significance (p<0.05) 

## Key Results

### Structural Quality (averaged across 5 legal documents)

| Metric | LexiChunk | RCTS | Sentence Split | Fixed-Size |
|--------|:---------:|:----:|:--------------:|:----------:|
| Clause Fragmentation (lower = better) | **0.000** | 0.611 | 0.786 | 0.657 |
| Definition Preservation | **1.000** | 0.988 | 0.988 | 0.976 |
| Cross-Reference Resolution | **1.000** | 1.000 | 1.000 | 0.995 |
| Hierarchy Depth Retained | **1.00** | 0.55 | 0.55 | 0.55 |

LexiChunk achieves **zero clause fragmentation** across all documents. Every legal clause, definition, and cross-reference is preserved intact.

### Retrieval Quality (all-MiniLM-L6-v2, 22 queries)

| Strategy | P@5 | MRR | NDCG@10 |
|----------|:---:|:---:|:-------:|
| **LexiChunk** | **0.236** | **0.295** | 0.200 |
| **LexiChunk Contextual** | 0.218 | 0.277 | **0.248** |
| RCTS | 0.118 `*` | 0.267 | 0.075 |
| Sentence Split | 0.082 `**` | 0.239 | 0.087 |
| Fixed-Size | 0.182 | 0.308 | 0.134 |

`*` p<0.05 `**` p<0.01 (paired t-test vs LexiChunk)

LexiChunk doubles P@5 versus RCTS (0.236 vs 0.118) with statistical significance. The contextual variant achieves the highest NDCG@10 (0.248), showing that prepending clause context headers improves ranking quality.

## Why This Exists

General-purpose text splitters (LangChain's RCTS, sentence splitting, fixed-size windowing) don't understand legal document structure. They split mid-clause, break definition sections, and orphan cross-references. This harness quantifies that damage with 10 metrics across 5 real legal documents and 22 annotated queries.

## Architecture

```
Fixtures --> ChunkingPipeline --> EmbeddingPipeline --> FAISS Index
(5 docs)    (5 strategies)      (3 models)            (top-k search)
                                                           |
Reports <-- Statistical    <-- RetrievalMetrics <-- RetrievalSimulator
(CLI/JSON/   Significance      (P@k, R@k, MRR,      (22 annotated
 HTML/       Testing            NDCG, DRM)            queries)
 Streamlit)
```

### Source Layout

```
src/scaffolder/
  config.py              # BenchmarkConfig -- YAML, env vars, validation
  models.py              # Shared data contracts (Document, Chunk, ChunkSet, etc.)
  queries.py             # YAML query loader with graded relevance
  chunking/              # Strategy wrappers + pipeline (LexiChunk, RCTS, sentence, fixed)
  embedding/             # SentenceTransformer adapter, Voyage adapter, disk cache
  retrieval/             # FAISS VectorIndex, IndexRegistry, RetrievalSimulator
  metrics/               # Structural (5), retrieval (5), statistical significance
  reporting/             # CLI (rich tables), JSON export, HTML (Jinja2 template)
  dashboard/             # Streamlit app (chunk comparison, retrieval demo, metrics)
```

### Strategies Compared

| Strategy | Description | Chunk Boundary |
|----------|-------------|----------------|
| `lexichunk` | Clause-aware legal chunking | Legal clause boundaries |
| `lexichunk_contextual` | LexiChunk + context headers | Clause boundaries + context prefix |
| `rcts` | LangChain RecursiveCharacterTextSplitter (512-char chunks, 50-char overlap) | Character count with overlap |
| `sentence_split` | Sentence boundary splitting (fragments under 100 chars merged forward) | Sentence boundaries |
| `fixed_size` | Fixed 512-char windows, no overlap | Character count |

> **Chunk sizes are not matched across strategies.** LexiChunk emits chunks
> averaging ~790 characters against RCTS's ~360. Several metrics below are
> sensitive to chunk length, so treat cross-strategy comparisons at these
> defaults as confounded by chunk size.

### Embedding Models

| Model | Type | Notes |
|-------|------|-------|
| `all-MiniLM-L6-v2` | Local (free) | Fast baseline, 384 dims |
| `bge-base-en-v1.5` | Local (free) | Higher quality, 768 dims |
| `voyage-law-2` | API (paid) | Legal-specialized, requires `VOYAGE_API_KEY` |

### Metrics

**Structural (5):** Clause fragmentation rate, definition preservation, cross-reference resolution, hierarchy depth retained, chunk size CV

**Retrieval (5):** Precision@k, Recall@k, MRR, NDCG@10, Definition Retrieval Match rate

**Statistical:** Paired t-tests with configurable significance level

See [docs/metrics.md](docs/metrics.md) for formulas, ranges, and interpretation.

## Quick Start

### Install

```bash
git clone https://github.com/emmcygn/legal-rag-eval.git
cd legal-rag-eval
pip install -e ".[dev]"
```

**Note on `lexichunk`.** LexiChunk is not published on PyPI, so it is declared as a
direct reference to its git repository and pip will clone it during install. If you
are developing against a local LexiChunk checkout, install that first and the
requirement is satisfied from your environment:

```bash
pip install -e /path/to/lexichunk
pip install -e ".[dev]"
```

### Run Structural Benchmark (no GPU/embeddings needed)

```bash
make benchmark
```

Outputs rich CLI tables showing all 5 structural metrics across 5 documents x 5 strategies.

### Run Full Retrieval Benchmark

```bash
pip install -e ".[all]"
make benchmark-embed
```

Runs the complete pipeline: chunking, embedding (MiniLM + BGE), FAISS indexing, retrieval simulation across 22 queries, and statistical significance testing. Takes 2-5 minutes.

### Generate HTML Report

```bash
python -m scaffolder benchmark --json   # export results
make report                              # render HTML
```

Opens `results/report.html` with bar charts, heatmaps, and methodology documentation.

### Launch Interactive Dashboard

```bash
pip install -e ".[dashboard]"
make dashboard
```

Three pages at http://localhost:8501:

- **Chunk Comparison** -- Side-by-side LexiChunk vs baseline on any document. Color-coded clause types, term badges, size distribution charts.
- **Retrieval Demo** -- Ask a legal question, compare retrieval results across strategies with P@k charts and relevance highlighting.
- **Metrics Dashboard** -- Headline cards, grouped bar charts, embedding model heatmap.

## Installation Extras

| Extra | Packages | Use Case |
|-------|----------|----------|
| `dev` | ruff, mypy, pytest, pytest-cov | Linting, type checking, testing |
| `embeddings` | sentence-transformers, faiss-cpu | Local embedding models |
| `voyage` | voyageai | Voyage AI legal embeddings |
| `dashboard` | streamlit | Interactive dashboard (plotly is a core dependency) |
| `all` | Everything above | Full installation |

```bash
pip install -e ".[dev]"           # development only
pip install -e ".[embeddings]"    # add local embeddings
pip install -e ".[all]"           # everything
```

## Configuration

```bash
cp scaffolder.yaml.example scaffolder.yaml
```

```yaml
strategies: [lexichunk, rcts, sentence_split, fixed_size]
embedding_models: [all-MiniLM-L6-v2]
k_values: [1, 3, 5, 10]
top_k: 10
significance_level: 0.05
```

Environment variable overrides:

```bash
export SCAFFOLDER_STRATEGIES="lexichunk,rcts"
export SCAFFOLDER_TOP_K=20
export VOYAGE_API_KEY=your-key-here    # enables voyage-law-2
```

See [EXTENSIBILITY.md](EXTENSIBILITY.md) for the full configuration reference and extension guides.

## Query Annotations

22 queries across 5 legal documents (UK, US, EU jurisdictions) with graded relevance:

```yaml
- id: uk_sa_q1
  text: "What are the termination rights under this agreement?"
  document_id: uk_service_agreement
  failure_mode: clause_boundary
  relevant_sections:
    - section_id: "6.1"
      relevance: exact      # 3 points
    - section_id: "6.2"
      relevance: same_section  # 2 points
```

See [queries/schema.md](queries/schema.md) for the annotation format.

## Development

```bash
make lint          # ruff check + format verification
make typecheck     # mypy --strict
make test          # pytest with 80% coverage minimum
make ci            # all of the above
```

| Check | Status |
|-------|--------|
| ruff (lint + format) | 0 errors |
| mypy --strict | 0 errors, 30 files |
| pytest | 270 passed, 86% coverage |
| Benchmark | End-to-end, 5 docs x 5 strategies |

## How This Was Built

This project was built by **two parallel AI agents** (Claude Code) over 2 days with 20 "days" planning files each, running autonomously with daily plans and shared coordination files. Agent A owned the pipeline (chunking, embedding, retrieval, metrics, reporting) while Agent B owned the UI and configuration (Streamlit dashboard, config system, query annotations, CI).

The agents coordinated through:
- **Shared data contracts** (`models.py`) as the interface boundary
- **Append-only handoff logs** for cross-agent deliveries
- **Issue tracking** for cross-agent bugs and blockers

See [guides/agents-orchestration-guide.md](guides/agents-orchestration-guide.md) for the complete methodology -- a reproducible playbook for running parallel AI agents on any software project.

## License

MIT -- see [LICENSE](LICENSE)
