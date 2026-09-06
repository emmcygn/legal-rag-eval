# Legacy Diagnostics and Extension Guide

The structural and embedding pipelines predate the anchored-evidence benchmark and remain available for regression testing and implementation experiments. They are separate from the authoritative benchmark described in the [README](README.md) and [methodology](docs/methodology.md).

The legacy fixtures duplicate SDK output, so their structural labels are not independent evidence. Their retrieval metrics and statistical tests must not be used to support legal-validity, customer-performance, or product-superiority claims.

## Architecture

```text
Documents → chunking strategies → embeddings → retrieval → metrics → reports
```

The main extension points are the shared protocols and enums in `src/scaffolder/models.py`, registries under `src/scaffolder/chunking/` and `src/scaffolder/embedding/`, metric functions in `src/scaffolder/metrics/`, and renderers in `src/scaffolder/reporting/`.

## Commands

Run the structural diagnostic without embeddings:

```bash
python -m scaffolder benchmark-legacy
```

Install the embedding dependencies from the repository root, then run the model-based retrieval diagnostic:

```bash
python -m pip install -e ".[embeddings]"
python -m scaffolder benchmark-embed
```

The model-based command may download `sentence-transformers/all-MiniLM-L6-v2`. Passing `--enable-voyage` adds `voyage-law-2` and requires both the `voyage` optional dependency and `VOYAGE_API_KEY`; install both extras with `python -m pip install -e ".[embeddings,voyage]"`. The adapter layer also retains `BAAI/bge-base-en-v1.5`, although the current legacy CLI does not select it.

Use `--json` to write legacy results under `results/`. These outputs use the older report schema and should remain separate from `evidence_benchmark_report_v1` reports.

## Test Fixtures and Queries

Legacy fixture documents live in `src/scaffolder/fixtures/documents/`. To add one:

1. Add a UTF-8 `.txt` file.
2. Add any required `Jurisdiction` or `DocumentType` value in `src/scaffolder/models.py`.
3. Register the filename in `_FIXTURE_METADATA` in `src/scaffolder/fixtures/__init__.py`.
4. Add a matching `queries/<document_id>.yaml` file.

`document_id` must match the fixture filename without `.txt`. Each query needs a unique `id`, query `text`, one supported `failure_mode`, and at least one `relevant_sections` entry with a section ID and relevance grade. The supported failure modes, relevance scale, naming convention, and complete YAML shape are documented in [`queries/schema.md`](queries/schema.md).

Legacy fixtures and query annotations are regression assets. New datasets for the primary benchmark must instead use the `anchored_evidence_v1` interface and follow the provenance requirements in the [dataset card](docs/dataset-card.md) and [contribution guide](CONTRIBUTING.md).

## Chunking Strategies

A legacy strategy must implement the `ChunkingStrategy` protocol and return a `ChunkSet` for each `Document`. To register one:

1. Add its identifier to `StrategyName` in `src/scaffolder/models.py`.
2. Implement the strategy under `src/scaffolder/chunking/`.
3. Add it to `_STRATEGY_REGISTRY` in `src/scaffolder/chunking/__init__.py`.
4. If configuration-based consumers should accept it, add its string value to `VALID_STRATEGIES` in `src/scaffolder/config.py`.
5. Add focused strategy and pipeline tests.

Keep canonical source text separate from generated context in chunk metadata. Changes to the legacy strategy registry must not alter the primary benchmark's boundary-only source-span accounting.

## Embedding Models

Local embedding adapters implement the `Embedder` protocol in `src/scaffolder/models.py`. To register a local model:

1. Add its enum value to `EmbeddingModelName` in `src/scaffolder/models.py`.
2. Add its exact Hugging Face model ID to `_MODEL_IDS` and embedding dimension to `_MODEL_DIMS` in `src/scaffolder/embedding/pipeline.py`.
3. Route the new enum value to its adapter in `EmbeddingPipeline._get_adapter()`.
4. Add its string value to `VALID_EMBEDDING_MODELS` in `src/scaffolder/config.py` for configuration-based consumers.

The registered local IDs are:

- `sentence-transformers/all-MiniLM-L6-v2`
- `BAAI/bge-base-en-v1.5`

API-backed models should follow `VoyageEmbedder` in `src/scaffolder/embedding/voyage.py` and its wrapper in `src/scaffolder/embedding/pipeline.py`: validate credentials, handle provider errors, expose the model name and dimension, and register the adapter explicitly. Keep API use opt-in; the presence of a credential alone must not change which benchmark runs.

Add model dependencies to an appropriate optional dependency group and test adapter selection, dimensions, batching, and failure handling. Record the exact model ID and relevant provider version in any published legacy result.

## Metrics

Structural metrics operate on `ChunkSet` and `Document` values in `src/scaffolder/metrics/structural.py`. Retrieval metrics operate on `RetrievalResult` values in `src/scaffolder/metrics/retrieval.py`; significance helpers are in `src/scaffolder/metrics/statistical.py`.

When adding a metric, update the corresponding result dataclass in `src/scaffolder/models.py`, wire the calculation into the legacy pipeline, expose it through the applicable reporter, and add tests with explicit expected values. Do not reuse the names of standard information-retrieval metrics for materially different formulas.

## Reports

CLI rendering is implemented in `src/scaffolder/reporting/cli.py`, JSON serialization in `src/scaffolder/reporting/json_export.py`, and HTML rendering in `src/scaffolder/reporting/html.py` with templates under `src/scaffolder/reporting/templates/`.

New output fields should be represented in the shared result dataclasses and covered by serialization and rendering tests. Preserve report schema distinctions between legacy diagnostics and the primary anchored-evidence benchmark.

## Legacy Configuration

`BenchmarkConfig` in `src/scaffolder/config.py` supports legacy library and configuration-based consumers. `BenchmarkConfig.load()` applies values in this order, from highest to lowest precedence:

1. `SCAFFOLDER_*` environment variables
2. an explicit YAML file, or `scaffolder.yaml` when present
3. dataclass defaults

Key defaults are:

| Setting | Default |
| --- | --- |
| Strategies | `lexichunk`, `rcts`, `sentence_split`, `fixed_size` |
| Embedding models | `all-MiniLM-L6-v2` |
| Retrieval cutoffs | `1`, `3`, `5`, `10` |
| Maximum retrieved results | `10` |
| Fixed-size chunks | `512` characters with `50` overlap |
| RCTS chunks | `1000` characters with `200` overlap |
| Output formats | `cli`, `json` |

Supported environment overrides are `SCAFFOLDER_STRATEGIES`, `SCAFFOLDER_EMBEDDING_MODELS`, `SCAFFOLDER_ENABLE_VOYAGE`, `SCAFFOLDER_FIXTURE_DIR`, `SCAFFOLDER_QUERY_DIR`, `SCAFFOLDER_OUTPUT_DIR`, `SCAFFOLDER_TOP_K`, `SCAFFOLDER_K_VALUES`, `SCAFFOLDER_OUTPUT_FORMATS`, `SCAFFOLDER_SIGNIFICANCE_LEVEL`, `SCAFFOLDER_USE_CACHE`, and `SCAFFOLDER_CACHE_DIR`.

The current `benchmark-legacy` and `benchmark-embed` commands do not load `BenchmarkConfig`; they construct their legacy runs directly. Do not describe YAML or environment overrides as affecting those commands unless the CLI wiring is updated and tested.

See [Contributing](CONTRIBUTING.md) for validation requirements. Extensions to legacy components must not introduce a second implementation of the primary anchored-evidence scoring path.
