# Legacy Extensibility Guide

This guide describes the deprecated structural and embedding diagnostics. The authoritative public benchmark interface is the anchored-evidence schema and CLI documented in `README.md` and `docs/methodology.md`.

This document describes how to extend the scaffolder with custom fixtures, chunking strategies, embedding models, metrics, queries, and output formats.

## 1. Architecture Overview

The scaffolder follows a plugin architecture:

```
Document → Chunking Strategies → ChunkSets → Embedding → FAISS Index → Retrieval → Metrics
```

Each stage is independently extensible through well-defined interfaces.

### Key Extension Points

| Component | Interface | Registration |
|-----------|----------|-------------|
| Chunking strategies | `ChunkingStrategy` protocol | `chunking/__init__.py` registry |
| Embedding models | `Embedder` protocol | `embedding/__init__.py` factory |
| Metrics | Functions in `metrics/` | Called by pipeline |
| Test fixtures | `.txt` files | `fixtures/documents/` + metadata |
| Queries | YAML files | `queries/*.yaml` |
| Output formats | Functions in `reporting/` | Config `output_formats` |

## 2. Adding Test Fixtures

### Step 1: Create the document file

Place a `.txt` file in `src/scaffolder/fixtures/documents/`:

```
src/scaffolder/fixtures/documents/my_contract.txt
```

### Step 2: Register metadata

Add an entry to `_FIXTURE_METADATA` in `src/scaffolder/fixtures/__init__.py`:

```python
_FIXTURE_METADATA["my_contract.txt"] = (Jurisdiction.US, DocumentType.MSA)
```

If you need new jurisdiction or document type values, add them to the enums in `models.py`.

### Step 3: Add queries

Create `queries/my_contract.yaml` following the schema in `queries/schema.md`.

## 3. Adding Chunking Strategies

### The Strategy Interface

A chunking strategy must implement the `ChunkingStrategy` protocol from `models.py`:

```python
from scaffolder.models import ChunkingStrategy, Document, ChunkSet, Chunk, StrategyName

class MyStrategy:
    name = StrategyName.LEXICHUNK  # Add your own enum value first

    def chunk(self, document: Document) -> ChunkSet:
        chunks = []
        # Your chunking logic here
        return ChunkSet(
            strategy=self.name,
            document_id=document.id,
            chunks=tuple(chunks),
            elapsed_seconds=0.0,
        )
```

### Registration in 5 Steps

1. **Add enum value** in `models.py`:
   ```python
   class StrategyName(str, enum.Enum):
       # ...existing values...
       CHONKIE = "chonkie"
   ```

2. **Create strategy class** in `src/scaffolder/chunking/strategies.py` (or a new file):
   ```python
   class ChonkieStrategy:
       name = StrategyName.CHONKIE

       def chunk(self, document: Document) -> ChunkSet:
           import time
           from chonkie import SemanticChunker

           start = time.perf_counter()
           chunker = SemanticChunker(max_chunk_size=512)
           raw_chunks = chunker.chunk(document.text)

           chunks = tuple(
               Chunk(
                   id=f"{self.name.value}_{document.id}_{i}",
                   text=c.text,
                   document_id=document.id,
                   strategy=self.name,
                   index=i,
                   metadata={"source": "chonkie"},
               )
               for i, c in enumerate(raw_chunks)
           )

           return ChunkSet(
               strategy=self.name,
               document_id=document.id,
               chunks=chunks,
               elapsed_seconds=time.perf_counter() - start,
           )
   ```

3. **Register** in `src/scaffolder/chunking/__init__.py`:
   ```python
   _STRATEGY_REGISTRY[StrategyName.CHONKIE] = ChonkieStrategy
   ```

4. **Add to config** in `config.py`:
   ```python
   VALID_STRATEGIES = [..., "chonkie"]
   ```

5. **Test** your strategy:
   ```python
   def test_chonkie_strategy():
       from scaffolder.chunking import get_strategy
       from scaffolder.fixtures import FixtureManager

       strategy = get_strategy(StrategyName.CHONKIE)
       doc = FixtureManager().load_all()[0]
       result = strategy.chunk(doc)
       assert result.count > 0
       assert all(c.text for c in result.chunks)
   ```

## 4. Adding Embedding Models

### Local Models (sentence-transformers)

The `EmbeddingPipeline` in `embedding/pipeline.py` supports any sentence-transformers model. To add a new one:

1. **Add enum value** in `models.py`:
   ```python
   class EmbeddingModelName(str, enum.Enum):
       # ...existing values...
       NOMIC_EMBED = "nomic-embed-text-v1.5"
   ```

2. **Add HuggingFace model ID** in `embedding/pipeline.py`:
   ```python
   _MODEL_IDS[EmbeddingModelName.NOMIC_EMBED] = "nomic-ai/nomic-embed-text-v1.5"
   ```

3. **Add to config** in `config.py`:
   ```python
   VALID_EMBEDDING_MODELS = [..., "nomic-embed-text-v1.5"]
   ```

4. **Test** that embedding works:
   ```python
   def test_nomic_embedding():
       from scaffolder.embedding.pipeline import EmbeddingPipeline
       from scaffolder.models import EmbeddingModelName

       pipeline = EmbeddingPipeline()
       embeddings = pipeline.embed_texts(["test clause"], EmbeddingModelName.NOMIC_EMBED)
       assert embeddings.shape[0] == 1
       assert embeddings.shape[1] > 0
   ```

### API-Based Models

Follow the `VoyageEmbedder` pattern in `embedding/voyage.py`:

1. Implement the `Embedder` protocol: `model_name` (property), `dimension` (property), `embed_texts(texts)` method
2. Handle rate limiting, retries, and error cases
3. Register in the `EmbeddingPipeline._get_adapter()` factory
4. Gate availability on the API key environment variable (see `embedding/__init__.py`)

## 5. Adding Metrics

### Structural Metrics

Add new metric functions in `src/scaffolder/metrics/structural.py`. Each function takes a `ChunkSet` and `Document` and returns a float:

```python
def citation_preservation_rate(chunk_set: ChunkSet, document: Document) -> float:
    """Measure how well legal citations are preserved within chunks.

    Returns 0.0-1.0 where 1.0 means all citations are kept intact.
    """
    import re

    citation_pattern = re.compile(r"\b\d+\s+U\.S\.C\.\s+§\s*\d+")
    doc_citations = set(citation_pattern.findall(document.text))

    if not doc_citations:
        return 1.0  # No citations to fragment

    preserved = 0
    for citation in doc_citations:
        # Check if the full citation appears in at least one chunk
        if any(citation in c.text for c in chunk_set.chunks):
            preserved += 1

    return preserved / len(doc_citations)
```

Wire it into `compute_structural_metrics()` in the same file, and add a corresponding field to `StructuralMetrics` in `models.py`.

### Retrieval Metrics

Add to `src/scaffolder/metrics/retrieval.py`. Retrieval metrics take `RetrievalResult` objects:

```python
def reciprocal_rank_fusion(results_a: list[RetrievalResult], results_b: list[RetrievalResult], k: int = 60) -> list[float]:
    """Compute RRF scores combining two retrieval result sets."""
    # Implementation here
    ...
```

### Testing Custom Metrics

```python
def test_citation_preservation():
    from scaffolder.metrics.structural import citation_preservation_rate

    # Create test ChunkSet with known citations
    chunks = (Chunk(id="c1", text="Under 42 U.S.C. § 1983...", ...),)
    chunk_set = ChunkSet(strategy=StrategyName.LEXICHUNK, ...)
    doc = Document(text="Under 42 U.S.C. § 1983, the plaintiff...", ...)

    assert citation_preservation_rate(chunk_set, doc) == 1.0
```

## 6. Adding Query Sets

### YAML Schema

Each query file follows this schema (see `queries/schema.md` for full details):

```yaml
document_id: my_contract
queries:
  - id: my_q1
    text: "What are the payment terms?"
    failure_mode: clause_fragmentation
    relevant_sections:
      - section_id: "clause_payment"
        relevance: 3
        description: "Payment terms clause"
    notes: "Tests whether the chunker keeps payment terms together."
```

### Valid Failure Modes

- `clause_fragmentation` — clause split across chunks
- `orphaned_cross_refs` — cross-references lost
- `lost_definitions` — defined terms separated from usage
- `destroyed_hierarchy` — section hierarchy broken
- `cross_doc_contamination` — wrong document's content retrieved

### Adding Queries

1. Create `queries/<document_id>.yaml`
2. Follow the naming convention: `{jurisdiction}_{doc_abbrev}_q{N}` for query IDs
3. Each query must have at least one `relevant_section` with a `relevance` grade (3=exact, 2=partial, 1=background)

## 7. Adding Output Formats

### CLI Reporters

Add functions to `src/scaffolder/reporting/cli.py` that accept `BenchmarkResult` and render with Rich:

```python
def render_my_table(result: BenchmarkResult, console: Console | None = None) -> None:
    # Use Rich Table, Panel, etc.
    pass
```

### JSON Export

The JSON exporter in `reporting/json_export.py` uses `dataclasses.asdict()` — any new dataclass fields are automatically included.

### HTML Reports

Add Jinja2 templates to `src/scaffolder/reporting/templates/` and rendering functions to `reporting/html.py`.

## 8. Configuration Reference

All configuration is managed through `BenchmarkConfig` in `config.py`.

### Precedence (highest to lowest)
1. Environment variables (`SCAFFOLDER_*`)
2. YAML config file (`scaffolder.yaml`)
3. Defaults

### Key Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `strategies` | `list[str]` | All 4 | Chunking strategies to compare |
| `embedding_models` | `list[str]` | `["all-MiniLM-L6-v2"]` | Embedding models |
| `enable_voyage` | `bool` | `False` | Enable Voyage AI |
| `k_values` | `list[int]` | `[1,3,5,10]` | K values for P@k |
| `top_k` | `int` | `10` | Max results per query |
| `significance_level` | `float` | `0.05` | Statistical test threshold |
| `fixed_chunk_size` | `int` | `512` | Fixed-size baseline chunk size |
| `rcts_chunk_size` | `int` | `1000` | RCTS baseline chunk size |

### Environment Variables

All fields can be overridden with `SCAFFOLDER_` prefix:

```bash
export SCAFFOLDER_STRATEGIES="lexichunk,fixed_size"
export SCAFFOLDER_TOP_K=20
export SCAFFOLDER_ENABLE_VOYAGE=true
export VOYAGE_API_KEY=your-key-here
```
