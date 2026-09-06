# Methodology

Sections 1-6 cover the **anchored-evidence benchmark**
(`legal-rag-eval benchmark`). The gold-scored structural and retrieval benchmarks are a
separate evaluation against a separate ground truth; their metric definitions are in
[metrics.md](metrics.md) sections 1-5, their ground truth in
[ground-truth.md](ground-truth.md) and [`gold/SCHEMA.md`](../gold/SCHEMA.md). The external
LEDGAR and CUAD evaluations are in [external_evals.md](external_evals.md).

The last section describes the harness as a whole: its pipeline, its source layout, the
strategies it compares and every metric it reports.

## Evaluation Unit

Each query is evaluated against exact character intervals in canonical UTF-8 source documents. Document bytes are protected by SHA-256 values in the dataset manifest. A label is invalid if its stored text is not exactly `source[start:end]`.

Answer spans and evidence spans are authored independently of chunker output. Every answer span must be covered by the evidence IDs referenced by that query, including the correct document. Duplicate IDs, missing spans, invalid grades, unsupported jurisdictions, and contradictory corpus/document review declarations fail dataset loading.

## Candidate Construction

- `lexichunk`: calls the public SDK with the document jurisdiction and reads only `char_start` and `char_end`. Candidate text is always the canonical source slice.
- `token_window`: creates fixed windows over non-whitespace token spans with configurable overlap.
- `rcts`: uses RecursiveCharacterTextSplitter with source start indexes and verifies every returned slice.

LexiChunk generated headers, definition expansion, and cross-reference expansion are outside the first-release scope. They receive no ranking text, budget, or evidence credit.

## Shared Retrieval

All strategies use deterministic cosine similarity over case-folded Unicode word term frequencies. This is a lexical baseline, not semantic or model-based retrieval. Scores are sorted deterministically, with document and source offsets breaking ties.

Up to `retrieval_depth` candidates are ranked. Selected context is clipped to a shared budget measured by non-whitespace spans matching `\S+`. The JSON report records this exact budget unit and every selected source interval.

## Scoring

For answerable queries, gold intervals are unioned per document. Evidence recall is covered gold characters divided by total gold characters. Evidence precision is covered gold characters divided by all selected characters. Overlapping selected spans are each charged, while overlapping gold is counted once.

For unanswerable queries, precision and recall are omitted. Abstention is correct only when no context was selected. This is a retrieval-emission decision and does not measure the correctness of a generated legal answer.

Evidence grade is preserved as annotation metadata in schema v1 but is not used as a weight in source-character coverage.

## Interpretation

Reports include per-query results and unfiltered win, loss, tie, or mixed aggregate comparisons. A larger chunk cannot win merely by containing evidence because all extra selected source characters reduce precision and the same token budget applies to every strategy.

That equalises the **selection** budget. It is not a control for chunk length itself: nothing here matches a baseline's chunk size to LexiChunk's, so a strategy whose chunks happen to sit closer to the evidence granularity is advantaged before selection begins. The gold-scored benchmark carries an explicit size-matched control (`rcts_1024`) for exactly this reason; read the two together rather than either alone.

There are no confidence intervals here. Aggregates are plain means over 12 answerable and 3 unanswerable queries, and comparisons are unadjusted deltas with a win/loss/tie/mixed label. At that sample size a delta is a direction, not a measurement. The gold-scored retrieval benchmark is where bootstrap intervals, Wilcoxon tests, Holm correction and leave-one-document-out figures live.

The bundled corpus is suitable for evaluator regression and demonstration only. Any product or scientific claim requires a separately authored, documented, reviewed, representative dataset.

## Architecture and metric inventory

The sections below describe the harness as a whole: how a document becomes a scored
chunk set, which strategies are compared and why, and what every reported metric is.
Each metric's formula, range and blind spots are in [metrics.md](metrics.md).

### Pipeline

```
Fixtures ---> ChunkingPipeline ---> EmbeddingPipeline ---> FAISS index
(5 docs)      (6 strategies,        (MiniLM / BGE /        (exact inner product,
               spans located)        Voyage)                one index per strategy+model)
   |                |                                            |
gold/          GoldStructural                            RetrievalSimulator
(spans)          Metrics  <------- span overlap -------> (annotated queries)
                    |                                            |
                    +----------> Reports (CLI / JSON / HTML / README) <---- Comparisons
                                                                            (bootstrap CI,
                                                                             Holm, Wilcoxon,
                                                                             LODO)
```

#### Source layout

```
src/legal_rag_eval/
  config.py              # BenchmarkConfig -- YAML + LEGAL_RAG_EVAL_* env vars, wired into the CLI
  models.py              # Shared data contracts
  gold.py                # Gold annotation loader, sanitisation, chunk span localisation
  queries.py             # Query loader; resolves clause identifiers to gold spans
  chunking/              # Strategy wrappers + pipeline (spans attached here)
  embedding/             # SentenceTransformer + Voyage adapters, disk cache
  retrieval/             # FAISS VectorIndex, IndexRegistry, RetrievalSimulator
  metrics/
    gold.py              # Reported structural metrics (span overlap vs gold/)
    structural.py        # Superseded, LexiChunk-derived metrics (legacy_*, not reported)
    retrieval.py         # P@k, R@k, MRR, NDCG -- one shared relevance definition
    statistical.py       # Bootstrap CIs, Holm, Wilcoxon, effect sizes, leave-one-doc-out
  reporting/             # CLI tables, JSON export, HTML, generated README section
  dashboard/             # Streamlit app
gold/                    # Hand-checked ground truth, one JSON per document
queries/                 # Annotated queries, relevant clauses given by gold identifier
tools/build_gold.py    # Seeds gold annotations from document numbering
```

### Strategies compared

| Strategy | What it does | Chunk boundary |
|---|---|---|
| `lexichunk` | Clause-aware legal chunking (LexiChunk) | Legal clause boundaries |
| `lexichunk_contextual` | LexiChunk plus a context header prepended before embedding | Clause boundaries + context prefix |
| `rcts_512` | LangChain `RecursiveCharacterTextSplitter`, 512 / 50 | Character count with overlap |
| `rcts_1024` | The same splitter at 1024 / 100 — **the size-matched control** | Character count with overlap |
| `sentence_split` | Sentence boundaries, fragments under 100 chars merged forward | Sentence boundaries |
| `fixed_size` | 512-character windows, **no overlap** | Character count |

The parameters above are the defaults in `legal-rag-eval.yaml.example`; the values that
actually ran are recorded in the results JSON and printed in the generated tables. RCTS
appears twice on purpose: LexiChunk emits chunks roughly twice the length of `rcts_512`, and
a comparison against an unmatched baseline measures chunk size as much as anything else.

`lexichunk_contextual` injects section labels into the chunk text before embedding. That
text is not in the source document, so it is excluded from span localisation and gives no
advantage to the structural metrics — but it does change what the embedder sees, which is
the point of the variant and a confound for the retrieval comparison.

### Embedding models

| Model | Type | Notes |
|---|---|---|
| `all-MiniLM-L6-v2` | Local (free) | 384 dims, the default |
| `bge-base-en-v1.5` | Local (free) | 768 dims, larger download; select with `--models` |
| `voyage-law-2` | API (paid) | Legal-specialised, needs `VOYAGE_API_KEY` |

### Metrics

Anchored-evidence metrics: source-span evidence recall and precision (answerable queries),
abstention accuracy (unanswerable queries). No grades, no confidence intervals — plain means
over 15 queries.

Reported structural metrics: clause fragmentation, top-level over-merge, sub-clause
grouping (informational), heading recall/precision, definition attachment recall,
cross-reference target precision/recall, chunk-size CV, and the localisation rate that says
how much of a strategy's output could be scored at all.

Retrieval metrics: Precision@k, Recall@k, MRR, NDCG@10, document-retrieval-mismatch rate.
NDCG here is `evidence_assignment_ndcg_v1`: each rank claims at most one distinct gold
section under a maximum-weight one-to-one assignment, so it is bounded by construction and
invariant to gold-list order and to `PYTHONHASHSEED`. It is deliberately not classical
per-rank chunk NDCG, every exported row records which variant produced it, and rows labelled
`legacy_unversioned` came from the old formula and are not comparable.

Statistics: paired bootstrap 95% confidence intervals on every delta, Holm-corrected
p-values across the whole family of tests, Wilcoxon signed-rank alongside the t-test,
Cohen's *d* and rank-biserial effect sizes, and leave-one-document-out sensitivity. `n` is
stated on every table.

See [docs/metrics.md](metrics.md) for each metric's formula, range, and — for every
one of them — what it does not measure.
