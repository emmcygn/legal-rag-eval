# Methodology

This document covers the **anchored-evidence benchmark** (`python -m scaffolder benchmark`).
The gold-scored structural and retrieval benchmarks are a separate evaluation against a
separate ground truth; their methodology is in [metrics.md](metrics.md) sections 1-5 and
[`gold/SCHEMA.md`](../gold/SCHEMA.md). The external LEDGAR and CUAD evaluations are in
[external_evals.md](external_evals.md).

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
