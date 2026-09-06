# Baseline vs fixed LexiChunk

A chunking benchmark that cannot tell two versions of the same chunker apart is not
measuring the chunker. This document is that check and, more importantly, what it does
and does not support. The headline table is repeated in the
[README](../README.md#baseline-vs-fixed-lexichunk).

A chunking benchmark that cannot tell two versions of the same chunker apart is not
measuring the chunker. This is the check, run twice with everything except the LexiChunk
build held fixed — same fixtures, same gold, same 30 queries, same seed (0), same two
embedding models. Both runs are committed: `results/lexichunk_baseline/`
(`dev@32078cd`, 0.8.0b1) and `results/lexichunk_fixed/`
(`release/0.9.0@0346a12`, 0.9.0). Reproduce the table with `make compare-builds`.

| Measure (strategy `lexichunk`) | baseline — LexiChunk `0.8.0b1` `32078cd` | fixed — LexiChunk `0.9.0` `0346a12` |
|---|---:|---:|
| Chunks located in the source text (n = 5 documents) | 0.686 | 1.000 |
| Leaf-clause fragmentation (lower is better) (n = 5 documents) | 0.422 | 0.030 |
| Top-level over-merge (lower is better) (n = 5 documents) | 0.040 | 0.020 |
| Heading attachment recall (n = 5 documents) | 0.441 | 0.453 |
| Definition attachment recall (n = 5 documents) | 0.092 | 0.082 |
| Cross-reference target recall (n = 5 documents) | 0.139 | 0.774 |
| Mean chunk length (chars) (n = 5 documents) | 514 | 552 |
| P@1, all-MiniLM-L6-v2 (n = 30 queries) | 0.333 | 0.533 |
| P@5, all-MiniLM-L6-v2 (n = 30 queries) | 0.107 | 0.200 |
| R@5, all-MiniLM-L6-v2 (n = 30 queries) | 0.467 | 0.789 |
| MRR, all-MiniLM-L6-v2 (n = 30 queries) | 0.408 | 0.666 |
| NDCG@10, all-MiniLM-L6-v2 (n = 30 queries) | 0.421 | 0.656 |
| P@1, bge-base-en-v1.5 (n = 30 queries) | 0.467 | 0.567 |
| P@5, bge-base-en-v1.5 (n = 30 queries) | 0.120 | 0.213 |
| R@5, bge-base-en-v1.5 (n = 30 queries) | 0.517 | 0.822 |
| MRR, bge-base-en-v1.5 (n = 30 queries) | 0.542 | 0.737 |
| NDCG@10, bge-base-en-v1.5 (n = 30 queries) | 0.525 | 0.728 |
| _control_: `rcts_1024` MRR, all-MiniLM-L6-v2 | 0.724 | 0.724 |
| _control_: `rcts_1024` MRR, bge-base-en-v1.5 | 0.794 | 0.794 |
| Anchored-evidence recall (n = 12 answerable) | 0.892 | 0.976 |
| Anchored-evidence precision (n = 12 answerable) | 0.164 | 0.255 |

**The harness discriminates between LexiChunk builds, but not uniformly.** In the baseline
run 61 of the 112 statistical comparisons are significant after Holm correction; in the fixed
run none is. The control rows are the reason to believe part of the remaining difference is
the dependency and not the harness: `rcts_1024` never touches LexiChunk, and its numbers are
identical to three decimal places across the two runs. The anchored-evidence benchmark moves
in the same direction on its own separate ground truth, which is a second, independent
confirmation — but see the retrieval caveat above: against the size-matched `rcts_1024`/
`rcts_512` controls, the *fixed* build's own MRR delta is negative under both embedding
models and every interval contains zero (Holm p = 1.000 throughout), so this table shows the
fixed build is a large, mostly-significant improvement over the baseline build, not that
LexiChunk beats a size-matched splitter.

That last row only exists because of a change made during reconciliation: the
anchored-evidence benchmark previously called `LegalChunker.sanitize()` unconditionally, a
classmethod the 0.8.0b1 build does not have, so it raised `AttributeError` instead of
producing a number for the older build. It now falls back and warns. A benchmark that cannot
run against the build you want to compare against reports nothing at all.

**What is actually being measured here.** Most of the gap is one thing: the baseline build
emitted chunk text that is not a contiguous substring of the source, because it prepended a
normalised heading breadcrumb to continuation chunks. Only 69% of its chunks could be located
in the document at all, and a chunk with no span can never overlap a gold clause's span, so
it can never be judged relevant. That is a real defect and this harness is the right
instrument for catching it — but it is a *localisation* failure, not evidence that the fixed
build retrieves better in a way a user would feel. The comparison that speaks to user-visible
retrieval quality is the one against the size-matched control above, and that one is null.

The cross-reference target recall row (0.139 -> 0.774) additionally mixes two different
things: a chunker version change *and* a harness scoring fix landed in this same branch (the
gold identifier normaliser now folds a roman-numeral article/section number to its arabic
form, and the strategy adapter now emits a cross-reference's label together with its
identifier — see the commit fixing `legal_rag_eval.metrics.gold.normalize_identifier` and
`legal_rag_eval.chunking.strategies._xref_target_string`). Some of this row's movement is a real
LexiChunk improvement and some is the harness no longer mis-scoring a correct answer as
wrong; this run does not separate the two.

`localization_rate` is the metric to watch when comparing chunker versions. It is the only
one in the structural table that moved by more than a rounding error between these two
builds, it has an unambiguous correct value (1.000), and every span-based metric downstream —
structural and retrieval alike — is silently capped by it.
