# Legal RAG Eval

A harness for comparing chunking strategies on legal documents. It runs three evaluations
against three separate ground truths, none of which is derived from a chunker's own output:

| Command | Question | Ground truth | Cost |
|---|---|---|---|
| `make benchmark` | Does retrieval put the answer text inside a fixed context budget? | [`src/scaffolder/data/synthetic_contracts_v1.json`](src/scaffolder/data/synthetic_contracts_v1.json) — 15 questions, 12 char-offset evidence spans, AI-authored | offline, seconds |
| `make benchmark-structural` / `make benchmark-embed` | Do chunk boundaries respect the document's own clause structure, and do the chunks support retrieval? | [`gold/`](gold/) — 365 hand-checked clause spans, 90 defined terms, 171 cross-references, 30 queries | offline seconds / model weights, minutes |
| `make evals` | How does the SDK do on data nobody here wrote? | LEDGAR (LexGLUE) + CUAD (Atticus Project) — see [docs/external_evals.md](docs/external_evals.md) | downloads public datasets |

It is a measuring instrument, not an argument. Everything below is stated so that the
numbers can be checked, including the ways in which they can mislead.

## What this measures, and what it does not

**Ground truth is independent of every chunker under test.** Each of the five fixture
documents has a gold annotation file in [`gold/`](gold/): an ordered, non-overlapping
partition of the numbered body into clauses with character spans, the defined terms with
the span of each definition, and the explicit internal cross-references. The spans were
seeded from each document's own numbering by
[`scripts/build_gold.py`](scripts/build_gold.py) and then corrected by hand against the
text; every correction is logged in [`gold/CHANGES.md`](gold/CHANGES.md). See
[`gold/SCHEMA.md`](gold/SCHEMA.md) for the format. The anchored-evidence dataset is
authored the same way in one respect — the spans are written directly, never read off a
chunker — and differently in another: it is AI-authored and unreviewed, and its
[dataset card](docs/dataset-card.md) says so.

**Every structural metric is computed by character-span overlap in the document.** A
chunk's span is found by matching its text against the document, the same way for every
strategy — never from a chunker's self-reported offsets, and never by searching for one
chunker's output inside another's. A chunk that is not a contiguous span of the source is
reported as unlocated rather than silently scored.

**Relevance is judged by span overlap too.** A retrieved chunk counts as relevant to a
query when its span overlaps the span of a gold clause the query is annotated against by at
least `relevance_min_overlap_chars` characters (default 100).

**This is explicitly not lawyer validated, held out, customer proof, or evidence of market
superiority.** Results are useful for testing evaluator behaviour and exposing wins,
losses, and mixed trade-offs — not for making legal or product-performance claims.

What the harness does **not** measure:

- **Answer quality.** Nothing here runs a generator or grades an answer. Retrieval
  precision is a proxy, and a coarse one.
- **Whether a chunk is *useful* to a reader.** Span overlap says a chunk contains the right
  characters, not that it is self-contained or comprehensible.
- **Generalisation.** Five fixture documents, four of them synthetic (invented parties,
  drafted for this repository); only `eu_gdpr_excerpt.txt` is a real instrument, and it is a
  5 KB abridgement. The anchored-evidence corpus is three more synthetic documents.
  Leave-one-document-out figures are reported precisely because a result that rests on one
  document is not a result. The external LEDGAR/CUAD evaluations exist because of this
  limit, not alongside it.
- **Chunk size, controlled away.** It is not controlled away. Longer chunks straddle more
  gold spans and score better on overlap-based relevance. That is why `rcts_1024` exists as
  a size-matched control and why every table states the mean chunk length that produced it.
  The anchored-evidence benchmark applies a shared 128-token context budget to every
  strategy, which equalises the *selection* budget but is not the same control.
- **LexiChunk's generated content.** In the anchored-evidence benchmark LexiChunk is scored
  boundary-only: ranking and budget use `source[char_start:char_end]` and nothing else.
  Generated ancestor headers, definition expansion and cross-reference expansion get no
  ranking text, no budget and no evidence credit.

## Ground truth

| Document | Source | Characters | Gold clauses (leaf) | Defined terms | Cross-refs | Queries |
|---|---|---:|---:|---:|---:|---:|
| `eu_gdpr_excerpt` | GDPR excerpt (real, abridged) | 5,141 | 21 (14) | 6 | 2 | 6 |
| `uk_service_agreement` | synthetic | 25,187 | 103 (77) | 18 | 31 | 6 |
| `uk_terms_conditions` | synthetic | 16,948 | 70 (55) | 16 | 40 | 6 |
| `us_msa` | synthetic | 25,346 | 57 (47) | 31 | 43 | 6 |
| `us_terms_of_service` | synthetic | 25,743 | 114 (92) | 19 | 55 | 6 |
| **Total** | | **98,365** | **365** (**285**) | **90** | **171** | **30** |

How the annotations were made, in order: `scripts/build_gold.py` reads each document's own
numbering with regexes and emits a first pass; a human then reads that pass against the
document and corrects it, clause span by clause span, adding the defined terms and
cross-references the regexes miss. `gold/CHANGES.md` records every correction per document,
including the categories an annotator checked and left alone, and the calls they were
unsure about. No chunker is consulted at any point, so no chunker is graded against its own
output.

Run `python -m pytest tests/test_gold_annotations.py` to validate every span against the
fixtures. The annotations pin each document by SHA-256, so editing a fixture fails loudly
instead of silently invalidating the spans.

LexiChunk is also evaluated against two public datasets it was not tuned on — LEDGAR for
clause-type classification and CUAD for boundary preservation. Those results are separate
from this harness and live in [docs/external_evals.md](docs/external_evals.md).

## Anchored-evidence benchmark

Deterministic, offline, no model weights. Every strategy ranks candidate source spans with
the same lexical cosine term-frequency ranker and fills the same 128-token context budget;
the score is how much of the independently annotated evidence ends up inside that budget,
and how much of the budget is spent on something else. Unanswerable queries are scored
separately: abstention is correct only when no context was selected at all.

Run against LexiChunk `0.9.0` (`lexichunk_claude@36fcca9`), n = 12 answerable and 3
unanswerable queries. Committed as `results/lexichunk_fixed/evidence_benchmark.json`;
reproduce with `make benchmark`.

| Strategy | Evidence recall ↑ | Evidence precision ↑ | Abstention accuracy ↑ |
|---|---:|---:|---:|
| `lexichunk` | 0.976 | **0.255** | 0.333 |
| `token_window` (64 tok) | **1.000** | 0.134 | 0.333 |
| `rcts` (512 chars) | **1.000** | 0.133 | 0.333 |

LexiChunk trades a little recall for roughly 1.9× the precision: it puts almost as much of
the evidence in the budget while spending far less of the budget on surrounding text. Both
halves of that trade are reported, and the report labels the outcome `mixed` rather than a
win. All three strategies abstain correctly on only one of the three unanswerable queries —
none of them is good at declining to answer, and the lexical ranker returns *something* for
a topical near-miss.

These are plain means with no confidence intervals, on 15 queries over three synthetic
AI-authored documents. Treat the deltas as a direction. The intervals, significance testing
and size-matched control live in the gold-scored benchmark below.

## Results

The tables below are generated from `results/lexichunk_fixed/full_benchmark.json` by
`make readme`. Do not edit them by hand; `make readme-check` fails if they drift.

**How to read them.**

*Retrieval.* With n = 30 queries and 56 tests per embedding model, **no comparison in this
run survives Holm correction** — not one, in either direction. Against the size-matched
control `rcts_1024`, LexiChunk is ahead on MRR under BGE (+0.018) and behind under MiniLM
(−0.045); both intervals comfortably contain zero, and `rcts_1024` has the better R@5 and
NDCG@10 of the two. At this sample size the harness does not separate clause-aware chunking
from a size-matched character splitter, and saying otherwise would mean reading unadjusted
p-values off a family of 112 tests.

*Structural.* On the metrics every strategy can be scored on, the size-matched control is
**not** behind: `rcts_1024` has lower leaf-clause fragmentation (0.008 vs 0.064), lower
top-level over-merge (0.159 vs 0.236) and higher sub-clause grouping (0.629 vs 0.472) than
LexiChunk on these five documents. Recursive character splitting on paragraph separators
lands on clause boundaries here more often than clause-aware parsing does. The columns where
LexiChunk is alone — heading recall/precision, cross-reference recall/precision — read `n/a`
for every other strategy because those strategies emit no headings or cross-references to
score. Those are capability measurements, not comparative wins: a metric only one entrant can
be scored on cannot rank the field, and this README does not use them to.

What the fixed build's numbers *do* support is narrower and checkable: its chunks are exactly
locatable in the source (100% vs the previous build's 54%), and it recovers cross-reference
targets its predecessor did not (0.350 vs 0.139). See
[Baseline vs fixed LexiChunk](#baseline-vs-fixed-lexichunk).

<!-- BEGIN GENERATED RESULTS -->

_Generated 2026-09-06T17:21:12.376208+00:00 — LexiChunk `0.9.0` (commit `36fcca9`); embedding model(s): all-MiniLM-L6-v2, bge-base-en-v1.5; seed `42`; 5 documents; 30 queries. Strategy parameters: `{"fixed_size": {"chunk_size": 512}, "lexichunk": {}, "lexichunk_contextual": {}, "rcts_1024": {"chunk_overlap": 100, "chunk_size": 1024}, "rcts_512": {"chunk_overlap": 50, "chunk_size": 512}, "sentence_split": {"min_chunk_chars": 100}}`._

### Structural quality

Macro-averaged over documents (n = 5). Ground truth is hand-checked span annotations in `gold/`; see the methodology note below for what these metrics do not measure.

| Strategy | Located | Leaf Frag ↓ | Over-merge ↓ | Sub-clause Grp | Head R | Head P | Def Attach ↑ | XRef R | XRef P | Size CV | Avg Chars | Chunks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fixed_size | 100.000% | 0.357 | 0.154 | 0.143 | n/a | n/a | 0.033 | n/a | n/a | 0.149 | 496 | 39 |
| lexichunk | 100.000% | 0.064 | 0.236 | 0.472 | 0.180 | 0.368 | 0.098 | 0.350 | 0.312 | 0.544 | 906 | 26 |
| lexichunk_contextual | 93.419% | 0.106 | 0.236 | 0.485 | 0.173 | 0.379 | 0.101 | 0.327 | 0.313 | 0.452 | 1105 | 26 |
| rcts_1024 | 100.000% | 0.008 | 0.159 | 0.629 | n/a | n/a | 0.088 | n/a | n/a | 0.212 | 826 | 24 |
| rcts_512 | 100.000% | 0.127 | 0.025 | 0.140 | n/a | n/a | 0.041 | n/a | n/a | 0.347 | 357 | 55 |
| sentence_split | 99.363% | 0.275 | 0.000 | 0.016 | n/a | n/a | 0.034 | n/a | n/a | 0.560 | 264 | 80 |

### Retrieval quality

**all-MiniLM-L6-v2** (mean over queries, n = 30)

| Strategy | P@1 | P@3 | P@5 | P@10 | R@1 | R@3 | R@5 | R@10 | MRR | NDCG@10 | DRM rate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fixed_size | 0.500 | 0.267 | 0.213 | 0.140 | 0.467 | 0.650 | 0.750 | 0.872 | 0.640 | 0.663 | 0.343 |
| lexichunk | 0.567 | 0.278 | 0.213 | 0.117 | 0.517 | 0.683 | 0.833 | 0.917 | 0.679 | 0.697 | 0.367 |
| lexichunk_contextual | 0.500 | 0.267 | 0.180 | 0.110 | 0.433 | 0.667 | 0.750 | 0.878 | 0.631 | 0.662 | 0.350 |
| rcts_1024 | 0.533 | 0.300 | 0.227 | 0.127 | 0.483 | 0.744 | 0.906 | 0.956 | 0.724 | 0.731 | 0.397 |
| rcts_512 | 0.567 | 0.289 | 0.193 | 0.113 | 0.461 | 0.678 | 0.744 | 0.839 | 0.686 | 0.665 | 0.310 |
| sentence_split | 0.500 | 0.322 | 0.200 | 0.120 | 0.417 | 0.672 | 0.689 | 0.806 | 0.650 | 0.625 | 0.340 |

**bge-base-en-v1.5** (mean over queries, n = 30)

| Strategy | P@1 | P@3 | P@5 | P@10 | R@1 | R@3 | R@5 | R@10 | MRR | NDCG@10 | DRM rate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fixed_size | 0.600 | 0.322 | 0.240 | 0.137 | 0.528 | 0.778 | 0.839 | 0.872 | 0.734 | 0.726 | 0.347 |
| lexichunk | 0.733 | 0.333 | 0.233 | 0.123 | 0.611 | 0.794 | 0.922 | 0.950 | 0.812 | 0.768 | 0.323 |
| lexichunk_contextual | 0.533 | 0.311 | 0.213 | 0.127 | 0.444 | 0.761 | 0.856 | 0.933 | 0.706 | 0.712 | 0.307 |
| rcts_1024 | 0.633 | 0.344 | 0.240 | 0.123 | 0.544 | 0.839 | 0.956 | 0.972 | 0.794 | 0.790 | 0.357 |
| rcts_512 | 0.667 | 0.333 | 0.227 | 0.127 | 0.561 | 0.767 | 0.850 | 0.867 | 0.786 | 0.734 | 0.317 |
| sentence_split | 0.567 | 0.300 | 0.213 | 0.113 | 0.511 | 0.694 | 0.772 | 0.806 | 0.698 | 0.675 | 0.340 |

### Strategy comparisons

The absolute delta and its bootstrap 95% CI are the headline figures — never a relative percent change. A row is significant only if it survives Holm correction across the whole family of tests below (every metric, pair and embedding model in this run). `rcts_1024` is the **size-matched control**: it uses the same target chunk size as LexiChunk, so it isolates what the chunking *strategy* contributes from what chunk *size* alone would contribute. The LODO range is the delta's spread across the five leave-one-document-out refits; a range straddling zero means one document carries the result.

**all-MiniLM-L6-v2** (56 tests, n = 30 queries each)

| Metric | A | B | n | Δ | 95% CI | p (Holm) | p (Wilcoxon) | Effect size (d) | LODO range |
|---|---|---|---|---|---|---|---|---|---|
| mrr | lexichunk | fixed_size | 30 | +0.040 | [-0.134, +0.211] | 1.000 | 0.647 | 0.082 | [-0.005, +0.067] |
| mrr | lexichunk | **rcts_1024 (size-matched control)** | 30 | -0.045 | [-0.207, +0.115] | 1.000 | 0.588 | -0.098 | [-0.105, +0.013] |
| mrr | lexichunk | rcts_512 | 30 | -0.007 | [-0.157, +0.131] | 1.000 | 0.917 | -0.017 | [-0.040, +0.007] |
| mrr | lexichunk | sentence_split | 30 | +0.029 | [-0.087, +0.152] | 1.000 | 0.649 | 0.087 | [-0.018, +0.078] |
| mrr | lexichunk_contextual | fixed_size | 30 | -0.009 | [-0.189, +0.174] | 1.000 | 1.000 | -0.017 | [-0.042, +0.021] |
| mrr | lexichunk_contextual | **rcts_1024 (size-matched control)** | 30 | -0.094 | [-0.244, +0.052] | 1.000 | 0.244 | -0.223 | [-0.148, -0.031] |
| mrr | lexichunk_contextual | rcts_512 | 30 | -0.056 | [-0.194, +0.087] | 1.000 | 0.421 | -0.140 | [-0.114, +0.007] |
| mrr | lexichunk_contextual | sentence_split | 30 | -0.019 | [-0.164, +0.137] | 1.000 | 0.670 | -0.045 | [-0.090, +0.034] |
| ndcg_at_10 | lexichunk | fixed_size | 30 | +0.034 | [-0.100, +0.166] | 1.000 | 0.601 | 0.091 | [+0.008, +0.049] |
| ndcg_at_10 | lexichunk | **rcts_1024 (size-matched control)** | 30 | -0.034 | [-0.159, +0.089] | 1.000 | 0.629 | -0.096 | [-0.083, +0.020] |
| ndcg_at_10 | lexichunk | rcts_512 | 30 | +0.032 | [-0.085, +0.149] | 1.000 | 0.629 | 0.097 | [-0.000, +0.049] |
| ndcg_at_10 | lexichunk | sentence_split | 30 | +0.072 | [-0.034, +0.186] | 1.000 | 0.327 | 0.230 | [+0.038, +0.102] |
| ndcg_at_10 | lexichunk_contextual | fixed_size | 30 | -0.001 | [-0.136, +0.135] | 1.000 | 0.968 | -0.001 | [-0.021, +0.025] |
| ndcg_at_10 | lexichunk_contextual | **rcts_1024 (size-matched control)** | 30 | -0.068 | [-0.193, +0.048] | 1.000 | 0.306 | -0.200 | [-0.107, -0.016] |
| ndcg_at_10 | lexichunk_contextual | rcts_512 | 30 | -0.002 | [-0.107, +0.107] | 1.000 | 0.968 | -0.008 | [-0.045, +0.043] |
| ndcg_at_10 | lexichunk_contextual | sentence_split | 30 | +0.037 | [-0.069, +0.156] | 1.000 | 0.981 | 0.116 | [-0.008, +0.066] |
| precision_at_1 | lexichunk | fixed_size | 30 | +0.067 | [-0.167, +0.300] | 1.000 | 0.564 | 0.104 | [+0.000, +0.125] |
| precision_at_1 | lexichunk | **rcts_1024 (size-matched control)** | 30 | +0.033 | [-0.167, +0.267] | 1.000 | 0.763 | 0.054 | [-0.042, +0.125] |
| precision_at_1 | lexichunk | rcts_512 | 30 | +0.000 | [-0.200, +0.167] | 1.000 | 1.000 | 0.000 | [-0.042, +0.042] |
| precision_at_1 | lexichunk | sentence_split | 30 | +0.067 | [-0.067, +0.200] | 1.000 | 0.317 | 0.183 | [+0.000, +0.125] |
| precision_at_1 | lexichunk_contextual | fixed_size | 30 | +0.000 | [-0.233, +0.233] | 1.000 | 1.000 | 0.000 | [-0.042, +0.042] |
| precision_at_1 | lexichunk_contextual | **rcts_1024 (size-matched control)** | 30 | -0.033 | [-0.233, +0.167] | 1.000 | 0.739 | -0.060 | [-0.125, +0.042] |
| precision_at_1 | lexichunk_contextual | rcts_512 | 30 | -0.067 | [-0.267, +0.100] | 1.000 | 0.480 | -0.128 | [-0.125, +0.000] |
| precision_at_1 | lexichunk_contextual | sentence_split | 30 | +0.000 | [-0.200, +0.200] | 1.000 | 1.000 | 0.000 | [-0.083, +0.042] |
| precision_at_10 | lexichunk | fixed_size | 30 | -0.023 | [-0.050, +0.003] | 1.000 | 0.131 | -0.321 | [-0.025, -0.021] |
| precision_at_10 | lexichunk | **rcts_1024 (size-matched control)** | 30 | -0.010 | [-0.027, +0.003] | 1.000 | 0.357 | -0.208 | [-0.017, -0.004] |
| precision_at_10 | lexichunk | rcts_512 | 30 | +0.003 | [-0.020, +0.027] | 1.000 | 0.792 | 0.050 | [-0.004, +0.008] |
| precision_at_10 | lexichunk | sentence_split | 30 | -0.003 | [-0.027, +0.020] | 1.000 | 0.931 | -0.050 | [-0.008, +0.004] |
| precision_at_10 | lexichunk_contextual | fixed_size | 30 | -0.030 | [-0.057, -0.003] | 1.000 | 0.055 | -0.400 | [-0.037, -0.025] |
| precision_at_10 | lexichunk_contextual | **rcts_1024 (size-matched control)** | 30 | -0.017 | [-0.033, -0.003] | 1.000 | 0.059 | -0.361 | [-0.021, -0.013] |
| precision_at_10 | lexichunk_contextual | rcts_512 | 30 | -0.003 | [-0.027, +0.020] | 1.000 | 0.792 | -0.050 | [-0.013, +0.004] |
| precision_at_10 | lexichunk_contextual | sentence_split | 30 | -0.010 | [-0.037, +0.017] | 1.000 | 0.493 | -0.140 | [-0.017, -0.004] |
| precision_at_5 | lexichunk | fixed_size | 30 | +0.000 | [-0.053, +0.053] | 1.000 | 1.000 | 0.000 | [-0.008, +0.017] |
| precision_at_5 | lexichunk | **rcts_1024 (size-matched control)** | 30 | -0.013 | [-0.053, +0.027] | 1.000 | 0.527 | -0.114 | [-0.025, +0.000] |
| precision_at_5 | lexichunk | rcts_512 | 30 | +0.020 | [-0.027, +0.067] | 1.000 | 0.405 | 0.151 | [+0.008, +0.033] |
| precision_at_5 | lexichunk | sentence_split | 30 | +0.013 | [-0.040, +0.067] | 1.000 | 0.644 | 0.085 | [+0.008, +0.017] |
| precision_at_5 | lexichunk_contextual | fixed_size | 30 | -0.033 | [-0.093, +0.027] | 1.000 | 0.302 | -0.191 | [-0.050, -0.008] |
| precision_at_5 | lexichunk_contextual | **rcts_1024 (size-matched control)** | 30 | -0.047 | [-0.087, -0.007] | 1.000 | 0.035 | -0.411 | [-0.058, -0.033] |
| precision_at_5 | lexichunk_contextual | rcts_512 | 30 | -0.013 | [-0.060, +0.033] | 1.000 | 0.589 | -0.104 | [-0.033, +0.008] |
| precision_at_5 | lexichunk_contextual | sentence_split | 30 | -0.020 | [-0.080, +0.040] | 1.000 | 0.509 | -0.118 | [-0.033, -0.008] |
| recall_at_10 | lexichunk | fixed_size | 30 | +0.044 | [-0.044, +0.133] | 1.000 | 0.461 | 0.174 | [+0.014, +0.056] |
| recall_at_10 | lexichunk | **rcts_1024 (size-matched control)** | 30 | -0.039 | [-0.128, +0.033] | 1.000 | 0.285 | -0.163 | [-0.069, -0.007] |
| recall_at_10 | lexichunk | rcts_512 | 30 | +0.078 | [-0.056, +0.211] | 1.000 | 0.393 | 0.203 | [+0.014, +0.118] |
| recall_at_10 | lexichunk | sentence_split | 30 | +0.111 | [+0.006, +0.233] | 1.000 | 0.121 | 0.341 | [+0.076, +0.139] |
| recall_at_10 | lexichunk_contextual | fixed_size | 30 | +0.006 | [-0.106, +0.117] | 1.000 | 0.832 | 0.018 | [-0.035, +0.069] |
| recall_at_10 | lexichunk_contextual | **rcts_1024 (size-matched control)** | 30 | -0.078 | [-0.178, +0.000] | 1.000 | 0.102 | -0.302 | [-0.097, -0.056] |
| recall_at_10 | lexichunk_contextual | rcts_512 | 30 | +0.039 | [-0.100, +0.183] | 1.000 | 0.569 | 0.095 | [-0.035, +0.090] |
| recall_at_10 | lexichunk_contextual | sentence_split | 30 | +0.072 | [-0.039, +0.200] | 1.000 | 0.196 | 0.217 | [+0.049, +0.090] |
| recall_at_5 | lexichunk | fixed_size | 30 | +0.083 | [-0.083, +0.250] | 1.000 | 0.374 | 0.183 | [+0.062, +0.104] |
| recall_at_5 | lexichunk | **rcts_1024 (size-matched control)** | 30 | -0.072 | [-0.217, +0.067] | 1.000 | 0.281 | -0.183 | [-0.111, -0.049] |
| recall_at_5 | lexichunk | rcts_512 | 30 | +0.089 | [-0.067, +0.250] | 1.000 | 0.275 | 0.198 | [+0.042, +0.132] |
| recall_at_5 | lexichunk | sentence_split | 30 | +0.144 | [-0.039, +0.328] | 1.000 | 0.198 | 0.278 | [+0.083, +0.181] |
| recall_at_5 | lexichunk_contextual | fixed_size | 30 | +0.000 | [-0.167, +0.167] | 1.000 | 1.000 | 0.000 | [-0.042, +0.062] |
| recall_at_5 | lexichunk_contextual | **rcts_1024 (size-matched control)** | 30 | -0.156 | [-0.311, -0.006] | 1.000 | 0.084 | -0.367 | [-0.194, -0.090] |
| recall_at_5 | lexichunk_contextual | rcts_512 | 30 | +0.006 | [-0.122, +0.139] | 1.000 | 0.864 | 0.015 | [-0.062, +0.069] |
| recall_at_5 | lexichunk_contextual | sentence_split | 30 | +0.061 | [-0.128, +0.250] | 1.000 | 0.409 | 0.116 | [-0.021, +0.139] |

**bge-base-en-v1.5** (56 tests, n = 30 queries each)

| Metric | A | B | n | Δ | 95% CI | p (Holm) | p (Wilcoxon) | Effect size (d) | LODO range |
|---|---|---|---|---|---|---|---|---|---|
| mrr | lexichunk | fixed_size | 30 | +0.079 | [-0.066, +0.220] | 1.000 | 0.343 | 0.192 | [+0.029, +0.109] |
| mrr | lexichunk | **rcts_1024 (size-matched control)** | 30 | +0.018 | [-0.136, +0.164] | 1.000 | 0.840 | 0.042 | [-0.072, +0.081] |
| mrr | lexichunk | rcts_512 | 30 | +0.026 | [-0.084, +0.136] | 1.000 | 0.667 | 0.083 | [-0.009, +0.053] |
| mrr | lexichunk | sentence_split | 30 | +0.114 | [-0.033, +0.262] | 1.000 | 0.152 | 0.272 | [+0.017, +0.149] |
| mrr | lexichunk_contextual | fixed_size | 30 | -0.028 | [-0.159, +0.099] | 1.000 | 0.664 | -0.077 | [-0.047, -0.000] |
| mrr | lexichunk_contextual | **rcts_1024 (size-matched control)** | 30 | -0.089 | [-0.220, +0.034] | 1.000 | 0.205 | -0.243 | [-0.139, -0.049] |
| mrr | lexichunk_contextual | rcts_512 | 30 | -0.081 | [-0.183, +0.022] | 1.000 | 0.169 | -0.276 | [-0.103, -0.038] |
| mrr | lexichunk_contextual | sentence_split | 30 | +0.007 | [-0.089, +0.104] | 1.000 | 0.971 | 0.026 | [-0.012, +0.030] |
| ndcg_at_10 | lexichunk | fixed_size | 30 | +0.042 | [-0.063, +0.139] | 1.000 | 0.617 | 0.142 | [-0.007, +0.074] |
| ndcg_at_10 | lexichunk | **rcts_1024 (size-matched control)** | 30 | -0.022 | [-0.132, +0.081] | 1.000 | 0.678 | -0.074 | [-0.084, +0.037] |
| ndcg_at_10 | lexichunk | rcts_512 | 30 | +0.033 | [-0.055, +0.124] | 1.000 | 0.410 | 0.131 | [+0.008, +0.060] |
| ndcg_at_10 | lexichunk | sentence_split | 30 | +0.093 | [-0.022, +0.213] | 1.000 | 0.147 | 0.282 | [+0.027, +0.131] |
| ndcg_at_10 | lexichunk_contextual | fixed_size | 30 | -0.015 | [-0.110, +0.076] | 1.000 | 0.807 | -0.055 | [-0.024, -0.001] |
| ndcg_at_10 | lexichunk_contextual | **rcts_1024 (size-matched control)** | 30 | -0.079 | [-0.168, +0.007] | 1.000 | 0.088 | -0.321 | [-0.112, -0.050] |
| ndcg_at_10 | lexichunk_contextual | rcts_512 | 30 | -0.023 | [-0.116, +0.070] | 1.000 | 0.536 | -0.087 | [-0.057, +0.007] |
| ndcg_at_10 | lexichunk_contextual | sentence_split | 30 | +0.037 | [-0.046, +0.124] | 1.000 | 0.468 | 0.151 | [+0.020, +0.052] |
| precision_at_1 | lexichunk | fixed_size | 30 | +0.133 | [-0.100, +0.333] | 1.000 | 0.248 | 0.212 | [+0.042, +0.208] |
| precision_at_1 | lexichunk | **rcts_1024 (size-matched control)** | 30 | +0.100 | [-0.133, +0.333] | 1.000 | 0.405 | 0.151 | [-0.042, +0.167] |
| precision_at_1 | lexichunk | rcts_512 | 30 | +0.067 | [-0.100, +0.233] | 1.000 | 0.414 | 0.148 | [+0.000, +0.083] |
| precision_at_1 | lexichunk | sentence_split | 30 | +0.167 | [-0.033, +0.367] | 1.000 | 0.132 | 0.281 | [+0.042, +0.208] |
| precision_at_1 | lexichunk_contextual | fixed_size | 30 | -0.067 | [-0.267, +0.100] | 1.000 | 0.480 | -0.128 | [-0.125, +0.000] |
| precision_at_1 | lexichunk_contextual | **rcts_1024 (size-matched control)** | 30 | -0.100 | [-0.300, +0.100] | 1.000 | 0.317 | -0.183 | [-0.167, +0.000] |
| precision_at_1 | lexichunk_contextual | rcts_512 | 30 | -0.133 | [-0.300, +0.033] | 1.000 | 0.157 | -0.263 | [-0.208, -0.083] |
| precision_at_1 | lexichunk_contextual | sentence_split | 30 | -0.033 | [-0.200, +0.133] | 1.000 | 0.705 | -0.068 | [-0.083, +0.042] |
| precision_at_10 | lexichunk | fixed_size | 30 | -0.013 | [-0.037, +0.010] | 1.000 | 0.344 | -0.196 | [-0.021, -0.004] |
| precision_at_10 | lexichunk | **rcts_1024 (size-matched control)** | 30 | -0.000 | [-0.010, +0.010] | 1.000 | 0.655 | -0.000 | [-0.004, +0.004] |
| precision_at_10 | lexichunk | rcts_512 | 30 | -0.003 | [-0.030, +0.023] | 1.000 | 0.803 | -0.046 | [-0.004, +0.000] |
| precision_at_10 | lexichunk | sentence_split | 30 | +0.010 | [-0.013, +0.033] | 1.000 | 0.553 | 0.151 | [+0.004, +0.017] |
| precision_at_10 | lexichunk_contextual | fixed_size | 30 | -0.010 | [-0.037, +0.017] | 1.000 | 0.765 | -0.132 | [-0.021, +0.000] |
| precision_at_10 | lexichunk_contextual | **rcts_1024 (size-matched control)** | 30 | +0.003 | [-0.013, +0.023] | 1.000 | 0.705 | 0.068 | [-0.004, +0.013] |
| precision_at_10 | lexichunk_contextual | rcts_512 | 30 | +0.000 | [-0.023, +0.023] | 1.000 | 0.832 | 0.000 | [-0.004, +0.008] |
| precision_at_10 | lexichunk_contextual | sentence_split | 30 | +0.013 | [-0.010, +0.040] | 1.000 | 0.305 | 0.183 | [+0.008, +0.017] |
| precision_at_5 | lexichunk | fixed_size | 30 | -0.007 | [-0.047, +0.040] | 1.000 | 0.763 | -0.054 | [-0.017, +0.000] |
| precision_at_5 | lexichunk | **rcts_1024 (size-matched control)** | 30 | -0.007 | [-0.027, +0.013] | 1.000 | 0.564 | -0.104 | [-0.017, +0.008] |
| precision_at_5 | lexichunk | rcts_512 | 30 | +0.007 | [-0.033, +0.053] | 1.000 | 0.783 | 0.054 | [+0.000, +0.017] |
| precision_at_5 | lexichunk | sentence_split | 30 | +0.020 | [-0.020, +0.067] | 1.000 | 0.366 | 0.165 | [+0.008, +0.025] |
| precision_at_5 | lexichunk_contextual | fixed_size | 30 | -0.027 | [-0.067, +0.020] | 1.000 | 0.248 | -0.212 | [-0.042, -0.017] |
| precision_at_5 | lexichunk_contextual | **rcts_1024 (size-matched control)** | 30 | -0.027 | [-0.053, -0.007] | 1.000 | 0.046 | -0.386 | [-0.033, -0.017] |
| precision_at_5 | lexichunk_contextual | rcts_512 | 30 | -0.013 | [-0.053, +0.027] | 1.000 | 0.577 | -0.114 | [-0.017, -0.008] |
| precision_at_5 | lexichunk_contextual | sentence_split | 30 | +0.000 | [-0.040, +0.047] | 1.000 | 1.000 | 0.000 | [+0.000, +0.000] |
| recall_at_10 | lexichunk | fixed_size | 30 | +0.078 | [+0.011, +0.161] | 1.000 | 0.066 | 0.350 | [+0.042, +0.097] |
| recall_at_10 | lexichunk | **rcts_1024 (size-matched control)** | 30 | -0.022 | [-0.100, +0.033] | 1.000 | 0.655 | -0.114 | [-0.042, +0.014] |
| recall_at_10 | lexichunk | rcts_512 | 30 | +0.083 | [+0.000, +0.183] | 1.000 | 0.096 | 0.314 | [+0.062, +0.104] |
| recall_at_10 | lexichunk | sentence_split | 30 | +0.144 | [+0.028, +0.278] | 1.000 | 0.039 | 0.417 | [+0.097, +0.181] |
| recall_at_10 | lexichunk_contextual | fixed_size | 30 | +0.061 | [-0.056, +0.183] | 1.000 | 0.336 | 0.183 | [+0.014, +0.118] |
| recall_at_10 | lexichunk_contextual | **rcts_1024 (size-matched control)** | 30 | -0.039 | [-0.122, +0.022] | 1.000 | 0.285 | -0.183 | [-0.062, +0.014] |
| recall_at_10 | lexichunk_contextual | rcts_512 | 30 | +0.067 | [-0.050, +0.183] | 1.000 | 0.279 | 0.212 | [+0.042, +0.125] |
| recall_at_10 | lexichunk_contextual | sentence_split | 30 | +0.128 | [+0.028, +0.244] | 1.000 | 0.044 | 0.406 | [+0.104, +0.160] |
| recall_at_5 | lexichunk | fixed_size | 30 | +0.083 | [+0.017, +0.167] | 1.000 | 0.059 | 0.361 | [+0.062, +0.104] |
| recall_at_5 | lexichunk | **rcts_1024 (size-matched control)** | 30 | -0.033 | [-0.117, +0.033] | 1.000 | 0.414 | -0.148 | [-0.062, +0.021] |
| recall_at_5 | lexichunk | rcts_512 | 30 | +0.072 | [-0.022, +0.172] | 1.000 | 0.114 | 0.262 | [+0.049, +0.111] |
| recall_at_5 | lexichunk | sentence_split | 30 | +0.150 | [+0.033, +0.283] | 1.000 | 0.037 | 0.400 | [+0.104, +0.188] |
| recall_at_5 | lexichunk_contextual | fixed_size | 30 | +0.017 | [-0.083, +0.117] | 1.000 | 0.785 | 0.060 | [-0.021, +0.062] |
| recall_at_5 | lexichunk_contextual | **rcts_1024 (size-matched control)** | 30 | -0.100 | [-0.200, -0.017] | 1.000 | 0.063 | -0.363 | [-0.125, -0.062] |
| recall_at_5 | lexichunk_contextual | rcts_512 | 30 | +0.006 | [-0.078, +0.100] | 1.000 | 0.891 | 0.022 | [-0.021, +0.028] |
| recall_at_5 | lexichunk_contextual | sentence_split | 30 | +0.083 | [-0.033, +0.217] | 1.000 | 0.163 | 0.238 | [+0.062, +0.104] |

### How to read this

Mean chunk length by strategy — a retrieval comparison is only size-matched when these are close; otherwise an apparent win may just be a chunk-size effect:

- `fixed_size`: 496 chars/chunk
- `lexichunk`: 906 chars/chunk
- `lexichunk_contextual`: 1105 chars/chunk
- `rcts_1024`: 826 chars/chunk
- `rcts_512`: 357 chars/chunk
- `sentence_split`: 264 chars/chunk

<!-- END GENERATED RESULTS -->

## Baseline vs fixed LexiChunk

A chunking benchmark that cannot tell two versions of the same chunker apart is not
measuring the chunker. This is the check, run twice with everything except the LexiChunk
build held fixed — same fixtures, same gold, same 30 queries, same seed (42), same two
embedding models. Both runs are committed: `results/lexichunk_baseline/`
(`lexichunk_baseline@32078cd`, 0.8.0b1) and `results/lexichunk_fixed/`
(`lexichunk_claude@36fcca9`, 0.9.0). Reproduce the table with `make compare-builds`.

| Measure (strategy `lexichunk`) | baseline `32078cd` | fixed `36fcca9` |
|---|---:|---:|
| Chunks located in the source text (n = 5 documents) | 0.541 | 1.000 |
| Leaf-clause fragmentation, lower is better (n = 5) | 0.677 | 0.064 |
| Top-level over-merge, lower is better (n = 5) | 0.079 | 0.236 |
| Heading attachment recall (n = 5) | 0.203 | 0.180 |
| Definition attachment recall (n = 5) | 0.105 | 0.098 |
| Cross-reference target recall (n = 5) | 0.139 | 0.350 |
| Mean chunk length, chars (n = 5) | 802 | 906 |
| P@1, all-MiniLM-L6-v2 (n = 30 queries) | 0.233 | 0.567 |
| R@5, all-MiniLM-L6-v2 (n = 30 queries) | 0.267 | 0.833 |
| MRR, all-MiniLM-L6-v2 (n = 30 queries) | 0.262 | 0.679 |
| P@1, bge-base-en-v1.5 (n = 30 queries) | 0.200 | 0.733 |
| R@5, bge-base-en-v1.5 (n = 30 queries) | 0.328 | 0.922 |
| MRR, bge-base-en-v1.5 (n = 30 queries) | 0.276 | 0.812 |
| _control_ `rcts_1024` MRR, all-MiniLM-L6-v2 | 0.724 | 0.724 |
| _control_ `rcts_1024` MRR, bge-base-en-v1.5 | 0.794 | 0.794 |
| Anchored-evidence recall (n = 12 answerable) | 0.892 | 0.976 |
| Anchored-evidence precision (n = 12 answerable) | 0.164 | 0.255 |

**The harness discriminates between LexiChunk builds.** In the baseline run every one of the
112 statistical comparisons is significant after Holm correction; in the fixed run none is.
The control rows are the reason to believe the difference is the dependency and not the
harness: `rcts_1024` never touches LexiChunk, and its numbers are identical to three decimal
places across the two runs. The anchored-evidence benchmark moves in the same direction on
its own separate ground truth, which is a second, independent confirmation.

That last row only exists because of a change made during reconciliation: the
anchored-evidence benchmark previously called `LegalChunker.sanitize()` unconditionally, a
classmethod the 0.8.0b1 build does not have, so it raised `AttributeError` instead of
producing a number for the older build. It now falls back and warns. A benchmark that cannot
run against the build you want to compare against reports nothing at all.

**What is actually being measured here.** Most of the gap is one thing: the baseline build
emitted chunk text that is not a contiguous substring of the source, because it prepended a
normalised heading breadcrumb to continuation chunks. Only 54% of its chunks could be located
in the document at all, and a chunk with no span can never overlap a gold clause's span, so
it can never be judged relevant. That is a real defect and this harness is the right
instrument for catching it — but it is a *localisation* failure, not evidence that the fixed
build retrieves better in a way a user would feel. The comparison that speaks to user-visible
retrieval quality is the one against the size-matched control above, and that one is null.

`localization_rate` is the metric to watch when comparing chunker versions. It is the only
one in the structural table that moved by more than a rounding error between these two
builds, it has an unambiguous correct value (1.000), and every span-based metric downstream —
structural and retrieval alike — is silently capped by it.

## External evaluations

Five synthetic-ish documents cannot support a claim about legal documents in general, so
LexiChunk is also measured against two public datasets with independent human labels that
neither this harness nor the SDK was tuned on. Full methodology, per-class tables and the
issues found are in [docs/external_evals.md](docs/external_evals.md); the runs are committed
under `results/external/`.

**LEDGAR** (LexGLUE, 10,000-row test split, 5,000 sampled, 3,955 in scope after mapping
100 LEDGAR labels onto 31 LexiChunk clause types) — clause-type classification, bootstrap
CIs over 2,000 resamples:

| System | Accuracy | Macro-F1 |
|---|---|---|
| `lexichunk` keyword classifier | 39.6% [38.0, 41.2] | 42.5% [40.7, 44.1] |
| majority-class floor | 21.6% [20.3, 22.8] | 1.9% [1.8, 2.0] |
| TF-IDF + logistic regression (supervised) | **90.8%** [89.9, 91.7] | **89.9%** [88.5, 91.0] |

The keyword classifier is well above the majority floor and far below a supervised model
trained on the same data — which is what a rule-based classifier should look like, stated
plainly rather than compared only against the floor. Its confidence scores are poorly
calibrated (ECE 0.113, Spearman 0.375, not monotonic).

**CUAD** (Atticus Project, 100 real SEC contracts sampled from 510, 2,458 lawyer-annotated
spans verified against their own source text) — does at least one chunk wholly contain a
gold answer span, at matched mean chunk length:

| Strategy | Mean chars | Containment | Lift vs length-matched grid |
|---|---:|---|---:|
| `lexichunk-128tok` | 346 | **83.7%** [82.2, 85.2] | **+36.7 pp** |
| `rcts-512` | 382 | 72.1% [70.2, 74.0] | +22.2 pp |

Raw containment for the larger settings (`lexichunk-512tok` 98.1%, `sentence-window-3`
98.9%) is not quoted as a win: containment rises mechanically with chunk length and with
overlap, and only 54.7% of the sentence-window matches are single-chunk. The length-matched
comparison is the one that carries information.

The same run also found that LexiChunk's `us` profile finds five or more top-level clauses
in only **14.0%** of these 100 US contracts, falling back to flat text in 48% — while the
`uk` profile on the *same* US contracts manages 31.0%. That is a finding against the SDK,
not for it, and it is filed with a reproducer in `results/external/lexichunk_issues.md`
along with five others. No exceptions were raised across 150 contracts.

## Architecture

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

### Source layout

```
src/scaffolder/
  config.py              # BenchmarkConfig -- YAML + SCAFFOLDER_* env vars, wired into the CLI
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
scripts/build_gold.py    # Seeds gold annotations from document numbering
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

The parameters above are the defaults in `scaffolder.yaml.example`; the values that
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

See [docs/metrics.md](docs/metrics.md) for each metric's formula, range, and — for every
one of them — what it does not measure.

## Quick start

Python 3.10-3.12.

### Install

```bash
git clone https://github.com/emmcygn/legal-rag-eval.git
cd legal-rag-eval
pip install -e ".[dev]"
```

Windows PowerShell:

```powershell
git clone https://github.com/emmcygn/legal-rag-eval.git
Set-Location legal-rag-eval
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

**LexiChunk is not published on PyPI.** `pyproject.toml` declares it as a direct git
reference pinned to an exact commit
(`lexichunk @ git+https://github.com/emmcygn/lexichunk.git@397274a...`), so pip clones it
during install and a clean clone resolves to one specific build. Pinning it by version
specifier resolved to a PyPI 404, which is why a clean install — and therefore CI — could
not previously succeed.

Every result file also records `lexichunk_version` and `lexichunk_commit`, the committed
runs live in directories named after the build, and `make compare-builds` diffs two of them.
Set `SCAFFOLDER_LEXICHUNK_COMMIT` when running against a local checkout, which records no
VCS metadata of its own.

To develop against a local LexiChunk checkout, install the extras **first** and the editable
checkout **last**:

```bash
pip install -e ".[dev]"
pip install -e /path/to/lexichunk        # must come last
python -c "import lexichunk; print(lexichunk.__file__)"   # verify
```

The order matters. Installing the extras after the editable checkout silently replaces it
with the pinned wheel, and the run then measures the pinned build while you believe it is
measuring your working tree. The verification line above is not optional advice; this trap
has caught more than one run in this repository's history.

### Anchored-evidence benchmark (offline, no model downloads, seconds)

```bash
make benchmark        # writes results/evidence-benchmark.json
```

### Structural benchmark against the gold annotations (offline, seconds)

```bash
make benchmark-structural
```

### Full retrieval benchmark (downloads embedding weights, minutes)

```bash
pip install -e ".[embeddings]"
make benchmark-embed
```

Output lands in `results/full_benchmark.json` and is not committed. The two published runs
are written to build-named directories instead, so a local run never overwrites the evidence
behind the tables above:

```bash
SCAFFOLDER_LEXICHUNK_COMMIT=$(git -C /path/to/lexichunk rev-parse --short HEAD)   python -m scaffolder benchmark-embed --json --seed 42   --models all-MiniLM-L6-v2,bge-base-en-v1.5   --output-dir results/lexichunk_fixed
```

Select a different embedding model or strategy set without editing source:

```bash
python -m scaffolder benchmark-embed --models bge-base-en-v1.5 --json
python -m scaffolder benchmark-embed --strategies lexichunk,rcts_1024 --top-k 20 --seed 7
```

`voyage-law-2` needs **both** `--enable-voyage` and `VOYAGE_API_KEY`. An exported key on its
own never causes a paid API call.

### External evaluations (downloads public datasets)

```bash
pip install -e ".[evals]"
make evals           # LEDGAR + CUAD, writes results/external/
make evals-smoke     # offline synthetic path; the numbers are meaningless by design
```

### Superseded circular diagnostics

```bash
python -m scaffolder benchmark-legacy --json
```

Scores strategies against LexiChunk's own parse, so LexiChunk's fragmentation is 0 by
construction. Retained for provenance, warns when it runs, and is never evidence.

### Regenerate the README results section

```bash
make readme          # rewrite the generated block from results/lexichunk_fixed/
make readme-check    # exit 1 if the block is stale
```

### Compare two LexiChunk builds

```bash
make compare-builds
```

### HTML report

```bash
make benchmark-structural
make report
```

### Dashboard

```bash
pip install -e ".[dashboard]"
make dashboard
```

The dashboard is a viewer for committed report JSON. It does not recompute scoring.

## Installation extras

| Extra | Packages | Use case |
|---|---|---|
| `dev` | ruff, mypy, pytest, pytest-cov | Linting, type checking, testing |
| `embeddings` | sentence-transformers | Local embedding models (torch, weight downloads) |
| `voyage` | voyageai | Voyage AI legal embeddings |
| `dashboard` | streamlit | Interactive dashboard (plotly is a core dependency) |
| `all` | Everything above | Full installation |

`faiss-cpu` is a core dependency, not an extra: `scaffolder.retrieval.index` imports it at
module scope and the query-annotation checks import that, so `pip install -e ".[dev]" &&
make test` — the documented developer setup, and what CI runs — could not otherwise collect
three test modules. The five tests that load a real sentence-transformer model skip when the
`embeddings` extra is absent.

## Configuration

```bash
cp scaffolder.yaml.example scaffolder.yaml
```

The CLI reads this file (or `--config PATH`), applies `SCAFFOLDER_*` environment overrides
on top, then CLI flags, and writes the resolved configuration into the results JSON. Every
documented key takes effect.

```bash
export SCAFFOLDER_STRATEGIES="lexichunk,rcts_1024"
export SCAFFOLDER_TOP_K=20
export SCAFFOLDER_RELEVANCE_MIN_OVERLAP_CHARS=150
export VOYAGE_API_KEY=your-key-here    # enables voyage-law-2
```

See [EXTENSIBILITY.md](EXTENSIBILITY.md) for extension guides.

## Query annotations

**30 queries, 6 per document**, in [`queries/`](queries/) — one YAML file per document.
Relevant passages are given by **gold clause identifier**, not by prose, and the loader
resolves each identifier to that clause's subtree span:

```yaml
document_id: uk_service_agreement
queries:
  - id: uk_sa_q4
    text: "If the Client disputes part of an invoice, do they still have to pay the rest of
      it while the dispute is being sorted out?"
    category: conditional
    relevant_clauses:
      - identifier: "3.6(a)"
        relevance: 3
        description: "Undisputed portions remain payable while a dispute is resolved."
      - identifier: "3.6(b)"
        relevance: 2
        description: "How a dispute must be raised, and by when."
    notes: >
      The query says "disputes part of an invoice" rather than the clause's own
      "disputed in good faith", so a lexical match does not win by default.
```

`queries/schema.md` documents the schema and the six `category` values. Those categories
name the *kind of question* (`definition_lookup`, `clause_lookup`, `numeric_lookup`,
`conditional`, `multi_clause`, `cross_reference`), deliberately replacing the earlier
labelling by LexiChunk failure mode: those five labels mapped almost one-to-one onto the
structural metrics used to argue for LexiChunk, which meant the query set was organised
around the conclusion.

`tests/test_query_annotations.py` enforces, in CI, that every identifier exists in the
document's gold file, that every query has at least one grade-3 clause, that no query's
relevant spans cover more than 25% of its document or more than 30% of its gold clauses,
that the query text does not name its own answer's identifier, and that every document has
at least six queries. The previous set failed most of these: five of its twenty-two queries
named clauses that were not in the document at all — a guaranteed zero for every strategy,
in every paired difference — and two marked a majority of the document relevant.

## Development

```bash
make lint            # ruff check + format verification
make typecheck       # mypy --strict
make test            # pytest with 80% coverage minimum
make ci              # all of the above
make readme-check    # fail if the generated results block is stale
```

`.github/workflows/ci.yml` runs lint, mypy, the test suite on Python 3.10/3.11/3.12, the
gold and query annotation validators, and a structural-benchmark smoke run on every push.
The retrieval benchmark downloads model weights, so it is `workflow_dispatch` only.

## Provenance

This repository was built by AI agents, independently audited — the audit found the original
methodology circular and the original headline claims unsupported — and then rebuilt twice in
parallel by two different tools. What is here now reconciles both rebuilds:

- The **anchored-evidence benchmark** (`src/scaffolder/evidence/`), its dataset interface,
  the report hashing, the wheel/offline CI smoke tests and the honesty framing come from the
  overhaul merged as PRs #1 and #2.
- The **gold annotations** (`gold/`), the span-overlap metrics, the statistics module, the
  size-matched `rcts_1024` control, the 30-query gold-anchored set and the generated results
  tables come from the parallel audit-fix branch.
- The **external LEDGAR and CUAD evaluations** (`src/scaffolder/evals/`) come from a third
  branch and are the only ground truth in this repository that neither rebuild authored.

Where the two rebuilds disagreed on the same point, the reconciliation kept the one with
evidence behind it and recorded the choice. The circular original methodology is reachable
only as `benchmark-legacy`, which warns when it runs.

## License

MIT — see [LICENSE](LICENSE)
