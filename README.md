# legal-rag-eval

**What this measures:** whether a chunking strategy puts the right text in front of a
retriever, on legal documents, judged against ground truth that no chunker produced. It
compares LexiChunk against character, sentence and size-matched splitters on three
evaluations with three independent ground truths, and reports the losses as plainly as the
wins.

It is a measuring instrument, not an argument. Every number below can be traced to a
committed run in [`results/`](results/), and the tables that carry the headline figures are
generated from those runs rather than typed by hand.

## The three benchmarks

| Benchmark | Ground truth | What it can conclude | What it cannot | Command | Runtime |
|---|---|---|---|---|---|
| **Anchored evidence** | [`synthetic_contracts_v1.json`](src/legal_rag_eval/data/synthetic_contracts_v1.json) — 15 questions, 12 char-offset evidence spans, AI-authored and unreviewed ([dataset card](docs/dataset-card.md)) | Whether a strategy fits the annotated answer text into a fixed context budget, and how much of that budget it wastes | Nothing about semantic retrieval, answer quality, or significance: 15 queries, a lexical ranker, plain means with no intervals | `make benchmark` | offline, seconds |
| **Gold structural + retrieval** | [`gold/`](gold/) — 365 hand-checked clause spans, 90 defined terms, 171 cross-references, 30 anchored queries ([how they were made](docs/ground-truth.md)) | Whether chunk boundaries land on the document's own clause structure, with bootstrap intervals, Holm correction and a size-matched control | That clause-aware chunking beats a size-matched splitter — at n = 30 nothing in the published run survives Holm correction, in either direction | `make benchmark-structural`, `make benchmark-embed` | offline seconds / model weights, minutes |
| **External** | LEDGAR (LexGLUE) + CUAD (Atticus Project) — labels from third parties, on real filings ([methodology](docs/external_evals.md)) | How LexiChunk does on data nobody here wrote or tuned against, including where it fails | Generalisation to a customer's corpus; neither dataset was designed to evaluate chunking | `make evals` | downloads public datasets |

Full limits, per benchmark, are in [docs/limitations.md](docs/limitations.md). The
superseded circular diagnostics are still reachable as `make benchmark-legacy`; they score
strategies against LexiChunk's own parse, warn when they run, and are never evidence.

## Relationship to LexiChunk

[LexiChunk](https://pypi.org/project/lexichunk/) is a clause-aware chunker for legal
documents. This repository is not part of it: it is the instrument LexiChunk is measured
with, it is maintained separately, and it is written so that it can produce a result
LexiChunk's author does not want. It has — see
[`results/external/lexichunk_issues.md`](results/external/lexichunk_issues.md), which files
six defects the external evaluations found in the SDK, with reproducers, and the retrieval
comparison below, in which a plain character splitter is not beaten.

`lexichunk` is an ordinary dependency, installed from PyPI (`>=0.9.0,<0.10`). Every result
records the `lexichunk_version` and `lexichunk_commit` it ran against, so a published
number is always attached to a build. The local-editable route in
[Install](#install) exists for comparing two LexiChunk builds against each other, which is
what [`results/lexichunk_baseline/`](results/README.md) is for.

## Install

Python 3.10-3.12.

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

That is enough for `make benchmark`, `make benchmark-structural`, `make test` and the CLI.
`pip install lexichunk` is the default route for the chunker itself and needs nothing
else; it used to be a git reference pinned to a commit, because a version specifier
resolved to a PyPI 404 and a clean install — and therefore CI — could not succeed at all.

| Extra | Packages | Needed for |
|---|---|---|
| `dev` | ruff, mypy, pytest, pytest-cov | linting, type checking, the test suite |
| `embeddings` | sentence-transformers | `make benchmark-embed` (torch, weight downloads) |
| `evals` | datasets, scikit-learn | `make evals` (LEDGAR, CUAD) |
| `voyage` | voyageai | the paid `voyage-law-2` model |
| `dashboard` | streamlit | `make dashboard` |
| `all` | everything above | |

`faiss-cpu` and `plotly` are core dependencies, not extras: `legal_rag_eval.retrieval.index`
and `legal_rag_eval.reporting.html` import them at module scope, and the query-annotation
checks import the first, so `pip install -e ".[dev]" && make test` could not otherwise
collect three test modules. The five tests that load a real sentence-transformer model skip
when the `embeddings` extra is absent.

### Comparing two LexiChunk builds

To measure a LexiChunk working tree against the released one, install the extras **first**
and the editable checkout **last**:

```bash
pip install -e ".[dev]"
pip install -e /path/to/lexichunk        # must come last
python -c "import lexichunk; print(lexichunk.__file__)"   # verify
```

The order matters. Installing the extras after the editable checkout silently replaces it
with the released wheel, and the run then measures the released build while you believe it
is measuring your working tree. The verification line is not optional advice; this trap has
caught more than one run in this repository's history. A directory install records no VCS
metadata, so set `LEGAL_RAG_EVAL_LEXICHUNK_COMMIT` to have the run record which commit it
measured.

## Results

Everything in this section comes from a committed run. The structural and retrieval tables
are generated from `results/lexichunk_fixed/full_benchmark.json` by `make readme`; do not
edit them by hand, `make readme-check` fails if they drift and CI runs it on every push.
[`results/README.md`](results/README.md) lists every committed run with the LexiChunk build
that produced it and the command that regenerates it.

### Anchored evidence

Deterministic, offline, no model weights. Every strategy ranks candidate source spans with
the same lexical cosine term-frequency ranker and fills the same 128-token context budget;
the score is how much of the independently annotated evidence ends up inside that budget,
and how much of the budget is spent on something else. Unanswerable queries are scored
separately: abstention is correct only when no context was selected at all.

LexiChunk `0.9.0`, n = 12 answerable and 3 unanswerable queries. Committed as
`results/lexichunk_fixed/evidence_benchmark.json`; reproduce with `make benchmark`.

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
and size-matched control are in the gold-scored benchmark below.

### Structural and retrieval, against the gold annotations

**How to read these tables.**

*Retrieval.* With n = 30 queries and 56 tests per embedding model, **no comparison in this
run survives Holm correction** — not one, in either direction. Against the size-matched
control `rcts_1024`, LexiChunk is behind on MRR under both models (−0.059 under MiniLM,
−0.057 under BGE); both intervals comfortably contain zero, and `rcts_1024` has the better
R@5 and NDCG@10 of the two. At this sample size the harness does not separate clause-aware
chunking from a size-matched character splitter, and saying otherwise would mean reading
unadjusted p-values off a family of 112 tests.

*Structural.* The picture is split rather than one-sided. Against the size-matched control,
LexiChunk has much lower top-level over-merge (0.020 vs 0.159) and slightly higher
leaf-clause fragmentation (0.030 vs 0.008); `rcts_1024` groups sub-clauses together more
often (0.629 vs 0.279). Recursive character splitting on paragraph separators lands on
clause boundaries here more often than one would expect of a strategy that knows nothing
about clauses. The columns where LexiChunk is alone — heading recall/precision,
cross-reference recall/precision — read `n/a` for every other strategy, because those
strategies emit no headings or cross-references to score. Those are capability
measurements, not comparative wins: a metric only one entrant can be scored on cannot rank
the field, and this README does not use them to.

What the fixed build's numbers *do* support is narrower and checkable: its chunks are
exactly locatable in the source (100% against the previous build's 68.6%), and it recovers
cross-reference targets its predecessor did not (0.774 against 0.139). See
[Baseline vs fixed LexiChunk](#baseline-vs-fixed-lexichunk).

<!-- BEGIN GENERATED RESULTS -->

_Generated 2026-09-06T18:01:54.095994+00:00 — LexiChunk `0.9.0` (commit `0346a12`); embedding model(s): all-MiniLM-L6-v2, bge-base-en-v1.5; seed `0`; 5 documents; 30 queries. Strategy parameters: `{"fixed_size": {"chunk_size": 512}, "lexichunk": {}, "lexichunk_contextual": {}, "rcts_1024": {"chunk_overlap": 100, "chunk_size": 1024}, "rcts_512": {"chunk_overlap": 50, "chunk_size": 512}, "sentence_split": {"min_chunk_chars": 100}}`._

### Structural quality

Macro-averaged over documents (n = 5). Ground truth is hand-checked span annotations in `gold/`; see the methodology note below for what these metrics do not measure.

| Strategy | Located | Leaf Frag ↓ | Over-merge ↓ | Sub-clause Grp | Head R | Head P | Def Attach ↑ | XRef R | XRef P | Size CV | Avg Chars | Chunks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fixed_size | 100.000% | 0.357 | 0.154 | 0.143 | n/a | n/a | 0.033 | n/a | n/a | 0.149 | 496 | 39 |
| lexichunk | 100.000% | 0.030 | 0.020 | 0.279 | 0.453 | 0.875 | 0.082 | 0.774 | 0.459 | 0.494 | 552 | 38 |
| lexichunk_contextual | 87.002% | 0.089 | 0.025 | 0.300 | 0.443 | 0.922 | 0.087 | 0.744 | 0.463 | 0.375 | 755 | 38 |
| rcts_1024 | 100.000% | 0.008 | 0.159 | 0.629 | n/a | n/a | 0.088 | n/a | n/a | 0.212 | 826 | 24 |
| rcts_512 | 100.000% | 0.127 | 0.025 | 0.140 | n/a | n/a | 0.041 | n/a | n/a | 0.347 | 357 | 55 |
| sentence_split | 99.363% | 0.275 | 0.000 | 0.016 | n/a | n/a | 0.034 | n/a | n/a | 0.560 | 264 | 80 |

### Retrieval quality

**all-MiniLM-L6-v2** (mean over queries, n = 30)

| Strategy | P@1 | P@3 | P@5 | P@10 | R@1 | R@3 | R@5 | R@10 | MRR | NDCG@10 | DRM rate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fixed_size | 0.500 | 0.267 | 0.213 | 0.140 | 0.467 | 0.650 | 0.750 | 0.872 | 0.640 | 0.663 | 0.343 |
| lexichunk | 0.533 | 0.289 | 0.200 | 0.120 | 0.461 | 0.706 | 0.789 | 0.889 | 0.666 | 0.656 | 0.337 |
| lexichunk_contextual | 0.433 | 0.289 | 0.187 | 0.110 | 0.367 | 0.689 | 0.756 | 0.872 | 0.617 | 0.633 | 0.343 |
| rcts_1024 | 0.533 | 0.300 | 0.227 | 0.127 | 0.483 | 0.744 | 0.906 | 0.956 | 0.724 | 0.731 | 0.397 |
| rcts_512 | 0.567 | 0.289 | 0.193 | 0.113 | 0.461 | 0.678 | 0.744 | 0.839 | 0.686 | 0.665 | 0.310 |
| sentence_split | 0.500 | 0.322 | 0.200 | 0.120 | 0.417 | 0.672 | 0.689 | 0.806 | 0.650 | 0.625 | 0.340 |

**bge-base-en-v1.5** (mean over queries, n = 30)

| Strategy | P@1 | P@3 | P@5 | P@10 | R@1 | R@3 | R@5 | R@10 | MRR | NDCG@10 | DRM rate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fixed_size | 0.600 | 0.322 | 0.240 | 0.137 | 0.528 | 0.778 | 0.839 | 0.872 | 0.734 | 0.726 | 0.347 |
| lexichunk | 0.567 | 0.333 | 0.213 | 0.120 | 0.517 | 0.806 | 0.822 | 0.906 | 0.737 | 0.728 | 0.317 |
| lexichunk_contextual | 0.467 | 0.278 | 0.180 | 0.110 | 0.411 | 0.711 | 0.739 | 0.856 | 0.655 | 0.665 | 0.287 |
| rcts_1024 | 0.633 | 0.344 | 0.240 | 0.123 | 0.544 | 0.839 | 0.956 | 0.972 | 0.794 | 0.790 | 0.357 |
| rcts_512 | 0.667 | 0.333 | 0.227 | 0.127 | 0.561 | 0.767 | 0.850 | 0.867 | 0.786 | 0.734 | 0.317 |
| sentence_split | 0.567 | 0.300 | 0.213 | 0.113 | 0.511 | 0.694 | 0.772 | 0.806 | 0.698 | 0.675 | 0.340 |

### Strategy comparisons

The absolute delta and its bootstrap 95% CI are the headline figures — never a relative percent change. A row is significant only if it survives Holm correction across the whole family of tests below (every metric, pair and embedding model in this run). `rcts_1024` is the **size-matched control**: it uses the same target chunk size as LexiChunk, so it isolates what the chunking *strategy* contributes from what chunk *size* alone would contribute. The LODO range is the delta's spread across the five leave-one-document-out refits; a range straddling zero means one document carries the result.

**all-MiniLM-L6-v2** (56 tests, n = 30 queries each)

| Metric | A | B | n | Δ | 95% CI | p (Holm) | p (Wilcoxon) | Effect size (d) | LODO range |
|---|---|---|---|---|---|---|---|---|---|
| mrr | lexichunk | fixed_size | 30 | +0.026 | [-0.126, +0.179] | 1.000 | 0.670 | 0.059 | [+0.001, +0.045] |
| mrr | lexichunk | **rcts_1024 (size-matched control)** | 30 | -0.059 | [-0.212, +0.097] | 1.000 | 0.498 | -0.134 | [-0.099, -0.014] |
| mrr | lexichunk | rcts_512 | 30 | -0.021 | [-0.150, +0.104] | 1.000 | 0.795 | -0.057 | [-0.067, +0.012] |
| mrr | lexichunk | sentence_split | 30 | +0.016 | [-0.095, +0.134] | 1.000 | 0.955 | 0.047 | [-0.015, +0.053] |
| mrr | lexichunk_contextual | fixed_size | 30 | -0.023 | [-0.193, +0.144] | 1.000 | 0.808 | -0.047 | [-0.032, -0.010] |
| mrr | lexichunk_contextual | **rcts_1024 (size-matched control)** | 30 | -0.107 | [-0.243, +0.023] | 1.000 | 0.147 | -0.281 | [-0.144, -0.069] |
| mrr | lexichunk_contextual | rcts_512 | 30 | -0.069 | [-0.193, +0.050] | 1.000 | 0.325 | -0.199 | [-0.137, +0.000] |
| mrr | lexichunk_contextual | sentence_split | 30 | -0.033 | [-0.152, +0.084] | 1.000 | 0.448 | -0.098 | [-0.073, -0.019] |
| ndcg_at_10 | lexichunk | fixed_size | 30 | -0.007 | [-0.118, +0.105] | 1.000 | 0.913 | -0.024 | [-0.029, +0.012] |
| ndcg_at_10 | lexichunk | **rcts_1024 (size-matched control)** | 30 | -0.075 | [-0.189, +0.035] | 1.000 | 0.205 | -0.235 | [-0.119, -0.016] |
| ndcg_at_10 | lexichunk | rcts_512 | 30 | -0.009 | [-0.108, +0.088] | 1.000 | 0.970 | -0.033 | [-0.036, +0.013] |
| ndcg_at_10 | lexichunk | sentence_split | 30 | +0.030 | [-0.077, +0.145] | 1.000 | 0.962 | 0.096 | [-0.005, +0.067] |
| ndcg_at_10 | lexichunk_contextual | fixed_size | 30 | -0.030 | [-0.154, +0.092] | 1.000 | 0.575 | -0.087 | [-0.037, -0.025] |
| ndcg_at_10 | lexichunk_contextual | **rcts_1024 (size-matched control)** | 30 | -0.098 | [-0.204, +0.005] | 1.000 | 0.064 | -0.329 | [-0.129, -0.056] |
| ndcg_at_10 | lexichunk_contextual | rcts_512 | 30 | -0.032 | [-0.127, +0.059] | 1.000 | 0.695 | -0.119 | [-0.076, +0.014] |
| ndcg_at_10 | lexichunk_contextual | sentence_split | 30 | +0.007 | [-0.096, +0.114] | 1.000 | 0.765 | 0.026 | [-0.016, +0.028] |
| precision_at_1 | lexichunk | fixed_size | 30 | +0.033 | [-0.167, +0.233] | 1.000 | 0.739 | 0.060 | [+0.000, +0.042] |
| precision_at_1 | lexichunk | **rcts_1024 (size-matched control)** | 30 | +0.000 | [-0.200, +0.200] | 1.000 | 1.000 | 0.000 | [-0.042, +0.042] |
| precision_at_1 | lexichunk | rcts_512 | 30 | -0.033 | [-0.233, +0.167] | 1.000 | 0.739 | -0.060 | [-0.125, +0.000] |
| precision_at_1 | lexichunk | sentence_split | 30 | +0.033 | [-0.100, +0.167] | 1.000 | 0.655 | 0.081 | [+0.000, +0.083] |
| precision_at_1 | lexichunk_contextual | fixed_size | 30 | -0.067 | [-0.300, +0.167] | 1.000 | 0.564 | -0.104 | [-0.083, -0.042] |
| precision_at_1 | lexichunk_contextual | **rcts_1024 (size-matched control)** | 30 | -0.100 | [-0.267, +0.067] | 1.000 | 0.257 | -0.208 | [-0.125, -0.042] |
| precision_at_1 | lexichunk_contextual | rcts_512 | 30 | -0.133 | [-0.300, +0.033] | 1.000 | 0.157 | -0.263 | [-0.250, -0.083] |
| precision_at_1 | lexichunk_contextual | sentence_split | 30 | -0.067 | [-0.233, +0.100] | 1.000 | 0.414 | -0.148 | [-0.083, -0.042] |
| precision_at_10 | lexichunk | fixed_size | 30 | -0.020 | [-0.047, +0.007] | 1.000 | 0.192 | -0.280 | [-0.025, -0.017] |
| precision_at_10 | lexichunk | **rcts_1024 (size-matched control)** | 30 | -0.007 | [-0.027, +0.010] | 1.000 | 0.680 | -0.128 | [-0.013, +0.000] |
| precision_at_10 | lexichunk | rcts_512 | 30 | +0.007 | [-0.007, +0.020] | 1.000 | 0.317 | 0.183 | [+0.004, +0.008] |
| precision_at_10 | lexichunk | sentence_split | 30 | +0.000 | [-0.020, +0.020] | 1.000 | 0.861 | 0.000 | [-0.008, +0.008] |
| precision_at_10 | lexichunk_contextual | fixed_size | 30 | -0.030 | [-0.053, -0.007] | 1.000 | 0.029 | -0.461 | [-0.037, -0.025] |
| precision_at_10 | lexichunk_contextual | **rcts_1024 (size-matched control)** | 30 | -0.017 | [-0.033, -0.003] | 1.000 | 0.059 | -0.361 | [-0.021, -0.013] |
| precision_at_10 | lexichunk_contextual | rcts_512 | 30 | -0.003 | [-0.023, +0.013] | 1.000 | 0.705 | -0.068 | [-0.008, +0.004] |
| precision_at_10 | lexichunk_contextual | sentence_split | 30 | -0.010 | [-0.033, +0.010] | 1.000 | 0.366 | -0.165 | [-0.017, -0.004] |
| precision_at_5 | lexichunk | fixed_size | 30 | -0.013 | [-0.073, +0.047] | 1.000 | 0.660 | -0.081 | [-0.025, +0.008] |
| precision_at_5 | lexichunk | **rcts_1024 (size-matched control)** | 30 | -0.027 | [-0.067, +0.013] | 1.000 | 0.206 | -0.233 | [-0.042, -0.008] |
| precision_at_5 | lexichunk | rcts_512 | 30 | +0.007 | [-0.020, +0.033] | 1.000 | 0.655 | 0.081 | [+0.000, +0.017] |
| precision_at_5 | lexichunk | sentence_split | 30 | -0.000 | [-0.040, +0.040] | 1.000 | 1.000 | 0.000 | [-0.017, +0.008] |
| precision_at_5 | lexichunk_contextual | fixed_size | 30 | -0.027 | [-0.087, +0.033] | 1.000 | 0.396 | -0.155 | [-0.042, -0.008] |
| precision_at_5 | lexichunk_contextual | **rcts_1024 (size-matched control)** | 30 | -0.040 | [-0.080, -0.007] | 1.000 | 0.058 | -0.363 | [-0.050, -0.033] |
| precision_at_5 | lexichunk_contextual | rcts_512 | 30 | -0.007 | [-0.047, +0.027] | 1.000 | 0.705 | -0.068 | [-0.025, +0.017] |
| precision_at_5 | lexichunk_contextual | sentence_split | 30 | -0.013 | [-0.060, +0.027] | 1.000 | 0.564 | -0.104 | [-0.025, +0.000] |
| recall_at_10 | lexichunk | fixed_size | 30 | +0.017 | [-0.083, +0.117] | 1.000 | 0.785 | 0.060 | [-0.021, +0.042] |
| recall_at_10 | lexichunk | **rcts_1024 (size-matched control)** | 30 | -0.067 | [-0.167, +0.017] | 1.000 | 0.194 | -0.233 | [-0.104, -0.021] |
| recall_at_10 | lexichunk | rcts_512 | 30 | +0.050 | [+0.000, +0.100] | 1.000 | 0.083 | 0.328 | [+0.042, +0.062] |
| recall_at_10 | lexichunk | sentence_split | 30 | +0.083 | [-0.033, +0.200] | 1.000 | 0.238 | 0.238 | [+0.042, +0.125] |
| recall_at_10 | lexichunk_contextual | fixed_size | 30 | +0.000 | [-0.100, +0.100] | 1.000 | 1.000 | 0.000 | [-0.042, +0.062] |
| recall_at_10 | lexichunk_contextual | **rcts_1024 (size-matched control)** | 30 | -0.083 | [-0.167, -0.017] | 1.000 | 0.059 | -0.361 | [-0.104, -0.062] |
| recall_at_10 | lexichunk_contextual | rcts_512 | 30 | +0.033 | [-0.067, +0.133] | 1.000 | 0.577 | 0.114 | [+0.000, +0.083] |
| recall_at_10 | lexichunk_contextual | sentence_split | 30 | +0.067 | [-0.033, +0.183] | 1.000 | 0.234 | 0.212 | [+0.021, +0.083] |
| recall_at_5 | lexichunk | fixed_size | 30 | +0.039 | [-0.133, +0.211] | 1.000 | 0.754 | 0.078 | [-0.014, +0.090] |
| recall_at_5 | lexichunk | **rcts_1024 (size-matched control)** | 30 | -0.117 | [-0.267, +0.033] | 1.000 | 0.149 | -0.272 | [-0.167, -0.062] |
| recall_at_5 | lexichunk | rcts_512 | 30 | +0.044 | [-0.061, +0.150] | 1.000 | 0.414 | 0.150 | [-0.007, +0.076] |
| recall_at_5 | lexichunk | sentence_split | 30 | +0.100 | [-0.067, +0.267] | 1.000 | 0.305 | 0.216 | [+0.021, +0.167] |
| recall_at_5 | lexichunk_contextual | fixed_size | 30 | +0.006 | [-0.161, +0.167] | 1.000 | 0.952 | 0.012 | [-0.021, +0.049] |
| recall_at_5 | lexichunk_contextual | **rcts_1024 (size-matched control)** | 30 | -0.150 | [-0.300, +0.000] | 1.000 | 0.083 | -0.359 | [-0.188, -0.104] |
| recall_at_5 | lexichunk_contextual | rcts_512 | 30 | +0.011 | [-0.100, +0.117] | 1.000 | 0.892 | 0.037 | [-0.042, +0.076] |
| recall_at_5 | lexichunk_contextual | sentence_split | 30 | +0.067 | [-0.100, +0.233] | 1.000 | 0.420 | 0.142 | [+0.000, +0.125] |

**bge-base-en-v1.5** (56 tests, n = 30 queries each)

| Metric | A | B | n | Δ | 95% CI | p (Holm) | p (Wilcoxon) | Effect size (d) | LODO range |
|---|---|---|---|---|---|---|---|---|---|
| mrr | lexichunk | fixed_size | 30 | +0.003 | [-0.091, +0.098] | 1.000 | 0.886 | 0.012 | [-0.044, +0.042] |
| mrr | lexichunk | **rcts_1024 (size-matched control)** | 30 | -0.057 | [-0.181, +0.069] | 1.000 | 0.546 | -0.161 | [-0.145, +0.007] |
| mrr | lexichunk | rcts_512 | 30 | -0.049 | [-0.138, +0.029] | 1.000 | 0.356 | -0.204 | [-0.082, -0.024] |
| mrr | lexichunk | sentence_split | 30 | +0.039 | [-0.081, +0.168] | 1.000 | 0.704 | 0.108 | [-0.056, +0.086] |
| mrr | lexichunk_contextual | fixed_size | 30 | -0.079 | [-0.175, -0.001] | 1.000 | 0.138 | -0.314 | [-0.099, -0.043] |
| mrr | lexichunk_contextual | **rcts_1024 (size-matched control)** | 30 | -0.140 | [-0.270, -0.015] | 1.000 | 0.041 | -0.387 | [-0.202, -0.116] |
| mrr | lexichunk_contextual | rcts_512 | 30 | -0.131 | [-0.248, -0.022] | 1.000 | 0.036 | -0.414 | [-0.149, -0.081] |
| mrr | lexichunk_contextual | sentence_split | 30 | -0.044 | [-0.144, +0.048] | 1.000 | 0.530 | -0.160 | [-0.055, -0.020] |
| ndcg_at_10 | lexichunk | fixed_size | 30 | +0.002 | [-0.067, +0.071] | 1.000 | 0.909 | 0.009 | [-0.048, +0.024] |
| ndcg_at_10 | lexichunk | **rcts_1024 (size-matched control)** | 30 | -0.062 | [-0.154, +0.032] | 1.000 | 0.211 | -0.235 | [-0.125, -0.012] |
| ndcg_at_10 | lexichunk | rcts_512 | 30 | -0.007 | [-0.071, +0.057] | 1.000 | 1.000 | -0.036 | [-0.032, +0.010] |
| ndcg_at_10 | lexichunk | sentence_split | 30 | +0.053 | [-0.041, +0.158] | 1.000 | 0.469 | 0.186 | [-0.013, +0.081] |
| ndcg_at_10 | lexichunk_contextual | fixed_size | 30 | -0.061 | [-0.134, -0.001] | 1.000 | 0.117 | -0.320 | [-0.076, -0.053] |
| ndcg_at_10 | lexichunk_contextual | **rcts_1024 (size-matched control)** | 30 | -0.125 | [-0.222, -0.030] | 1.000 | 0.020 | -0.458 | [-0.170, -0.103] |
| ndcg_at_10 | lexichunk_contextual | rcts_512 | 30 | -0.069 | [-0.144, +0.002] | 1.000 | 0.129 | -0.331 | [-0.088, -0.038] |
| ndcg_at_10 | lexichunk_contextual | sentence_split | 30 | -0.010 | [-0.072, +0.056] | 1.000 | 0.649 | -0.053 | [-0.019, -0.001] |
| precision_at_1 | lexichunk | fixed_size | 30 | -0.033 | [-0.200, +0.133] | 1.000 | 0.705 | -0.068 | [-0.125, +0.042] |
| precision_at_1 | lexichunk | **rcts_1024 (size-matched control)** | 30 | -0.067 | [-0.267, +0.133] | 1.000 | 0.527 | -0.114 | [-0.208, +0.042] |
| precision_at_1 | lexichunk | rcts_512 | 30 | -0.100 | [-0.233, +0.033] | 1.000 | 0.180 | -0.248 | [-0.167, -0.042] |
| precision_at_1 | lexichunk | sentence_split | 30 | +0.000 | [-0.167, +0.167] | 1.000 | 1.000 | 0.000 | [-0.125, +0.083] |
| precision_at_1 | lexichunk_contextual | fixed_size | 30 | -0.133 | [-0.267, -0.033] | 1.000 | 0.046 | -0.386 | [-0.167, -0.083] |
| precision_at_1 | lexichunk_contextual | **rcts_1024 (size-matched control)** | 30 | -0.167 | [-0.333, +0.000] | 1.000 | 0.059 | -0.361 | [-0.250, -0.125] |
| precision_at_1 | lexichunk_contextual | rcts_512 | 30 | -0.200 | [-0.367, -0.033] | 1.000 | 0.034 | -0.413 | [-0.250, -0.125] |
| precision_at_1 | lexichunk_contextual | sentence_split | 30 | -0.100 | [-0.233, +0.033] | 1.000 | 0.180 | -0.248 | [-0.125, -0.083] |
| precision_at_10 | lexichunk | fixed_size | 30 | -0.017 | [-0.043, +0.007] | 1.000 | 0.275 | -0.238 | [-0.021, -0.012] |
| precision_at_10 | lexichunk | **rcts_1024 (size-matched control)** | 30 | -0.003 | [-0.017, +0.010] | 1.000 | 0.655 | -0.081 | [-0.008, +0.000] |
| precision_at_10 | lexichunk | rcts_512 | 30 | -0.007 | [-0.033, +0.013] | 1.000 | 0.739 | -0.096 | [-0.012, +0.000] |
| precision_at_10 | lexichunk | sentence_split | 30 | +0.007 | [-0.010, +0.023] | 1.000 | 0.480 | 0.128 | [+0.000, +0.013] |
| precision_at_10 | lexichunk_contextual | fixed_size | 30 | -0.027 | [-0.050, -0.003] | 1.000 | 0.066 | -0.386 | [-0.033, -0.017] |
| precision_at_10 | lexichunk_contextual | **rcts_1024 (size-matched control)** | 30 | -0.013 | [-0.030, +0.000] | 1.000 | 0.102 | -0.307 | [-0.021, -0.008] |
| precision_at_10 | lexichunk_contextual | rcts_512 | 30 | -0.017 | [-0.043, +0.003] | 1.000 | 0.197 | -0.238 | [-0.021, -0.012] |
| precision_at_10 | lexichunk_contextual | sentence_split | 30 | -0.003 | [-0.020, +0.013] | 1.000 | 0.705 | -0.068 | [-0.004, +0.000] |
| precision_at_5 | lexichunk | fixed_size | 30 | -0.027 | [-0.067, +0.013] | 1.000 | 0.206 | -0.233 | [-0.033, -0.017] |
| precision_at_5 | lexichunk | **rcts_1024 (size-matched control)** | 30 | -0.027 | [-0.067, +0.007] | 1.000 | 0.157 | -0.263 | [-0.033, -0.017] |
| precision_at_5 | lexichunk | rcts_512 | 30 | -0.013 | [-0.053, +0.020] | 1.000 | 0.480 | -0.128 | [-0.025, +0.000] |
| precision_at_5 | lexichunk | sentence_split | 30 | +0.000 | [-0.033, +0.033] | 1.000 | 1.000 | 0.000 | [-0.017, +0.017] |
| precision_at_5 | lexichunk_contextual | fixed_size | 30 | -0.060 | [-0.100, -0.020] | 0.515 | 0.007 | -0.561 | [-0.067, -0.050] |
| precision_at_5 | lexichunk_contextual | **rcts_1024 (size-matched control)** | 30 | -0.060 | [-0.107, -0.020] | 1.000 | 0.014 | -0.503 | [-0.075, -0.050] |
| precision_at_5 | lexichunk_contextual | rcts_512 | 30 | -0.047 | [-0.087, -0.013] | 1.000 | 0.020 | -0.463 | [-0.058, -0.042] |
| precision_at_5 | lexichunk_contextual | sentence_split | 30 | -0.033 | [-0.067, +0.000] | 1.000 | 0.059 | -0.361 | [-0.042, -0.025] |
| recall_at_10 | lexichunk | fixed_size | 30 | +0.033 | [-0.033, +0.100] | 1.000 | 0.317 | 0.183 | [+0.021, +0.062] |
| recall_at_10 | lexichunk | **rcts_1024 (size-matched control)** | 30 | -0.067 | [-0.133, -0.017] | 1.000 | 0.046 | -0.386 | [-0.083, -0.042] |
| recall_at_10 | lexichunk | rcts_512 | 30 | +0.039 | [-0.017, +0.100] | 1.000 | 0.131 | 0.232 | [+0.028, +0.049] |
| recall_at_10 | lexichunk | sentence_split | 30 | +0.100 | [+0.017, +0.200] | 1.000 | 0.063 | 0.363 | [+0.042, +0.125] |
| recall_at_10 | lexichunk_contextual | fixed_size | 30 | -0.017 | [-0.100, +0.067] | 1.000 | 0.705 | -0.068 | [-0.042, +0.021] |
| recall_at_10 | lexichunk_contextual | **rcts_1024 (size-matched control)** | 30 | -0.117 | [-0.217, -0.033] | 1.000 | 0.020 | -0.463 | [-0.146, -0.083] |
| recall_at_10 | lexichunk_contextual | rcts_512 | 30 | -0.011 | [-0.100, +0.067] | 1.000 | 1.000 | -0.047 | [-0.035, +0.028] |
| recall_at_10 | lexichunk_contextual | sentence_split | 30 | +0.050 | [-0.033, +0.133] | 1.000 | 0.257 | 0.208 | [+0.042, +0.062] |
| recall_at_5 | lexichunk | fixed_size | 30 | -0.017 | [-0.133, +0.083] | 1.000 | 0.679 | -0.054 | [-0.042, +0.021] |
| recall_at_5 | lexichunk | **rcts_1024 (size-matched control)** | 30 | -0.133 | [-0.250, -0.033] | 1.000 | 0.038 | -0.417 | [-0.167, -0.083] |
| recall_at_5 | lexichunk | rcts_512 | 30 | -0.028 | [-0.144, +0.072] | 1.000 | 0.750 | -0.089 | [-0.076, +0.007] |
| recall_at_5 | lexichunk | sentence_split | 30 | +0.050 | [-0.067, +0.167] | 1.000 | 0.450 | 0.151 | [-0.021, +0.083] |
| recall_at_5 | lexichunk_contextual | fixed_size | 30 | -0.100 | [-0.233, +0.000] | 1.000 | 0.098 | -0.301 | [-0.125, -0.062] |
| recall_at_5 | lexichunk_contextual | **rcts_1024 (size-matched control)** | 30 | -0.217 | [-0.367, -0.083] | 0.516 | 0.009 | -0.560 | [-0.271, -0.167] |
| recall_at_5 | lexichunk_contextual | rcts_512 | 30 | -0.111 | [-0.206, -0.033] | 1.000 | 0.024 | -0.454 | [-0.139, -0.076] |
| recall_at_5 | lexichunk_contextual | sentence_split | 30 | -0.033 | [-0.117, +0.067] | 1.000 | 0.480 | -0.128 | [-0.042, -0.021] |

### How to read this

Mean chunk length by strategy — a retrieval comparison is only size-matched when these are close; otherwise an apparent win may just be a chunk-size effect:

- `fixed_size`: 496 chars/chunk
- `lexichunk`: 552 chars/chunk
- `lexichunk_contextual`: 755 chars/chunk
- `rcts_1024`: 826 chars/chunk
- `rcts_512`: 357 chars/chunk
- `sentence_split`: 264 chars/chunk

<!-- END GENERATED RESULTS -->

### Baseline vs fixed LexiChunk

A chunking benchmark that cannot tell two versions of the same chunker apart is not
measuring the chunker. This is that check, run twice with everything except the LexiChunk
build held fixed — same fixtures, same gold, same 30 queries, same seed (0), same two
embedding models. Both runs are committed: `results/lexichunk_baseline/` (`32078cd`,
0.8.0b1) and `results/lexichunk_fixed/` (`0346a12`, 0.9.0). Reproduce the table with
`make compare-builds`.

| Measure (strategy `lexichunk`) | baseline — LexiChunk `0.8.0b1` `32078cd` | fixed — LexiChunk `0.9.0` `0346a12` |
|---|---:|---:|
| Chunks located in the source text (n = 5 documents) | 0.686 | 1.000 |
| Leaf-clause fragmentation (lower is better) (n = 5 documents) | 0.422 | 0.030 |
| Top-level over-merge (lower is better) (n = 5 documents) | 0.040 | 0.020 |
| Heading attachment recall (n = 5 documents) | 0.441 | 0.453 |
| Definition attachment recall (n = 5 documents) | 0.092 | 0.082 |
| Cross-reference target recall (n = 5 documents) | 0.139 | 0.774 |
| Mean chunk length (chars) (n = 5 documents) | 514 | 552 |
| MRR, all-MiniLM-L6-v2 (n = 30 queries) | 0.408 | 0.666 |
| MRR, bge-base-en-v1.5 (n = 30 queries) | 0.542 | 0.737 |
| R@5, all-MiniLM-L6-v2 (n = 30 queries) | 0.467 | 0.789 |
| R@5, bge-base-en-v1.5 (n = 30 queries) | 0.517 | 0.822 |
| _control_: `rcts_1024` MRR, all-MiniLM-L6-v2 | 0.724 | 0.724 |
| _control_: `rcts_1024` MRR, bge-base-en-v1.5 | 0.794 | 0.794 |
| Anchored-evidence recall (n = 12 answerable) | 0.892 | 0.976 |
| Anchored-evidence precision (n = 12 answerable) | 0.164 | 0.255 |

**The harness discriminates between LexiChunk builds, but this is not a retrieval win.** In
the baseline run 61 of the 112 statistical comparisons are significant after Holm
correction; in the fixed run none is. The control rows are the reason to believe the
difference is the dependency rather than the harness: `rcts_1024` never touches LexiChunk,
and its numbers are identical to three decimal places across the two runs. But most of the
gap is one defect — the baseline emitted chunk text that was not a contiguous substring of
the source, so only 69% of its chunks could be located and an unlocated chunk can never be
scored relevant. That is a localisation failure, not evidence that the fixed build retrieves
better in a way a user would feel.

`make compare-builds` prints the full table. The rest of the analysis — including which
part of the cross-reference improvement is a harness scoring fix rather than a chunker
change — is in [docs/build-comparison.md](docs/build-comparison.md).

### External: LEDGAR and CUAD

Five synthetic-ish documents cannot support a claim about legal documents in general, so
LexiChunk is also measured against two public datasets with independent human labels that
neither this harness nor the SDK was tuned on. Full methodology, per-class tables and the
issues found are in [docs/external_evals.md](docs/external_evals.md); the runs are committed
under [`results/external/`](results/external/).

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
| `lexichunk-128tok` | 306 | **81.2%** [79.6, 82.8] | **+37.5 pp** |
| `rcts-512` | 382 | 72.1% [70.2, 74.0] | +22.2 pp |

Raw containment for the larger settings (`lexichunk-512tok` 98.3%, `sentence-window-3`
98.9%) is not quoted as a win: containment rises mechanically with chunk length and with
overlap, and only 54.7% of the sentence-window matches are single-chunk. The length-matched
comparison is the one that carries information.

The same run also found that LexiChunk's `us` profile finds five or more top-level clauses
in **45%** of these 100 US contracts and falls back to flat text in 31% — while the `uk`
profile on the *same* US contracts manages only 31%. Getting a better result on US
contracts from the wrong jurisdiction profile is a finding against the SDK, not for it, and
it is filed with a reproducer in
[`results/external/lexichunk_issues.md`](results/external/lexichunk_issues.md) along with
five others. No exceptions were raised across 150 contracts.

## Reproduce

Every command here is run in CI or by `make ci`, against the committed evidence.

```bash
make benchmark             # anchored evidence   — offline, seconds
make benchmark-structural  # gold structural     — offline, seconds
make benchmark-embed       # + retrieval         — needs [embeddings], minutes
make evals                 # LEDGAR + CUAD       — needs [evals], downloads datasets
make evals-smoke           # offline synthetic path; the numbers are meaningless by design
make compare-builds        # baseline vs fixed table
make report                # HTML report from the last structural or full run
make readme                # regenerate the results block above
make readme-check          # exit 1 if that block is stale
make dashboard             # Streamlit viewer — needs [dashboard]
```

Ad-hoc runs write to `results/` with the CLI's default filenames and are not committed, so
a local run never overwrites the evidence behind the tables above. The two published runs
are written to build-named directories:

```bash
LEGAL_RAG_EVAL_LEXICHUNK_COMMIT=$(git -C /path/to/lexichunk rev-parse --short HEAD) \
  python -m legal_rag_eval benchmark-embed --json --seed 0 \
    --models all-MiniLM-L6-v2,bge-base-en-v1.5 \
    --output-dir results/lexichunk_fixed
```

Select a different model or strategy set without editing source:

```bash
python -m legal_rag_eval benchmark-embed --models bge-base-en-v1.5 --json
python -m legal_rag_eval benchmark-embed --strategies lexichunk,rcts_1024 --top-k 20 --seed 7
```

`voyage-law-2` needs **both** `--enable-voyage` and `VOYAGE_API_KEY`. An exported key on
its own never causes a paid API call.

The CLI is installed as `legal-rag-eval`, and `python -m legal_rag_eval` reaches the same
entry point. Its subcommands are `benchmark`, `benchmark-structural`, `benchmark-embed`,
`benchmark-legacy` and `report`; the external evaluations are
`python -m legal_rag_eval.evals ledgar|cuad`.

## What this does not measure

**This is explicitly not lawyer validated, held out, customer proof, or evidence of market
superiority.** It is useful for testing evaluator behaviour and exposing wins, losses and
mixed trade-offs — not for making legal or product-performance claims. In particular it
does not measure answer quality (nothing here runs a generator), whether a chunk is
*useful* to a reader, or generalisation: five fixture documents, four of them synthetic,
and only `eu_gdpr_excerpt.txt` is a real instrument. Chunk size is not controlled away
either — longer chunks straddle more gold spans and score better on overlap-based
relevance, which is why `rcts_1024` exists as a size-matched control and why every table
states the mean chunk length that produced it.

The full list, benchmark by benchmark, is in [docs/limitations.md](docs/limitations.md).

Three properties are worth stating positively, because they are what make the numbers
checkable:

- **Ground truth is independent of every chunker under test.** No chunker is consulted when
  the gold annotations are made — see [docs/ground-truth.md](docs/ground-truth.md).
- **Every structural metric is character-span overlap in the document.** A chunk's span is
  found by matching its text against the source, the same way for every strategy, never
  from a chunker's self-reported offsets. A chunk that is not a contiguous span of the
  source is reported as unlocated rather than silently scored.
- **LexiChunk's generated content earns no credit.** In the anchored-evidence benchmark it
  is scored boundary-only: ranking and budget use `source[char_start:char_end]` and nothing
  else. Generated ancestor headers, definition expansion and cross-reference expansion get
  no ranking text, no budget and no evidence credit.

## Repository layout

```
src/legal_rag_eval/   The harness: chunking, embedding, retrieval, metrics, reporting,
                      CLI (`legal-rag-eval`), and the Streamlit dashboard
gold/                 Hand-checked structural ground truth, one JSON per fixture document
queries/              30 annotated queries, relevant clauses given by gold identifier
evals/                Label mappings for the external evaluations       (evals/README.md)
results/              Committed benchmark runs — the evidence behind   (results/README.md)
                      every published number
tools/                Maintenance scripts, each behind a make target     (tools/README.md)
tests/                The test suite; `make test` enforces 80% coverage
docs/                 Methodology, metrics, limitations, ground truth, external evals
```

Detail on the pipeline, the strategies compared and every metric reported is in
[docs/methodology.md](docs/methodology.md); formulas and blind spots per metric are in
[docs/metrics.md](docs/metrics.md); extension guides are in
[EXTENSIBILITY.md](EXTENSIBILITY.md).

## Configuration

```bash
cp legal-rag-eval.yaml.example legal-rag-eval.yaml
```

One file configures all three benchmarks. It has a section per benchmark — `evidence:`
for the anchored-evidence benchmark, `gold:` for the structural and retrieval ones —
because the two have disjoint key sets and each rejects keys it does not recognise. Each
loader reads only its own section and ignores the other's. Every documented key takes
effect, and the resolved configuration is written into the results JSON.

**Precedence, highest first:**

| | Source | Example |
|---|---|---|
| 1 | CLI flags | `--top-k 20` |
| 2 | Environment | `LEGAL_RAG_EVAL_TOP_K=20` |
| 3 | Config file | `legal-rag-eval.yaml`, or `--config PATH` |
| 4 | Defaults | `legal_rag_eval/config.py`, `legal_rag_eval/evidence/benchmark.py` |

A setting named at a higher level replaces the same setting from a lower one; lists are
replaced whole, never merged key by key.

```bash
export LEGAL_RAG_EVAL_STRATEGIES="lexichunk,rcts_1024"
export LEGAL_RAG_EVAL_TOP_K=20
export LEGAL_RAG_EVAL_RELEVANCE_MIN_OVERLAP_CHARS=150
export VOYAGE_API_KEY=your-key-here    # enables voyage-law-2
```

The pre-rename `SCAFFOLDER_*` prefix is still read for one release and warns when used; a
`LEGAL_RAG_EVAL_*` value always wins over the old name.

See [EXTENSIBILITY.md](EXTENSIBILITY.md) for extension guides.

## Development

```bash
make lint            # ruff check + format verification
make typecheck       # mypy --strict
make test            # pytest with 80% coverage minimum
make ci              # all of the above
make readme-check    # fail if the generated results block is stale
```

`.github/workflows/ci.yml` runs lint, mypy, the test suite on Python 3.10/3.11/3.12, the
gold and query annotation validators, a clean wheel install, and a structural-benchmark
smoke run on every push. The retrieval benchmark downloads model weights, so it is
`workflow_dispatch` only.

Notable changes are recorded in [CHANGELOG.md](CHANGELOG.md), and
[CONTRIBUTING.md](CONTRIBUTING.md) covers the review expectations.

## Provenance

This repository was built by AI agents, independently audited — the audit found the original
methodology circular and the original headline claims unsupported — and then rebuilt twice in
parallel by two different tools. What is here now reconciles both rebuilds:

- The **anchored-evidence benchmark** (`src/legal_rag_eval/evidence/`), its dataset interface,
  the report hashing, the wheel/offline CI smoke tests and the honesty framing come from the
  overhaul merged as PRs #1 and #2.
- The **gold annotations** (`gold/`), the span-overlap metrics, the statistics module, the
  size-matched `rcts_1024` control, the 30-query gold-anchored set and the generated results
  tables come from the parallel audit-fix branch.
- The **external LEDGAR and CUAD evaluations** (`src/legal_rag_eval/evals/`) come from a third
  branch and are the only ground truth in this repository that neither rebuild authored.

Where the two rebuilds disagreed on the same point, the reconciliation kept the one with
evidence behind it and recorded the choice. The circular original methodology is reachable
only as `benchmark-legacy`, which warns when it runs.

## License

MIT — see [LICENSE](LICENSE)
