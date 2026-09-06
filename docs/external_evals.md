# External evaluations: LexiChunk on public legal datasets

LexiChunk's README advertises a 31-type keyword clause classifier and
clause-boundary chunking, with no published accuracy number. This document
supplies two, measured on public data that LexiChunk was not tuned against:

1. **LEDGAR** (LexGLUE) — how accurate is the clause-type classifier?
2. **CUAD** — do LexiChunk's chunk boundaries preserve expert-annotated answer
   spans better than general-purpose splitters?

Both are run by `legal_rag_eval.evals`, are deterministic given a seed, and write
their raw output to `results/external/`.

## Headline

| Question | Answer |
|----------|--------|
| How accurate is the keyword clause classifier? | **39.6%** accuracy, **42.5%** macro-F1 on 3,955 LEDGAR test provisions |
| Is that good? | Roughly double the 21.6% majority-class floor, and less than half the 90.8% a TF-IDF + logistic regression baseline reaches on the same items |
| Does higher confidence mean higher accuracy? | Directionally yes (Spearman 0.375), but the curve is not monotonic and the model is badly under-confident; ECE 0.113 |
| Do LexiChunk's boundaries preserve gold answer spans? | **81.2%** vs **72.1%** for `RecursiveCharacterTextSplitter` at the same mean chunk length — a genuine +9.1pp |
| Does LexiChunk find contract structure on real filings? | Better than before under `us`, still partial: 45% of CUAD contracts yield >= 5 top-level clauses (up from 13% on 0.8.0b1, re-run fresh — the 0.9.0 bare-decimal heading fix); 31% still get no usable hierarchy at all. The `uk` profile moved too, in the other direction: 31% now vs 42% on 0.8.0b1 |
| Does it crash on real filings? | No. Zero exceptions across 150 real SEC contracts |

## Results

### LEDGAR — clause-type classification

Dataset `coastalcph/lex_glue`, config `ledgar`, revision
`c23fdff1a6bf74e0e1a71cb86f1e781d37da888c`, split `test` (10,000 rows).
5,000 rows sampled with seed 0; **3,955 in scope** (79.1% item coverage,
1,045 dropped). **67 of 100** LEDGAR labels map onto **19** distinct LexiChunk
classes.

| System | Accuracy (95% CI) | Macro-F1 (95% CI) | Seconds |
| --- | --- | --- | --- |
| **lexichunk-keyword** | **39.6% [38.0, 41.2]** | **42.5% [40.7, 44.1]** | 0.4 |
| majority (`boilerplate`) | 21.6% [20.3, 22.8] | 1.9% [1.8, 2.0] | 0.0 |
| tfidf+logreg (supervised) | 90.8% [89.9, 91.7] | 89.9% [88.5, 91.0] | 18.5 |

The keyword classifier is unambiguously doing real work — 39.6% is 18 points
clear of the majority floor, and 42.5% macro-F1 against the majority
baseline's 1.9% shows it is spreading its predictions across classes rather
than exploiting the class imbalance. It is also unambiguously far from what
even a weak supervised model achieves on the same task, and the gap is much
too large to be noise.

**Per-class: the classifier has two populations.**

| LexiChunk class | Support | Predicted | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- |
| boilerplate | 854 | 214 | 79.9% | 20.0% | 32.0% |
| representations | 591 | 128 | 50.0% | 10.8% | 17.8% |
| payment | 379 | 395 | 54.7% | 57.0% | 55.8% |
| governing_law | 316 | 594 | 42.8% | 80.4% | 55.8% |
| covenants | 251 | 320 | 8.1% | 10.4% | 9.1% |
| entire_agreement | 232 | 169 | 91.7% | 66.8% | 77.3% |
| notices | 226 | 217 | 73.3% | 70.4% | 71.8% |
| severability | 200 | 113 | 93.8% | 53.0% | 67.7% |
| assignment | 152 | 205 | 57.6% | 77.6% | 66.1% |
| dispute_resolution | 145 | 42 | 26.2% | 7.6% | 11.8% |
| amendment | 133 | 354 | 24.0% | 63.9% | 34.9% |
| definitions | 88 | 81 | 12.3% | 11.4% | 11.8% |
| indemnification | 82 | 59 | 79.7% | 57.3% | 66.7% |
| confidentiality | 74 | 76 | 52.6% | 54.1% | 53.3% |
| insurance | 65 | 51 | 84.3% | 66.2% | 74.1% |
| warranties | 53 | 34 | 2.9% | 1.9% | 2.3% |
| termination | 50 | 126 | 26.2% | 66.0% | 37.5% |
| conditions | 38 | 10 | 20.0% | 5.3% | 8.3% |
| intellectual_property | 26 | 68 | 36.8% | 96.2% | 53.2% |

**Dominant confusions** (2,389 errors in total):

| Gold | Predicted | Count | % of errors |
| --- | --- | --- | --- |
| representations | unknown | 192 | 8.0% |
| boilerplate | covenants | 153 | 6.4% |
| boilerplate | amendment | 146 | 6.1% |
| boilerplate | unknown | 125 | 5.2% |
| covenants | unknown | 83 | 3.5% |
| dispute_resolution | governing_law | 80 | 3.3% |
| representations | governing_law | 67 | 2.8% |
| boilerplate | payment | 60 | 2.5% |
| payment | unknown | 52 | 2.2% |
| covenants | governing_law | 51 | 2.1% |

**Calibration.**

| Confidence bin | n | Mean confidence | Accuracy |
| --- | --- | --- | --- |
| [0.0, 0.1) | 675 | 0.006 | 0.7% |
| [0.1, 0.2) | 307 | 0.137 | 27.4% |
| [0.2, 0.3) | 1,202 | 0.246 | 49.2% |
| [0.3, 0.4) | 578 | 0.339 | **24.7%** |
| [0.4, 0.5) | 150 | 0.445 | 49.3% |
| [0.5, 0.6) | 586 | 0.513 | 53.4% |
| [0.6, 0.7) | 65 | 0.649 | 67.7% |
| [0.7, 0.8) | 199 | 0.747 | 78.9% |
| [0.8, 0.9) | 68 | 0.822 | **72.1%** |
| [0.9, 1.0) | 125 | 1.000 | 84.8% |

Expected calibration error **0.113**; Spearman correlation between confidence
and correctness **0.375**; **not monotonic**.

### CUAD — gold-span containment

Dataset `theatticusproject/cuad-qa` at revision `refs/convert/parquet`, splits
`train+test`; 100 of 510 contracts sampled with seed 0, carrying **2,458
verified gold spans** (mean 242 chars, median 182). Mean contract length 47,432
chars.

| Strategy | Containment (95% CI) | Mean chunk chars | Chunks/contract | Length-matched grid | Lift | s/contract |
| --- | --- | --- | --- | --- | --- | --- |
| lexichunk-512tok | 98.3% [97.7, 98.7] | 993 | 47.8 | 76.4% | +21.8pp | 0.073 |
| lexichunk-1024tok | 99.0% [98.6, 99.4] | 1,231 | 38.5 | 80.7% | +18.3pp | 0.067 |
| **lexichunk-128tok** | **81.2% [79.6, 82.8]** | 306 | 155.2 | 43.7% | **+37.5pp** | 0.077 |
| **rcts-512** | **72.1% [70.2, 74.0]** | 382 | 123.0 | 49.9% | +22.2pp | 0.006 |
| rcts-1024 | 89.3% [88.1, 90.6] | 744 | 63.4 | 69.5% | +19.9pp | 0.004 |
| sentence-window-3 | 98.9% [98.5, 99.3] | 647 | 109.3 | 65.6% | +33.3pp | 0.001 |

**How many chunks a gold span is split across:**

| Strategy | 1 | 2 | 3 | 4+ | Mean |
| --- | --- | --- | --- | --- | --- |
| lexichunk-512tok | 98.3% | 1.5% | 0.2% | 0.0% | 1.02 |
| lexichunk-1024tok | 99.0% | 0.8% | 0.2% | 0.0% | 1.01 |
| lexichunk-128tok | 81.2% | 13.6% | 2.8% | 2.4% | 1.30 |
| rcts-512 | 72.1% | 24.4% | 2.6% | 0.9% | 1.33 |
| rcts-1024 | 89.3% | 10.0% | 0.4% | 0.2% | 1.12 |
| sentence-window-3 | 54.7% | 44.4% | 0.8% | 0.1% | 1.47 |

Chunks per contract rose and mean chunk length shrank for every `lexichunk-*`
row relative to a **fresh, same-harness 0.8.0b1 run** (`lexichunk-512tok`:
2,106 -> 993 mean chars, 33.8 -> 47.8 chunks/contract; `lexichunk-1024tok`:
3,091 -> 1,231; `lexichunk-128tok`: 737 -> 306). This follows from the
jurisdiction/structure fix below: on the 46% of these filings where 0.8.0b1
found no usable structure, it produced chunks sized against the raw
`max_chunk_size` token ceiling regardless of clause boundaries — the largest
chunks it can legally emit; 0.9.0 recognises far more of the real heading
structure and stops chunks at clause boundaries below that ceiling, so its
chunks are consistently smaller. Per-contract latency did **not** meaningfully
change (0.8.0b1: 0.075/0.071/0.105s for 512/1024/128tok; 0.9.0: 0.073/0.067/
0.077s) — this correction supersedes an earlier draft of this document that
compared 0.9.0's fresh latency against a stale 0.66–0.71s figure from a
much older harness snapshot and reported a false ~10x speedup; see the
"One version, mostly" note under Threats to validity. Containment and lift
move only slightly because both are dominated by chunk *length*, and larger
0.8.0b1 chunks trivially contain more (see the artefact discussion below).

### CUAD — structure recall and cost

| Metric | Value |
| --- | --- |
| Contracts with >= 5 **top-level** clauses | 45 (45.0%) |
| Contracts with >= 5 nodes at **any** level | 69 (69.0%) |
| Fell back to flat text | 31 (31.0%) |
| Mean / median top-level nodes | 7.6 / 3 |
| Mean / median total nodes | 54.1 / 12 |
| Contracts yielding a single chunk | 4 |
| **Exceptions during parsing** | **0** |

Same contracts, both jurisdiction profiles:

| Measure | `us` | `uk` |
| --- | --- | --- |
| Parse rate (>= 5 top-level clauses) | **45.0%** | 31.0% |
| Mean top-level nodes | **7.6** | 4.8 |

## Interpretation

### What the keyword classifier is good at

It is genuinely reliable on clauses that are *named by a distinctive phrase*
appearing in the clause body:

* `entire_agreement` (F1 77.3%), `insurance` (74.1%), `notices` (71.8%),
  `severability` (67.7%), `indemnification` (66.7%), `assignment` (66.1%).
  These clauses say "entire agreement", "insure", "notices shall be in
  writing", "invalid or unenforceable", "indemnify and hold harmless", "assign".
  The phrase *is* the clause.
* Precision on the top-scoring classes is high where it matters:
  `severability` 93.8%, `entire_agreement` 91.7%, `insurance` 84.3`%`,
  `indemnification` 79.7%, `boilerplate` 79.9%. When it fires on these, believe
  it.

### What it is bad at, and why

* **Clauses defined by legal *function* rather than vocabulary.**
  `representations` gets 10.8% recall: 192 of its provisions come back
  `unknown`. The signal list looks for "represents" / "represents and
  warrants", which is the *lead-in* to a representation. The body of a
  representation — "the Company is duly organized and validly existing under
  the laws of Delaware" — contains none of those words. The classifier detects
  the framing phrase, not the concept. `warranties` is worse still (F1 2.3%),
  for the same reason compounded by "represents and warrants" being scored to
  `representations`.
* **Signals that are too general.** `covenants` includes the keyword
  `"shall not"`, which appears across nearly every clause type in commercial
  drafting; measured precision is 8.1%. `governing_law` includes
  `"jurisdiction"` and `"applicable law"`, so it swallows forum-selection and
  arbitration clauses: 80.4% recall against 42.8% precision, and it is the most
  frequent single wrong prediction in the corpus (`dispute_resolution` ->
  `governing_law` 80 times, plus 67 from `representations` and 51 from
  `covenants`). Correspondingly `dispute_resolution` itself only reaches 7.6%
  recall — its clauses were already taken.
* **Silence, not error, is the largest single failure mode.** `unknown` is the
  prediction in four of the ten most common confusions, accounting for roughly
  a fifth of all errors. On provisions outside its keyword vocabulary the
  classifier declines rather than guessing, which is the right direction to
  fail, but it means clause metadata is simply absent for a large slice of
  real contract text.

### Is the confidence score usable?

Partly. Confidence and correctness correlate positively (Spearman 0.375), and
the extremes behave: below 0.1, accuracy is 0.7%; at 1.0, 84.8%. As a
**filter** — "discard anything under 0.5" — it works.

As a **probability** it does not. It is systematically under-confident (the
0.2–0.3 bin is 49% accurate; the 0.5–0.6 bin is 53%), so it cannot be read as
"the chance this label is right". And the curve is not monotonic: the
[0.3, 0.4) bin is *half* as accurate as the [0.2, 0.3) bin below it (24.7% vs
49.2%), across 578 and 1,202 items respectively — far too many for that
inversion to be sampling noise. Do not threshold anywhere near 0.3.

The documented definition ("a saturation-scaled margin, not a probability") is
honest about this. The number to publish is the rank correlation, not a
calibration claim.

### What containment does and does not show

**It shows** that LexiChunk's boundaries are better placed than a character
splitter's. At matched mean chunk length (~310–380 chars), LexiChunk contains
81.2% of gold spans against RecursiveCharacterTextSplitter's 72.1% — a 9.1pp
gap with non-overlapping confidence intervals. Fragmentation tells the same
story: 24.4% of gold spans are split across two `rcts-512` chunks versus 13.6%
for the length-matched LexiChunk. That is a real, defensible win, though
narrower than the 11.6pp gap measured against 0.8.0b1.

**It does not show** that LexiChunk is the best chunker here, and the headline
98.3% number is largely an artefact.

* *The length confound is severe.* `max_chunk_size` is in **tokens**, so
  `lexichunk-512tok` produces chunks averaging 993 characters — 2.6x
  `rcts-512`. A length-matched fixed grid alone would score 76.4% on those
  spans. Comparing "LexiChunk at 512" with "RCTS at 512" is comparing 993
  characters with 382 and is not a meaningful comparison.
* *Lift ranks the strategies differently from raw containment.* By raw
  containment, `lexichunk-1024tok` (99.0%) beats `lexichunk-128tok` (81.2%). By
  lift over a length-matched grid, the ordering reverses: +18.3pp against
  +37.5pp. The 1024-token configuration is mostly winning by being long.
* *Overlapping windows game the metric outright.* `sentence-window-3` ties for
  the best raw containment (98.9%) with chunks averaging 647 characters, purely
  because overlapping windows give every span multiple chances to land inside
  one. Its fragmentation column gives it away: only 54.7% of spans sit in a
  single chunk, against 98.3% for `lexichunk-512tok`. Containment and
  fragmentation must be read together, and neither alone is a retrieval result.
* *Containment is necessary, not sufficient.* A chunk that contains a gold span
  can still fail to retrieve it — that depends on embeddings, chunk length, and
  what else is in the chunk. Containment is a ceiling on retrieval quality, not
  a measurement of it.

### Structure recall improved under `us`, but is still the weak point

**This run is against LexiChunk 0.9.0, which fixed one bug this section
originally reported against 0.8.0b1** (issue 1 in
[`results/external/lexichunk_issues.md`](../results/external/lexichunk_issues.md),
a snapshot of the original 0.8.0b1 finding and not itself rewritten): the `us`
jurisdiction profile required a literal `Section N` / `ARTICLE N` marker and
missed the bare decimal heading style (`1. Definitions.` / `1.1 ...`) that
dominates US commercial drafting.

To measure the fix's actual effect, this document's `us`/`uk` figures for
0.8.0b1 were **re-run fresh against the current harness** rather than quoted
from the original (older) `lexichunk_issues.md` snapshot, because the harness's
own CUAD eval code has changed since that snapshot was written (see the
sampling/latency note below) — comparing a fresh 0.9.0 run against a
never-re-run 0.8.0b1 number would confound the LexiChunk fix with unrelated
harness changes. On that fresh, same-harness baseline: `us` recovered five or
more top-level clauses in **13.0%** of contracts (mean 2.9 top-level nodes),
against **42.0%** for `uk` on the *same* filings (mean 8.7) — the profile
named for the jurisdiction was the worse choice for it, and both figures are
close to but not identical to the original snapshot's 14.0%/31.0% (a
difference in the harness, not in LexiChunk 0.8.0b1 itself, which was not
recompiled between the two measurements).

0.9.0 now recognises bare-decimal headings under `us` too: on the same 100
CUAD contracts, `us` recovers five or more top-level clauses in **45.0%** of
contracts — a 3.5x improvement over the fresh 13.0% baseline — and now
*exceeds* `uk`. Only 31% fall back to flat text entirely, down from 46%. But
**the `uk` profile itself got *worse* on these same US filings between the two
builds** — 42.0% (0.8.0b1) down to 31.0% (0.9.0) — which this document does
not have an explanation for; it is filed as a new, second finding rather than
folded into the `us` story, and is worth a follow-up before treating "try `uk`
as a fallback" (below) as safe advice against 0.9.0 specifically. A
structure-aware chunker that cannot find the structure has quietly degraded to
a slow fixed-size splitter on the remaining fraction, and it will not tell you
that it has; the `us` improvement is real progress, not a closed issue.

**Cost did not meaningfully change.** 0.9.0 averages 0.067–0.077 s per
contract on these strategies; a fresh, same-harness 0.8.0b1 run averages
0.071–0.105 s — comparable, not the ~10x gap an earlier draft of this document
reported from comparing 0.9.0's fresh timing against a stale 0.66–0.71 s
figure carried over from the original, much older harness snapshot. That
comparison was wrong and has been corrected here. The historical 100x-slower,
20.6s-worst-case comparison against `RecursiveCharacterTextSplitter` in
`lexichunk_issues.md` is from that same older snapshot and was not re-run for
`RecursiveCharacterTextSplitter` in this pass either; treat it as stale until
both sides are re-measured together.

Mean chunk length also moved for `lexichunk-*` alone between the two builds —
0.8.0b1's `lexichunk-512tok` averages 2,106 characters against 0.9.0's 993 —
because on contracts where 0.8.0b1 finds no usable structure it still emits
chunks sized up against the raw `max_chunk_size` token ceiling, while 0.9.0
increasingly stops at real clause boundaries below that ceiling. See the
containment section above for how this feeds into the raw containment number.

### Robustness

Zero exceptions across 150 real SEC contracts, and zero across the 100-contract
evaluation sample. LexiChunk does not crash on messy real-world input. The
issues found were all wrong or surprising *output* — most importantly that
`chunk.content` is not `text[chunk.char_start:chunk.char_end]` for roughly half
of all chunks, because the parent heading is prepended to `content` without
adjusting the offsets. Full list, with reproducers, in
[`results/external/lexichunk_issues.md`](../results/external/lexichunk_issues.md).

## Threats to validity

* **The mapping is a judgement call.** 33 of LEDGAR's 100 labels are out of
  scope, and the 17-label `representations` cluster in particular collapses
  narrow SEC rep-topics (Capitalization, Solvency, Subsidiaries, Brokers) onto
  one class. That cluster is where the classifier scores worst, so a reader who
  disagrees with the mapping should discount the `representations` row
  specifically. The mapping went through a second legal review, which changed
  four entries; its residual-risk notes are in the review record.
* **LEDGAR provisions are isolated.** LexiChunk classifies chunks *in a
  document*, with a hierarchy path and a position. LEDGAR gives neither. The
  39.6% figure is therefore a measurement of the keyword signals alone, and is
  a **lower bound** on in-context performance: `classify_all` supplies a
  relative position and `hierarchy_path`, both of which add signal the LEDGAR
  setup withholds.
* **Domain shift.** LEDGAR is SEC credit/M&A/employment agreements. LexiChunk's
  taxonomy has a general-commercial and SaaS flavour (`acceptable_use`,
  `data_protection`, `account_security` have no LEDGAR counterpart at all).
  Accuracy on a SaaS-agreement corpus could differ substantially in either
  direction.
* **CUAD spans are answers, not clauses.** A CUAD gold span is the text an
  annotator highlighted to answer a question, which is often a sentence
  fragment inside a clause rather than the clause itself. Containment measures
  "does a chunk boundary cut this answer", which is the right question for
  retrieval, but it is not the same as "does a chunk equal a clause".
* **Sampling.** 100 of 510 CUAD contracts and 5,000 of 10,000 LEDGAR test rows,
  both seeded at 0. Confidence intervals reflect that sampling; they do not
  reflect the choice of dataset or mapping.
* **One version for the headline numbers; a fresh, same-harness comparison run
  for the CUAD structure-recall claim only.** The headline LEDGAR and CUAD
  tables above are LexiChunk `0.9.0` (commit `0346a12`). The CUAD
  structure-recall section additionally reports a **freshly re-run** 0.8.0b1
  (`32078cd`) comparison — not the numbers this document originally
  published — because the harness's own CUAD eval code changed since that
  original run, so quoting its old numbers against a fresh 0.9.0 run would
  have confounded the LexiChunk fix with unrelated harness drift.
* **LEDGAR cannot be run against the 0.8.0b1 baseline at all.**
  `evals/ledgar_label_map.yaml` maps onto LexiChunk's current 31-value
  `ClauseType` enum (including e.g. `insurance`); 0.8.0b1 has a 27-value enum
  that predates several of those labels, so `python -m legal_rag_eval.evals
  ledgar` against a `.venv` pointed at 0.8.0b1 fails outright with
  `ValueError: ... maps to 'insurance', which is not a LexiChunk ClauseType`.
  The LEDGAR numbers above are therefore not a "both builds, unchanged"
  result — they are the only version they can be computed against. The
  historical `RecursiveCharacterTextSplitter` cost comparison in the
  structure-recall section was not re-run and is labelled stale there.

## Paragraph for LexiChunk's README

The following is offered for a "Measured accuracy" section in LexiChunk's
README. Every number is from this document.

> ### Measured accuracy
>
> LexiChunk is evaluated against two public legal datasets it was not tuned on.
>
> **Clause classification (LEDGAR).** On 3,955 provisions from the LEDGAR test
> split (LexGLUE), with LEDGAR's 100 classes mapped onto LexiChunk's clause
> types by an explicit reviewed mapping, the keyword classifier scores **39.6%
> accuracy (95% CI 38.0–41.2) and 42.5% macro-F1** — against 21.6% for a
> majority-class baseline and 90.8% for a TF-IDF + logistic regression model
> trained on LEDGAR's train split. It is strong on clauses named by a
> distinctive phrase (`entire_agreement` F1 77.3%, `insurance` 74.1%, `notices`
> 71.8%, `severability` 67.7%, `indemnification` 66.7%) and weak on clauses
> defined by legal function rather than vocabulary (`representations` 17.8%,
> `warranties` 2.3%). `classification_confidence` correlates with correctness
> (Spearman 0.375) and is usable as a filter, but it is not a probability and
> the curve is not monotonic.
>
> **Boundary quality (CUAD).** On 100 CUAD contracts (2,458 expert-annotated
> answer spans), comparing chunkers at matched mean chunk length (~310–380
> chars), **81.2% of gold spans fall wholly inside a single LexiChunk chunk
> versus 72.1% for `RecursiveCharacterTextSplitter`**, and LexiChunk splits
> roughly half as many spans across two chunks (13.6% vs 24.4%). Note that raw
> containment rewards longer chunks: `max_chunk_size` is measured in tokens, so
> `max_chunk_size=512` yields ~1,000-character chunks and a correspondingly
> higher 98.3% containment.
>
> **Robustness.** Zero exceptions across 150 real SEC contract filings.
>
> **Known limitation.** Structure detection on real US filings is improved in
> 0.9.0 under `jurisdiction="us"` but still partial: 45% of CUAD contracts
> yield five or more top-level clauses (up from 13% on a fresh 0.8.0b1 run),
> and 31% still fall back to flat text. The `uk` profile moved the other way
> on the same documents (31% now vs 42% on 0.8.0b1) and is no longer clearly a
> better fallback than it was — if your contracts are not parsing well under
> `us`, try `uk` too, but verify on your own documents rather than assuming
> it still wins by the old margin.
>
> Full methodology, per-class numbers and reproduction commands:
> `docs/external_evals.md` (this file).


## Reproduction

```bash
pip install -e ".[evals]"

# Cache datasets under the repo (gitignored)
export HF_HOME="$PWD/.cache/hf"

python -m legal_rag_eval.evals ledgar --sample 5000 --seed 0
python -m legal_rag_eval.evals cuad   --contracts 100 --seed 0
```

Each command writes `results/external/<name>.json` (every number in this
document, machine-readable) and `results/external/<name>.md` (the tables).
`--smoke` runs the identical pipeline on a handful of synthetic provisions and
contracts with no download at all; its numbers are meaningless and are labelled
as such in the output.

## Methodology

### LEDGAR

LEDGAR is ~80k contract provisions scraped from EDGAR filings, each labelled
with one of 100 clause-type classes. LexiChunk emits one of 31 `ClauseType`
values. The two taxonomies are bridged by an explicit, hand-written mapping in
[`evals/ledgar_label_map.yaml`](../evals/ledgar_label_map.yaml), reviewed by a
second reviewer for legal sense. Each entry carries a one-line rationale.

Three rules keep the mapping from flattering the classifier:

* **Nothing maps to `unknown`.** `unknown` is what LexiChunk returns when no
  keyword fires. Using it as a gold label would score the classifier correct
  for giving up.
* **Nothing maps to `preamble` or `recitals`.** LexiChunk assigns those from a
  chunk's position in a document. A LEDGAR provision arrives in isolation and
  has no position, so those classes are unpredictable by construction.
* **LEDGAR labels with no honest LexiChunk counterpart are marked
  out of scope** and excluded from scoring rather than forced onto the nearest
  class. Coverage is reported as a first-class number, because a mapping that
  quietly discards the hard classes would inflate accuracy.

Provisions are classified with
`ClauseTypeClassifier.classify_detailed(text, relative_position=0.5)`. LexiChunk
only applies its end-of-document bonus above `relative_position > 0.75`, so 0.5
is a neutral choice: it is identical in effect to 0.0 and does not hand the
classifier positional information the task does not provide.

Two baselines give the number context:

* **Majority class** — always predict the most frequent mapped class. This is
  the floor; anything at or below it has learned nothing.
* **TF-IDF + logistic regression** — word/bigram TF-IDF into multinomial
  logistic regression, fitted on LEDGAR's *train* split and evaluated on the
  same test items. It is a weak, cheap, *supervised* model, and it marks roughly
  what "fit the task from data" buys over hand-written keywords.

### CUAD

CUAD is 510 real commercial contracts (SEC filings) with expert answer spans for
41 clause categories. Rows sharing a `title` are one contract; their `context`
is byte-identical across rows, verified. Each answer contributes a gold span,
kept only when `context[start:start+len(answer)] == answer` exactly — a span
whose offset does not reproduce its own text would make containment
meaningless.

The measure is **span containment**: for each gold span, does at least one chunk
*wholly* contain it? A chunker that cuts through the middle of an answer has
destroyed it for retrieval, no matter how good its embeddings are.

Chunk spans must index the original document exactly. LexiChunk reports
`char_start`/`char_end` directly and they were verified to reproduce chunk text
character-for-character. LangChain's splitters expose no offsets and strip
whitespace, so their output is re-located in the source with a forward-only
cursor (which keeps repeated boilerplate from matching back at the top of the
file).

**The length confound.** Containment rewards long chunks for free: a splitter
that emits the whole document as one chunk scores 100%. Two corrections are
reported alongside the raw rate.

* *Mean chunk length* for every strategy, so the reader can see who is winning
  by being longer.
* *Length-matched grid expectation*: the containment a naive fixed-stride
  splitter with the **same mean chunk length** would achieve on the same spans.
  For a span of length `s` placed uniformly against boundaries every `L`
  characters, it survives with probability `max(0, 1 - s/L)`; averaging that
  over the observed spans gives the expectation. Subtracting it gives a **lift**
  that credits a strategy only for boundaries that beat cutting blindly.

Note also that LexiChunk's `max_chunk_size` is measured in **tokens** (four
characters per token by default), not characters. `max_chunk_size=512` therefore
produces chunks around 2000 characters — four times `RecursiveCharacterTextSplitter(chunk_size=512)`.
Comparing those two directly is not a fair fight, so a length-matched
`lexichunk-128tok` configuration (~512 characters) is included.

**Structure recall** is measured separately: the fraction of contracts where
`parse_structure` recovers at least N top-level clauses rather than falling back
to flat text. A structure-aware chunker that cannot find the structure is just a
slower fixed-size splitter.

**Failures.** CUAD text is real: OCR noise, tables, exhibit headers, page
furniture. Every chunking call is wrapped, and any exception is recorded as a
LexiChunk bug with a minimal reproducer in
[`results/external/lexichunk_issues.md`](../results/external/lexichunk_issues.md).

### Confidence intervals

Every interval is a 2000-resample percentile bootstrap at 95%, resampling over
*items* — provisions for LEDGAR, gold spans for CUAD containment — because that
is the unit of independence. The RNG is seeded, so intervals are reproducible
byte for byte.

---

*Generated by `legal_rag_eval.evals` (`--seed 0`) against LexiChunk 0.9.0, commit
`0346a12`. The CUAD structure-recall section's 0.8.0b1 (`32078cd`) comparison
numbers come from a fresh same-harness re-run kept only in this document's
prose, not committed under `results/external/`. Raw output for the headline
0.9.0 tables:
[`results/external/ledgar.json`](../results/external/ledgar.json),
[`results/external/cuad.json`](../results/external/cuad.json).*
