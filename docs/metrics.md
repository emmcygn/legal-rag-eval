# Metrics reference

This document describes what the harness actually computes today, not what any docstring
aspires to. Every formula below was checked against the implementation in
`src/legal_rag_eval/metrics/gold.py`, `src/legal_rag_eval/metrics/retrieval.py`,
`src/legal_rag_eval/metrics/statistical.py`, `src/legal_rag_eval/metrics/structural.py`,
`src/legal_rag_eval/gold.py` and `src/legal_rag_eval/models.py`. Where a docstring and its code
disagreed, this document follows the code and the disagreement is called out explicitly.

If you take one thing from this document: **read `localization_rate` and
`localization_coverage` before any other number in a results table.** Every structural
metric here is computed only over chunks that were successfully matched back to the
document; a strategy whose chunks cannot be matched is only partially measured, and the
partial measurement can look better than it is.

## 0. The two ground truths, and which metric uses which

The harness carries **two independent ground truths**, for two different questions. They
are never mixed, and no metric is defined against a chunker's own output.

| | `benchmark` (anchored evidence) | `benchmark-structural` / `benchmark-embed` (gold) |
|---|---|---|
| Question | Does retrieval put the *answer text* in the context budget? | Do chunk boundaries respect the document's *clause structure*? |
| Ground truth | `src/legal_rag_eval/data/synthetic_contracts_v1.json` | `gold/<document_id>.json` |
| Unit | evidence spans (char offsets) for 15 queries | 365 clause spans, 90 defined terms, 171 cross-references |
| Corpus | 3 synthetic AI-authored contracts | 5 fixture documents (98,365 chars) |
| Provenance | authored directly as spans, never derived from a chunker; SHA-256 pinned to the document text | regex-seeded from each document's own numbering, then hand-corrected with every change logged in `gold/CHANGES.md`; SHA-256 pinned |
| Ranker | lexical cosine term-frequency, offline and deterministic | MiniLM / BGE embeddings over a FAISS index |
| Metrics | source-span evidence recall/precision, abstention accuracy | sections 2-4 below |

### Anchored-evidence metrics (`legal_rag_eval.evidence.benchmark`)

Computed once in `score_selected_spans()` and consumed unchanged by the CLI, the JSON
report and the dashboard viewer.

- **Source-span evidence recall** (answerable queries only): unioned evidence characters
  covered by the selected source spans, divided by all unioned evidence characters.
  Overlapping gold spans are merged first, so annotating the same sentence twice cannot
  inflate the denominator.
- **Source-span evidence precision** (answerable queries only): unioned evidence characters
  covered, divided by *all* selected context characters. Duplicate and overlapping selected
  spans remain fully charged, so a strategy cannot buy recall with unlimited context.
- **Abstention accuracy** (unanswerable queries only): 1 when retrieval selects no context
  at all, 0 otherwise. This records context emission, not the correctness of any legal
  answer — nothing here reads an answer, because nothing here generates one.
- Evidence **grade** is stored in the dataset but is deliberately *not* used as a weight.

Every strategy is given the same `context_budget_tokens` (default 128, measured in
whitespace tokens rather than a model tokenizer), so a larger chunk cannot win merely by
containing more. That equalises the *selection* budget; it is not a control for chunk
length itself, which is what `rcts_1024` in section 2 exists for.

## 1. How ground truth is made

Gold annotations live in `gold/<document_id>.json`, one file per fixture document, and are
described in full in `gold/SCHEMA.md`. The short version:

- They are seeded from each document's **own** numbering by `scripts/build_gold.py` (a
  regex seeder), then corrected by hand against the document text, with corrections logged
  in `gold/CHANGES.md`. No LexiChunk output is consulted at any point in producing them —
  they are independent of every chunker under test, LexiChunk included.
- Every offset (`char_start`/`char_end`) indexes into the **sanitised** text:
  `legal_rag_eval.gold.sanitize()` strips `U+FEFF` BOMs, normalises `\r\n`/`\r` to `\n`, then
  applies Unicode NFC normalisation, in that order. This mirrors
  `LegalChunker._sanitize_input` exactly and is the single implementation used by the
  seeder, the loader, the validator, and every metric in this document.
- Each annotation file is pinned to an exact fixture revision by `text_sha256` (a SHA-256
  of the sanitised text). `legal_rag_eval.gold.verify_document()` is called at the start of
  `compute_gold_structural_metrics` and raises `GoldError` if the fixture no longer hashes
  to that value, so a fixture edit can never silently invalidate the gold spans it is
  compared against.

### What is deliberately excluded

Stated plainly, because it bounds every recall number that follows:

- **Anaphoric cross-references are not annotated.** "The foregoing", "this Section" and
  similar back-references are out of scope. Cross-reference recall/precision are measured
  against **explicit** references only (a named target such as "clause 12.3").
- **Clause depth is capped at level 3.** Deeper lettered romanettes are folded into their
  parent clause's span rather than annotated as their own level-4+ clause.
- **Undefined terms are not annotated.** `defined_terms` covers only terms the document
  itself marks as defined (quoted-and-defined, or listed in a definitions clause); a term
  used but never defined in the document has no gold entry and cannot be scored.
- **Front matter is not partitioned.** Title blocks, party recitals and similar text that
  belongs to no numbered clause are not covered by any `clauses` span — gaps between spans
  are legal, overlaps are not.

### Clause spans vs subtree spans vs leaf clauses

A `clauses` entry's own span (`char_start`/`char_end`) runs from the clause's own
number/heading up to the first character of the next clause at *any* level, so it
**excludes descendant clauses**. `GoldAnnotation.subtree_span(clause)`
(`src/legal_rag_eval/gold.py`) extends that to the end of the clause's last descendant — "the
whole of clause 7, including 7.1, 7.2, ...". `GoldAnnotation.leaf_clauses` is every clause
with no children. Metrics that need an atomic unit (fragmentation) use leaf clauses;
metrics that need "does this chunk swallow a whole top-level clause" (over-merge) use
subtree spans.

## 2. How a chunk is located in the document

Chunk spans are **never** taken from a chunker's self-reported offsets. Instead
`legal_rag_eval.gold.locate_chunks()` matches every chunk's raw text against the sanitised
document, uniformly across strategies, for two reasons stated in the module docstring:
some strategies (LexiChunk's contextual variant, in particular) prepend synthesised text —
a heading breadcrumb — that is not verbatim in the source, and a chunker's own offsets have
at least once been wrong for a LexiChunk build. Locating chunks by text, the same way for
every strategy, means a bug in one chunker's bookkeeping cannot corrupt its own scores.

For each chunk, `locate_chunks` (`src/legal_rag_eval/gold.py::_locate_one`) tries, in order:

1. **Exact match** — the chunk's full text found verbatim in the document (searching from
   a cursor that advances past the previous chunk, falling back to a search from the start
   of the document if that fails).
2. **Longest verbatim suffix** — binary search for the smallest `i` such that `chunk[i:]`
   occurs somewhere in the document (`_smallest_suffix_index`). This is what strips a
   synthesised heading a chunker prepended: the prefix is garbage, but the suffix — the
   chunk's actual document text — is still verbatim. Rejected if the matched suffix covers
   less than 50% of the chunk's own characters.
3. **Longest verbatim prefix** — the symmetric case (`_largest_prefix_index`), for text
   appended rather than prepended. Same 50%-coverage floor.
4. **Union of both** — if both a suffix and a prefix match are found, and their located
   spans in the document are within `2 * len(chunk)` characters of each other
   (`_span_distance`), the two are merged into one span (`min` of the starts, `max` of the
   ends). If they land further apart than that, they cannot both be describing the same
   occurrence, so the harness trusts whichever covers more of the chunk's own text.

A chunk that matches none of the above — including any chunk that is entirely whitespace —
is **unlocated**: `ChunkSpan` is `None`, and `Chunk.located` (`src/legal_rag_eval/models.py`) is
`False`. Unlocated chunks are excluded from every span-based numerator and denominator in
`metrics/gold.py` and `metrics/retrieval.py`. They are **reported**, via
`localization_rate`, and never silently counted as a scoring failure against some other
metric.

```
localization_rate      = located_chunks / total_chunks                       (0.0 if total_chunks == 0)
localization_coverage  = sum(covered_chars for located chunks) / sum(len(chunk_text) for all chunks)
                                                                              (0.0 if total chars == 0)
```

`covered_chars` is the number of the chunk's own characters accounted for by the match (all
of them for an exact match; the matched suffix/prefix length, or the larger of the two, for
a suffix/prefix/union match). **Read both numbers before any other structural metric.** A
strategy at 60% localisation rate is having 40% of its output excluded from
`clause_fragmentation_rate`, `top_level_over_merge_rate`, heading recall/precision,
definition attachment recall and cross-reference recall/precision alike — every one of
those numbers is then computed over a non-random subset of the strategy's actual output,
and there is no guarantee the excluded 40% resembles the included 60%.

## 3. Reported structural metrics

All of these are computed by `compute_gold_structural_metrics`
(`src/legal_rag_eval/metrics/gold.py`) into a `GoldStructuralMetrics`
(`src/legal_rag_eval/models.py`). Fields that can be `None` are documented as such below; `None`
means *not applicable to this strategy* and must never be rendered as `0` — a strategy that
exposes no heading metadata has not scored zero on heading recall, it has not been tested on
it at all.

> **Clause fragmentation rate** (`clause_fragmentation_rate`)
> **Formula.** Over eligible leaf clauses — `gold.leaf_clauses` at least 20 characters long
> (`clause.char_end - clause.char_start >= 20`) — a clause is *fragmented* when
> ```
> max(overlap_chars(chunk_span, clause_span) for chunk in located_chunks) / clause_len < 0.80
> ```
> Returns `fragmented / eligible`, or `0.0` when there are no eligible leaf clauses.
> **Range.** `[0.0, 1.0]`. **Direction.** lower is better.
> **What it does not measure.** Over-merging. A single chunk spanning the entire document
> contains 100% of every leaf clause and scores a perfect `0.0` here.
> **How to game it.** Emit one chunk per document. This is caught only by
> `top_level_over_merge_rate`, below — the two must be read together.

> **Top-level over-merge rate** (`top_level_over_merge_rate`)
> **Formula.** Over located chunks, a chunk *over-merges* when its span overlaps the
> **subtree spans** of more than one top-level gold clause by at least `min_overlap_chars`
> (default `50`) characters each:
> ```
> hits = count(subtree_span for subtree_span in top_level_subtree_spans
>              if overlap_chars(chunk_span, subtree_span) >= 50)
> over_merging when hits > 1
> ```
> Returns `over_merging_chunks / located_chunks`, or `0.0` when there are no located chunks.
> **Range.** `[0.0, 1.0]`. **Direction.** lower is better.
> **What it does not measure.** Fragmentation — this is the metric the superseded harness
> (section 6) lacked entirely, and it is the direct counterweight to
> `clause_fragmentation_rate`: a whole-document chunk scores `0.0` fragmentation but
> `1.0` (or close to it) over-merge.
> **How to game it.** Emit one chunk per top-level clause with hard boundaries at every
> top-level number, ignoring sub-clause structure entirely; over-merge cannot fire below
> two top-level clauses per chunk. Because `0.0` located chunks also returns `0.0`, a
> strategy whose chunks all fail to locate scores a deceptively clean `0.0` here — check
> `localization_rate` first.

> **Sub-clause grouping rate** (`sub_clause_grouping_rate`)
> **Formula.** Over located chunks, a chunk counts when it fully contains (overlap
> `>= 0.95` of the clause's own character count) two or more **sibling** leaf clauses (same
> `parent`). Returns `grouping_chunks / located_chunks`, or `0.0` when there are no located
> chunks.
> **Range.** `[0.0, 1.0]`. **Direction.** informational — no direction. A high value is
> neither good nor bad by itself; it says a strategy tends to keep sibling sub-clauses
> together, which trades off directly against `top_level_over_merge_rate` (group enough
> siblings and eventually you span more than one top-level clause). Report it alongside
> that metric, not instead of it.
> **What it does not measure.** Whether grouping siblings is *appropriate* for a given
> clause — that depends on document semantics this harness does not model.
> **How to game it.** Not really gameable in isolation since it carries no direction; a
> strategy cannot "cheat" a metric that isn't scored pass/fail.

> **Heading recall and precision** (`heading_metrics`)
> Only scored for strategies that expose `metadata["section_hierarchy"]` on at least one
> chunk — this is one of two metrics explicitly about what a strategy *chooses to claim*
> (see the module docstring's rule 2), so a strategy exposing nothing returns
> `(None, None, len(gold.clauses))`, not `(0, 0, ...)`.
> **Formula.** For each located chunk with that metadata key, `claimed_identifier` walks the
> whole `">"`-separated breadcrumb rather than reading its last component: each component's
> numbering token either replaces the accumulated identifier (a new numbered clause) or, for
> a lettered romanette, extends it. So
> `"2 - Services > 2.5 - The Client shall: > (a) - provide..."` claims `2.5(a)`, where the
> last component alone would give `"(a)"`, which names nothing. The composed identifier is
> normalised (`normalize_identifier`) and resolved against gold identifiers — both the
> document's own convention (`section_7.2`) and the bare numbering suffix (`7.2`), so a
> naming convention is not scored as a wrong claim. A claim is **correct** when it resolves
> to a gold clause AND that clause's span overlaps the chunk's span by at least 1 character.
> Chunks whose span touches no annotated clause at all (a title block or recitals) make no
> claim and enter neither numerator nor denominator.
>
> **Only LexiChunk is scored on this today.** No baseline strategy emits
> `section_hierarchy`, so every one of them reads `n/a`. That makes this a capability
> measurement — does the chunker surface a hierarchy, and is what it surfaces right — and
> **not** a ranking of the field. A metric only one entrant can be scored on cannot rank the
> field, and neither the README nor the CLI uses it to.
> ```
> precision = correct_claims / total_claims                (1.0 if no located chunk makes a claim)
> recall    = |distinct correctly-claimed gold clauses| / |gold clauses overlapped by >= 50 chars
>                                                             by some located chunk|   (1.0 if denom is 0)
> ```
> The recall denominator (`n_gold_headings`) counts only clauses some located chunk actually
> reached, so a strategy is not penalised for headings on clauses nobody's chunks touched.
> **Range.** Each of recall/precision is `None` or `[0.0, 1.0]`. **Direction.** higher is
> better for both.
> **What it does not measure.** Heading wording or quality — only whether the claimed
> identifier resolves to the right gold clause.
> **How to game it.** Precision is trivially `1.0` for a strategy that claims nothing (or
> claims only identifiers you are certain are right); recall rewards claiming identifiers
> generously since a wrong claim only fails to add to the numerator, it does not subtract.

> **Definition attachment recall** (`definition_attachment_recall`)
> **Formula.** For each located chunk and each gold defined term, a **use** is counted when
> the term string occurs (case-insensitive, word-boundary regex) in
> `sanitize(document.text)[chunk.char_start:chunk.char_end]` — sliced from the document
> text, never from `chunk.text`, so a strategy cannot manufacture a use by injecting text —
> at an offset outside the term's own `definition_span`. At most one use per (chunk, term).
> A use is **attached** when
> ```
> overlap_chars(chunk_span, definition_span) / definition_len >= 0.80
> ```
> OR the chunk's `metadata["defined_terms_context"]` or `metadata["definitions"]` (a
> `dict[str, str]`) carries the actual definition text for that term — a bare list of term
> names does not count.
> ```
> recall = attached_uses / total_uses          (1.0 if total_uses == 0, vacuous)
> ```
> **Range.** `[0.0, 1.0]`. **Direction.** higher is better.
> **What it does not measure.** Over-merging. A strategy emitting one enormous chunk per
> document trivially attaches every definition to every use.
> **How to game it.** Same as fragmentation: make chunks larger. Read this next to
> `top_level_over_merge_rate`, never in isolation, and next to mean chunk length.

> **Cross-reference target recall and precision** (`cross_reference_metrics`)
> Only scored for strategies that emit `metadata["cross_references"]` (a list) on at least
> one chunk — the second of the two "claims about what the strategy exposes" metrics. If no
> chunk emits that key, returns `(None, None, n_gold_cross_refs)`, where
> `n_gold_cross_refs` counts gold references with a non-null `target_identifier` (internal
> references only; external references such as a statute citation have no in-document
> target and are excluded from both this count and every numerator/denominator).
> **Formula.** For each located chunk, the gold references *in scope* are those whose
> `char_start` falls inside the chunk's span (`chunk_start <= ref.char_start < chunk_end`).
> ```
> recall    = |recalled gold refs| / n_gold_cross_refs                (1.0 if n_gold_cross_refs == 0)
> precision = |emitted entries matching a gold ref in scope of that chunk| / total_emitted_entries
>                                                                       (1.0 if nothing emitted)
> ```
> A gold reference is **recalled** when the chunk in whose scope it falls emits an entry
> `{"raw_text": ..., "target": ...}` whose normalised `target`
> (`normalize_identifier`) equals the reference's normalised `target_identifier`.
>
> **Only LexiChunk is scored on this today**, for the same reason as heading recall: no
> baseline emits cross-reference metadata, so all of them read `n/a`. Treat it as a
> capability measurement, not a comparative result.
> **Range.** Each of recall/precision is `None` or `[0.0, 1.0]`. **Direction.** higher is
> better for both.
> **What it does not measure.** Reference *kind* (schedule vs. definition vs. clause), or
> whether `raw_text` itself was extracted correctly — only whether the claimed target
> resolves to the right gold clause. A reference whose owning chunk was never located is
> simply not recalled; it does not shrink `n_gold_cross_refs`.
> **How to game it.** Emit an entry for every plausible-looking clause number in scope
> (recall goes up, precision absorbs the cost) or emit only the references you are certain
> of (precision goes up, recall absorbs the cost) — the two must be read as a pair.

> **Chunk-size coefficient of variation** (`chunk_size_cv`)
> **Formula.**
> ```
> mean     = sum(char_count) / n
> variance = sum((char_count - mean) ** 2) / n        # population variance, ddof = 0
> cv       = sqrt(variance) / mean
> ```
> Returns `0.0` for fewer than 2 chunks or when the mean is 0.
> **Range.** `[0.0, inf)`. **Direction.** purely descriptive, not a quality signal.
> **What it does not measure.** Whether boundaries land anywhere sensible. A naive
> fixed-size splitter has a near-zero CV by construction and that says nothing about
> segmentation quality.
> **How to game it.** Not gameable as a target in itself, but a low CV should never be read
> as evidence of good chunking — pair it with the segmentation metrics above.

> **Localisation rate and coverage** (`localization_rate`, `localization_coverage`)
> See section 2 for the full mechanism. Repeated here because they are structural-metrics
> fields on `GoldStructuralMetrics`: `localization_rate = located_chunks / total_chunks`,
> `localization_coverage = located_chars / total_chunk_chars`. **Range.** both `[0.0, 1.0]`.
> **Direction.** higher is better — but read as a **precondition** for trusting every other
> number in this section, not as a quality metric in its own right.
> **What it does not measure.** Whether the located chunks are well-formed — a chunk can be
> perfectly located and still fragment a clause.
> **How to game it.** Not gameable in a way that helps a strategy's other scores: a low
> rate simply shrinks the population every other metric is computed over, and does not
> forgive bad segmentation among the chunks that *did* locate.

### Read these pairs together

- **Fragmentation vs. over-merge.** Fragmentation is trivially minimised by emitting one
  chunk per document — that is exactly the case `top_level_over_merge_rate` exists to
  catch. Neither number means anything about "good chunking" alone.
- **Definition attachment recall also rewards enormous chunks**, for the same reason as
  fragmentation: bigger chunks are more likely to contain both a use and its definition.
  Read it next to `top_level_over_merge_rate` and next to mean chunk length.

## 4. Retrieval metrics

All retrieval metrics share **one** relevance definition,
`matched_sections()` (`src/legal_rag_eval/metrics/retrieval.py`), also used by
`legal_rag_eval.retrieval.simulator`. This replaces a prior version of the harness that defined
relevance independently in three places — once for P@k/R@k, once in the simulator, and a
third, stricter way inside NDCG's own `elif` chain — so a chunk could be graded relevant for
P@5 and simultaneously irrelevant for the NDCG number printed beside it. That is fixed now:
`is_relevant()` and `relevance_grade()` are both thin wrappers over `matched_sections()`, and
P@k, R@k, MRR and NDCG@10 all call one of those two.

> **Relevance definition** (`matched_sections`)
> **Formula.** A chunk matches a `RelevantSection` when all hold: the chunk is located
> (`char_start`/`char_end` not `None`); `chunk.document_id == section.document_id`; and
> ```
> if section_len < min_overlap_chars:   overlap_chars(chunk, section) >= 0.5 * section_len
> else:                                 overlap_chars(chunk, section) >= min_overlap_chars
> ```
> where `min_overlap_chars` defaults to `DEFAULT_MIN_OVERLAP_CHARS = 100`, mirroring
> `BenchmarkConfig.relevance_min_overlap_chars` (also `100` by default,
> `src/legal_rag_eval/config.py`). The 50%-of-own-length fallback exists so a short clause (e.g.
> a 40-character definition) is not unmatchable purely because it is shorter than the
> absolute threshold. `is_relevant()` is the boolean view; `relevance_grade()` returns the
> maximum `RelevanceGrade` (`EXACT=3, SAME_SECTION=2, RELATED=1, IRRELEVANT=0`) among the
> sections matched, or `IRRELEVANT` if none.
> **Range.** boolean / one of four grades. **Direction.** n/a (definitional).
> **What it does not measure.** Whether the chunk's *text* is actually about the section's
> subject — only whether the spans overlap enough. This replaces an older text-similarity
> judge that was monotone in chunk length; span overlap is less severely biased toward
> larger chunks, but still biased (see the residual-bias note below).
> **How to game it.** Make chunks large enough to straddle a relevant clause's boundary
> while being mostly about something else — they still count as a match.

> **Precision@k** (`precision_at_k`)
> **Formula.** `P@k = |relevant chunks in top-k| / min(k, |chunks actually at rank <= k|)`.
> The denominator is the number of chunks **actually present** in the top-k, not a fixed
> `k` — a strategy returning only 3 chunks when asked for 10 is scored out of 3, not
> penalised by a denominator of 10 (which would conflate "retrieved nothing relevant" with
> "returned fewer chunks than requested"). Returns `0.0` for `k <= 0` or no hits at
> rank `<= k`.
> **Range.** `[0.0, 1.0]`. **Direction.** higher is better.
> **What it does not measure.** Grade — a RELATED match counts identically to an EXACT
> match; use NDCG@10 when grade matters.
> **How to game it.** Return fewer results than asked for but make every one of them a
> marginal, just-over-threshold match — the denominator shrinks with you.

> **Recall@k** (`recall_at_k`)
> **Formula.** `R@k = |distinct relevant sections matched by >= 1 chunk in top-k| /
> |relevant_sections|`. The numerator counts distinct **sections**, never chunks, so R@k
> cannot exceed 1 even if several retrieved chunks all match the same section. Returns
> `1.0` vacuously when the query has no annotated relevant sections, `0.0` for `k <= 0`.
> **Range.** `[0.0, 1.0]`, always. **Direction.** higher is better.
> **What it does not measure.** Whether a section was found once or five times over, and
> does not distinguish a marginal match from one where the chunk fully contains the
> section.
> **How to game it.** Return many overlapping/duplicate chunks that between them touch
> every relevant section — R@k cannot tell that from one precise chunk per section.

> **Mean Reciprocal Rank** (`mrr`)
> **Formula.** `1 / rank` of the first hit (by ascending rank) for which `is_relevant()` is
> true; `0.0` if none are relevant.
> **Range.** `[0.0, 1.0]`. **Direction.** higher is better.
> **What it does not measure.** Anything about hits after the first relevant one, or their
> grade.
> **How to game it.** Get exactly one marginal match to rank 1 and ignore everything else
> in the list — MRR cannot see the rest.

> **NDCG@10** (`ndcg_at_k`)
> **Formula.** Grades come from `relevance_grade()` — the same definition `precision_at_k`
> uses via `is_relevant()`, so nothing here can disagree with P@k about what counts as
> relevant (the bug this fixed: NDCG previously used its own, stricter `elif` chain, so a
> chunk P@k counted as relevant could be graded IRRELEVANT by NDCG).
> ```
> DCG@k  = sum_i (2**grade_i - 1) / log2(i + 2)     # i is 0-indexed over the top-k hits, rank order
> IDCG@k = same formula over relevant_sections' own grades, sorted descending, truncated to k
> NDCG@k = min(1.0, DCG@k / IDCG@k)
> ```
> Both `actual_grades` and `ideal_grades` are zero-padded up to length `k` if there are
> fewer than `k` hits or relevant sections. Returns `0.0` for `k <= 0`, no relevant
> sections, or `IDCG@k == 0` (guarded rather than dividing by zero).
> **Range.** nominally `[0.0, 1.0]`, and explicitly clamped there. **Direction.** higher is
> better.
> **The clamp, and why it exists.** If several retrieved chunks in the top-k each
> independently match the *same* single gold section (e.g. two overlapping chunks both
> straddling one clause), each contributes that section's full gain to DCG, while IDCG
> assumes one hit per annotation — so DCG can legitimately exceed IDCG. The result is
> clamped to `1.0` rather than left unbounded or hidden inside the ratio. Consequence: an
> NDCG of exactly `1.0` can mean either a perfect ranking or a clamped one; cross-check
> against P@k/R@k to tell which.
> **What it does not measure.** Anything beyond rank 10.
> **How to game it.** Return several chunks that all straddle the same one or two relevant
> clauses — the clamp caps the damage at `1.0`, it does not penalise the redundancy.

> **DRM rate** (`drm_rate`)
> **Formula.** `DRM = |top-k hits whose document_id is not in query_document_ids| / k`,
> called with a hardcoded `k=10` inside `compute_retrieval_metrics` regardless of the
> configured `top_k`. Returns `0.0` for `k <= 0`, empty `query_document_ids`, or no hits at
> rank `<= k`.
> **Range.** `[0.0, 1.0]`. **Direction.** lower is better.
> **What it does not measure.** Relevance within the correct document — a chunk from the
> right document can still be irrelevant, and DRM does not catch that.
> **Why `drm_hit` is uninformative here.** `RetrievalMetrics.drm_hit = drm_rate > 0.0` is a
> boolean derived from the rate. All five fixture documents currently share a single FAISS
> index per (strategy, embedding model) (see `models.py`'s own comment on the field), which
> is exactly the condition under which almost every query has at least one wrong-document
> hit somewhere in a top-10 — so `drm_hit` is close to always-true and carries no
> discriminating information. The underlying `drm_rate` is still informative; report that,
> not the boolean.
> **How to game it.** Not really gameable independent of retrieval quality — it is a
> property of the shared index plus the embedding, not of anything a chunking strategy
> controls directly, beyond how document-distinguishable its chunk boundaries make the
> text.

### Residual bias, honestly stated

Span-overlap relevance is not immune to chunk size the way the old text-similarity judge
was, but it is not immune to it either: a chunk large enough to straddle a relevant clause's
boundary matches under `matched_sections` even when most of its text is about something
else. This is why `rcts_1024` exists in `BenchmarkConfig.strategies`
(`src/legal_rag_eval/config.py`) — a fixed-size baseline sized to match LexiChunk's own typical
output (documented in `config.py` as "the size-matched control for lexichunk (~790-char
chunks)") — and why mean chunk length must be printed next to every retrieval table, not
just the metric values. A strategy that wins on P@k/R@k/NDCG purely by emitting larger
chunks than its comparator is not shown to chunk better; it is shown to be larger.

## 5. Statistics

Computed by `src/legal_rag_eval/metrics/statistical.py::compute_all_comparisons`, into a list of
`ComparisonResult`. This module replaces a prior implementation described in its own
docstring as "indefensible": 21 uncorrected two-tailed paired t-tests at n=22, no confidence
intervals, no distribution-free test, a Cohen's d that silently returned `0.0` for a
zero-variance non-zero-mean difference, an `improvement_pct` computed as a fragile ratio of
two small means, and no report of how much any result depends on a single document.

- **Bootstrap 95% CI on every delta** (`paired_bootstrap_ci`). Resamples the *pairs*
  (`a - b` per query, never `a` and `b` independently) with replacement, using
  `numpy.random.default_rng(seed)` — `seed` and `n_resamples` are threaded through from
  `BenchmarkConfig.seed` (default `42`) and `BenchmarkConfig.bootstrap_resamples` (default
  `10000`), so a given config produces a deterministic, reproducible interval. Returns the
  2.5th/97.5th percentiles of the resampled means as `(ci_low, ci_high)`. Requires at least
  2 pairs; returns `(nan, nan)` otherwise. Does not test normality and does not itself
  correct for multiple comparisons.
- **Holm correction, once, across the whole family** (`holm_correction`). Sorts p-values
  ascending, multiplies the `i`-th smallest (1-indexed) by `m - i + 1`, enforces
  monotonicity via a running maximum, clamps to `1.0`, and returns adjusted values in the
  original order. `compute_all_comparisons` calls this **exactly once**, over
  `[r.p_value_t for r in provisional]` — every metric, every baseline, and every embedding
  model in one run together form the family. **Concretely, this is applied to the paired
  t-test's p-values, not to the Wilcoxon p-values** — `p_value_wilcoxon` is carried on
  `ComparisonResult` unadjusted, for the reader to weigh alongside the Holm-adjusted t-test
  result, not as a second thing that itself gets corrected.
- **Wilcoxon signed-rank alongside the t-test** (`wilcoxon_signed_rank`), because a
  discrete, coarsely-graded metric (`RelevanceGrade` values 0-3, `min(k, hits)` denominators)
  produces heavy ties at small `n`, where a paired t-test's normality assumption is weakest.
  Uses `zero_method="wilcox"` (zero differences dropped before ranking), two-sided. When
  every paired difference is exactly zero, scipy has nothing to rank and raises
  `ValueError`; that is caught and reported as `(0.0, 1.0)` — "no evidence of a
  difference", not an error.
- **Cohen's d** (`cohens_d_paired`) = `mean(a - b) / sample_std(a - b, ddof=1)`. When the
  differences have zero variance (every pair improved by exactly the same amount) the ratio
  is undefined in the usual sense: if the mean difference is also `0.0` this returns `0.0`
  (no effect), but **if every pair moved by the same non-zero amount this returns `+inf` or
  `-inf`**, not `0.0` — a perfectly consistent, arbitrarily-precisely-estimated effect, and
  the bug the rewrite specifically fixed.
- **Rank-biserial correlation** (`rank_biserial_correlation`), the effect size that pairs
  with Wilcoxon: `(sum of ranks of positive diffs - sum of ranks of negative diffs) / sum
  of all ranks`, over `|a - b|` with zero differences dropped. Ranges `[-1, 1]`; `0.0` when
  every difference is zero.
- **Leave-one-document-out range** (`leave_one_document_out`). Recomputes `mean(a - b)`
  with each document's queries removed in turn; a document whose removal would leave fewer
  than 2 pairs is skipped. `lodo_min_delta`/`lodo_max_delta` on `ComparisonResult` are the
  smallest/largest such recomputed deltas; `lodo_worst_document` is the one whose removal
  moves the delta furthest from the full-sample delta. Document grouping comes from an
  explicit `query_documents` mapping when supplied, otherwise from a heuristic
  (`_infer_document_id`) that strips a trailing `_q<suffix>` token off the query id — a
  best-effort grouping, not guaranteed to match the true fixture document id when a
  document id itself contains an underscore-separated suffix.

**Read the absolute delta and its CI as the headline** — `ComparisonResult.delta` plus
`(ci_low, ci_high)` — not a percentage improvement over a small base.
`ComparisonResult.to_significance_result()` recomputes the legacy `improvement_pct =
(mean_a - mean_b) / mean_b * 100` purely for backward-compatible JSON export; its own
docstring says it is "unstable when mean_b is small" and to prefer `delta`/`ci_low`/
`ci_high` for anything new.

**Queries within a document are not independent.** A benchmark run's `n` is the number of
paired queries, but many queries share a document, so the effective sample size for
anything document-level is closer to `n_documents` than to `n`. This is exactly what the
leave-one-document-out range is for: check it before trusting a tight confidence interval
built on queries that are not really independent draws.

## 6. Superseded metrics

`src/legal_rag_eval/metrics/structural.py` holds the `legacy_*` metrics
(`legacy_clause_fragmentation_rate`, `legacy_definition_preservation_rate`,
`legacy_cross_ref_resolution_rate`, `legacy_hierarchy_depth_retained`,
`legacy_chunk_size_cv`), assembled into `LegacyStructuralMetrics`
(`src/legal_rag_eval/models.py`, aliased as `StructuralMetrics` for backward compatibility). The
module's own docstring calls them retained "for provenance; not part of the default
report", and the reasons are structural, not incidental:

- **Ground truth is LexiChunk's own parse** (`get_ground_truth`, which runs
  `LegalChunker` over the document). LexiChunk is therefore graded against its own output —
  its `legacy_clause_fragmentation_rate` is `0.000` by construction and cannot move no
  matter how its segmentation actually changes.
- **Hierarchy depth uses the same field as numerator and denominator, for the strategy the
  metric was built to flatter** (`legacy_hierarchy_depth_retained`). The denominator,
  `gt.max_hierarchy_depth`, comes from `LegalChunker.hierarchy_path` on a fresh run of
  LexiChunk over the document (falling back to a regex, `_infer_hierarchy_depth`, only if no
  chunk carries a `hierarchy_path` at all). For LexiChunk and LexiChunk-contextual, the
  numerator's `chunk.metadata["section_hierarchy"]` is set by the chunking wrapper itself
  to `str(lc.hierarchy_path)` (in both LexiChunk strategy wrappers in
  `src/legal_rag_eval/chunking/strategies.py`) — **literally the same field**, computed by two
  separate invocations of the same
  chunker over the same text, so the ratio is close to `1.0` almost by definition. Baseline
  strategies carry no such metadata and fall back to a plain regex
  (`r"(\d+(?:\.\d+)*)\."`) applied to their own chunk text — a much weaker channel, scored
  against a denominator that is still LexiChunk's own parse. Either way, a strategy that
  preserves the document's original numbering text (as most do, since they copy source text
  verbatim) scores well on this ratio regardless of whether its chunk *boundaries* respect
  that hierarchy; the metric measures "does the numbering text survive copying", not "is
  the hierarchy preserved by the chunking".
- **Definition preservation and cross-reference resolution grant a pass on
  LexiChunk-only metadata.** `legacy_definition_preservation_rate` counts a term as
  preserved if the chunk's text contains a definition indicator word/quoting pattern OR
  `chunk.metadata.get("defined_terms")` contains the term; `legacy_cross_ref_resolution_rate`
  similarly accepts `chunk.metadata.get("cross_references")` as sufficient for "resolved"
  with no further check. No baseline strategy emits this metadata, so only LexiChunk (and
  its contextual variant) can ever benefit from the metadata branch — a baseline is scored
  purely on regex/substring matching over its own chunk text.
- **The baselines' scores move when LexiChunk's parser changes, even though the
  baselines have not.** Because ground truth is `get_ground_truth(document)` — cached per
  `document.id`, computed by running the current `LegalChunker` — a byte-identical baseline
  `ChunkSet` scores differently after any change to LexiChunk's own segmentation, since the
  yardstick moved under it.

These are computed only when the CLI is run with `--legacy-metrics`
(`src/legal_rag_eval/__main__.py`) and are excluded from the default report entirely.

## 7. Reading a results table

Before comparing two strategies' numbers, in this order:

1. **Check `n`.** How many paired queries actually fed this comparison
   (`ComparisonResult.n`, or `RetrievalMetrics` count for a single strategy)?
2. **Check localisation rate** (`GoldStructuralMetrics.localization_rate`) for every
   strategy in the row. A structural number computed over 50% of a strategy's chunks is not
   directly comparable to one computed over 95% of another's.
3. **Check mean chunk length before comparing strategies** on any retrieval metric — span
   overlap relevance still favours longer chunks (section 4); a win is not shown to be a
   quality win until you know the comparison isn't just "bigger chunks".
4. **Prefer the CI to the p-value.** `ComparisonResult.delta` plus `(ci_low, ci_high)` tells
   you the size and the uncertainty of the effect; a p-value alone does not.
5. **Prefer the Holm-adjusted p to the raw p** (`p_value_holm` / `significant_holm`, not
   `p_value_t`) — the raw p-value has not been corrected for the fact that many metrics,
   baselines and embedding models were tested in the same run.
6. **Check the LODO range** (`lodo_min_delta`, `lodo_max_delta`, `lodo_worst_document`). A
   result that flips sign, or moves by more than its own CI width, when one document is
   dropped is a result driven by that one document, not a general property of the
   strategies being compared.
