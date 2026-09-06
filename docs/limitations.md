# Limitations

The harness runs three evaluations against three separate ground truths. Their limitations
differ, so they are listed separately.

## Anchored-evidence benchmark (`benchmark`)

- The bundled dataset is small, synthetic, AI-authored, and not reviewed by a lawyer.
- The lexical cosine term-frequency ranker is deterministic but not representative of semantic, hybrid, reranked, or generated-answer systems.
- Character-overlap evidence scoring measures retrieval context, not answer faithfulness, legal correctness, usefulness, or citation quality.
- Only boundary-only source slices are evaluated. LexiChunk context headers and expansion features are deliberately excluded.
- The first release supports the SDK's declared `uk`, `us`, and `eu` jurisdiction values and rejects others.
- The shared `\S+` budget is transparent and portable, but it is not a model tokenizer.
- Aggregate means can hide query-level failures; inspect the per-query records and comparison outcomes.
- Dataset provenance and review fields are declarations supplied by the dataset author, not independently verified certifications.
- n = 12 answerable and 3 unanswerable queries. The report gives means and win/loss/tie
  labels with no confidence intervals; treat a delta at this sample size as a direction, not
  a measurement.

## Gold-scored structural and retrieval benchmarks (`benchmark-structural`, `benchmark-embed`)

- Five documents, four of them synthetic and drafted for this repository. Only
  `eu_gdpr_excerpt.txt` is a real instrument, and it is a 5 KB abridgement.
- The gold annotations are hand-checked but by a single annotator, with no second reviewer
  and no adjudication. `gold/CHANGES.md` records the calls the annotator was unsure about.
- Anaphoric cross-references ("the foregoing", "this Section") are not annotated, clause
  depth is capped at level 3, and undefined terms have no gold entry. Each of those bounds
  a recall number rather than being a neutral omission.
- Relevance by span overlap is still biased toward longer chunks, just less severely than
  the text-matching judge it replaced. `rcts_1024` is the size-matched control, and every
  table reports mean chunk length next to the metric.
- Heading and cross-reference metrics read `n/a` for every strategy that emits no such
  metadata. A metric only one entrant can be scored on measures a capability; it cannot
  rank the field, and nothing here uses it to.
- With n = 30 queries, no retrieval comparison in the published run survives Holm
  correction in either direction.

## External evaluations (`python -m scaffolder.evals`)

- LEDGAR and CUAD carry independent human labels, which is their point, but neither was
  designed to evaluate chunking. LEDGAR provisions arrive isolated with no document
  position, so position-derived clause types cannot be scored at all; 34 of its 100 labels
  are out of scope for the mapping and are reported as coverage rather than forced.
  The LEDGAR label mapping is itself a judgement call, documented per entry.
- CUAD span containment rewards long chunks mechanically. It is reported against a
  length-matched analytic grid expectation for that reason, and the raw numbers for
  overlapping strategies are flagged rather than quoted.

## Everywhere

- The legacy structural diagnostics (`benchmark-legacy`) score strategies against
  LexiChunk's own parse. They are circular by construction, retained for provenance only,
  and must never be treated as independent evidence.

Before making external performance claims, use a representative corpus with lawful provenance, documented licensing, independent annotation, adjudication, held-out controls, and a predeclared analysis plan.
