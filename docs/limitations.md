# Limitations

- The bundled dataset is small, synthetic, AI-authored, and not reviewed by a lawyer.
- The lexical cosine term-frequency ranker is deterministic but not representative of semantic, hybrid, reranked, or generated-answer systems.
- Character-overlap evidence scoring measures retrieval context, not answer faithfulness, legal correctness, usefulness, or citation quality.
- Only boundary-only source slices are evaluated. LexiChunk context headers and expansion features are deliberately excluded.
- The first release supports the SDK's declared `uk`, `us`, and `eu` jurisdiction values and rejects others.
- The shared `\S+` budget is transparent and portable, but it is not a model tokenizer.
- Aggregate means can hide query-level failures; inspect the per-query records and comparison outcomes.
- The legacy structural and embedding diagnostics use SDK-duplicate fixtures and must not be treated as independent benchmark evidence.
- Dataset provenance and review fields are declarations supplied by the dataset author, not independently verified certifications.

Before making external performance claims, use a representative corpus with lawful provenance, documented licensing, independent annotation, adjudication, held-out controls, and a predeclared analysis plan.
