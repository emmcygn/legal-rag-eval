# Dataset Card: Synthetic Contracts v1

## Summary

`synthetic-contracts-v1` contains three compact contract-like documents, twelve source-exact evidence spans, twelve answerable queries, and three unanswerable queries. It exercises single evidence, multi-evidence, definitions with exceptions, cross-document distractors, topical negatives, and a zero-overlap negative.

## Provenance and Review

- Authorship: AI-authored.
- Provenance: synthetic; no source URLs are claimed.
- License status: distributed under this project's MIT license.
- Human legal review: none.
- Held-out status: false.
- Customer proof: false.
- Jurisdiction setting: UK, solely to select the SDK parser used by this synthetic fixture.

These declarations are stored in the manifest and repeated in reports. They are not independently certified and the content is not legal advice.

## Annotation Process

Questions, answer spans, and evidence spans are stored separately from chunker output. Character offsets point to exact UTF-8 document text and document SHA-256 values prevent silent source drift. Answer spans must be covered by the referenced evidence in the same document.

The challenge includes topical unanswerable questions about a service-credit percentage and critical-incident response time, even though nearby text uses those topics. This tests false-positive retrieval behavior more realistically than the zero-overlap tribunal control alone.

## Intended Use

Use this dataset to smoke-test installation, schema validation, deterministic scoring, reporting, and baseline trade-offs. Do not use it to claim production accuracy, legal validity, customer performance, or general superiority.

## Extending the Dataset

Create a new manifest rather than changing the meaning of `synthetic-contracts-v1`. Record truthful authorship, review, provenance, license, held-out, and customer-proof declarations. Public-source documents require a real source URL. Human-reviewed corpus status requires every included document to declare human review.
