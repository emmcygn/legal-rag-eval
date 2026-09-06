# Ground truth

Three evaluations, three separate ground truths, none of them derived from a
chunker's own output. This document covers the two that live in this repository: the
hand-checked clause annotations in [`gold/`](../gold/) and the queries anchored to
them. The anchored-evidence dataset has its own [dataset card](dataset-card.md); the
external LEDGAR and CUAD labels are described in [external_evals.md](external_evals.md).

## The gold annotations

| Document | Source | Characters | Gold clauses (leaf) | Defined terms | Cross-refs | Queries |
|---|---|---:|---:|---:|---:|---:|
| `eu_gdpr_excerpt` | GDPR excerpt (real, abridged) | 5,141 | 21 (14) | 6 | 2 | 6 |
| `uk_service_agreement` | synthetic | 25,187 | 103 (77) | 18 | 31 | 6 |
| `uk_terms_conditions` | synthetic | 16,948 | 70 (55) | 16 | 40 | 6 |
| `us_msa` | synthetic | 25,346 | 57 (47) | 31 | 43 | 6 |
| `us_terms_of_service` | synthetic | 25,743 | 114 (92) | 19 | 55 | 6 |
| **Total** | | **98,365** | **365** (**285**) | **90** | **171** | **30** |

How the annotations were made, in order: [`tools/build_gold.py`](../tools/build_gold.py) reads each document's own
numbering with regexes and emits a first pass; a human then reads that pass against the
document and corrects it, clause span by clause span, adding the defined terms and
cross-references the regexes miss. [`gold/CHANGES.md`](../gold/CHANGES.md) records every correction per document,
including the categories an annotator checked and left alone, and the calls they were
unsure about. No chunker is consulted at any point, so no chunker is graded against its own
output.

Run `python -m pytest tests/test_gold_annotations.py` to validate every span against the
fixtures. The annotations pin each document by SHA-256, so editing a fixture fails loudly
instead of silently invalidating the spans.

LexiChunk is also evaluated against two public datasets it was not tuned on — LEDGAR for
clause-type classification and CUAD for boundary preservation. Those results are separate
from this harness and live in [external_evals.md](external_evals.md).

## Query annotations

**30 queries, 6 per document**, in [`queries/`](../queries/) — one YAML file per document.
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

[`queries/schema.md`](../queries/schema.md) documents the schema and the six `category` values. Those categories
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

