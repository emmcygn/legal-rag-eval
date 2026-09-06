# Query annotation schema

Each fixture document has one query file, named after the document:

```
queries/eu_gdpr_excerpt.yaml
queries/uk_service_agreement.yaml
queries/uk_terms_conditions.yaml
queries/us_msa.yaml
queries/us_terms_of_service.yaml
```

There are 30 queries in total, 6 per document. `tests/test_query_annotations.py`
validates every rule stated below, so a violation cannot be committed.

## Structure

```yaml
document_id: <string>          # must match the fixture filename (without .txt)
queries:
  - id: <string>               # unique across all files
    text: <string>             # the question, in a reader's words
    category: <enum>           # see "Categories"
    relevant_clauses:
      - identifier: <string>   # must exist in gold/<document_id>.json
        relevance: <1|2|3>     # graded relevance
        description: <string>  # what that clause says, for a human reviewer
    notes: <string>            # why this query is here and what it tests
```

`identifier` names a clause in the document's **gold annotation**, using the
document's own printed numbering (`"3.6(a)"`, `"section_1.01(a)"`, `"article_5_1"`,
`"recital_1"`). `scaffolder.queries.resolve_queries` turns each identifier into the
character span of that clause's *subtree* — the clause plus its descendants — so
naming a parent clause counts its sub-clauses' text too. Nothing here is derived from
any chunker: the gold annotations were seeded from each document's numbering and then
hand-checked (see `gold/CHANGES.md`), and a query that names an identifier which does
not exist raises at load time rather than silently scoring zero for everyone.

The loader also accepts the pre-migration keys `relevant_sections` / `section_id` /
`failure_mode` as aliases, but no committed file uses them.

## Graded relevance

- **3 (exact)** — this clause answers the question. Every query has at least one.
- **2 (partial)** — needed to answer fully, or the clause the answer depends on.
- **1 (background)** — helps, but is not part of the answer.

## Categories

`category` records the *kind of question*, not a chunker failure mode. The earlier
schema labelled each query with one of five LexiChunk failure modes
(`clause_fragmentation`, `orphaned_cross_refs`, `lost_definitions`,
`destroyed_hierarchy`, `cross_doc_contamination`); an independent audit pointed out
that those five map almost one-to-one onto the structural metrics the project uses to
argue for LexiChunk, so the query set was labelled by the claims it was meant to
support. These labels are neutral with respect to any chunker:

| Category | What the query asks for |
|---|---|
| `definition_lookup` | the meaning the document assigns to a defined term |
| `clause_lookup` | the substantive rule stated in one clause |
| `numeric_lookup` | a specific number: a notice period, a cure period, a cap |
| `conditional` | what happens in a stated situation ("if X, then?") |
| `multi_clause` | an answer that needs two clauses, usually in different articles |
| `cross_reference` | a clause reached through another clause's reference to it |

The set is fixed; adding a category means changing `ALLOWED_CATEGORIES` in
`tests/test_query_annotations.py` and saying why here.

## Rules a query must satisfy

1. **Answerable.** Every `identifier` exists in the target document's gold file, and
   at least one has `relevance: 3`.
2. **Not degenerate.** The union of a query's relevant subtree spans covers at most
   25% of the document's characters, and its relevant clauses are at most 30% of the
   document's gold clauses. A query that marks most of a document relevant is
   satisfied by retrieving almost anything.
3. **No leakage.** The query text does not contain the identifier of its own answer.
4. **A reader's words.** Queries paraphrase; they avoid quoting the clause's own
   distinctive wording, so a lexical match does not win by default.
