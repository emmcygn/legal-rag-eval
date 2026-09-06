# Gold annotation schema

One JSON file per fixture document: `gold/<document_id>.json`.

Annotations are **independent of LexiChunk**. They are derived from the document's own
numbering by `scripts/build_gold.py` (a regex seeder) and then corrected by hand against the
document text. No LexiChunk output is consulted at any point. Corrections are logged in
`gold/CHANGES.md`.

## Text the offsets refer to

All `char_start` / `char_end` offsets index into the **sanitised** document text, defined as:

1. read the fixture file as UTF-8,
2. remove every `U+FEFF` (BOM),
3. replace `\r\n` and lone `\r` with `\n`,
4. apply Unicode NFC normalisation.

This matches `LegalChunker._sanitize_input`. `scaffolder.gold.sanitize()` is the single
implementation used by the seeder, the loader, the validator and every metric.

Offsets are Python slice semantics: `text[char_start:char_end]` is the annotated span, so
`char_end` is exclusive.

## File structure

```json
{
  "document_id": "uk_service_agreement",
  "source_file": "src/scaffolder/fixtures/documents/uk_service_agreement.txt",
  "text_sha256": "<sha256 of the sanitised text, hex>",
  "n_chars": 25681,
  "numbering_style": "uk_decimal",
  "clauses": [ ... ],
  "defined_terms": [ ... ],
  "cross_references": [ ... ]
}
```

`text_sha256` pins the annotation to an exact document revision. The loader raises if the
fixture no longer hashes to this value, so a fixture edit can never silently invalidate the
gold spans.

### `clauses` — ordered, non-overlapping partition

```json
{
  "identifier": "1.1",
  "level": 2,
  "parent": "1",
  "char_start": 1042,
  "char_end": 1310,
  "is_top_level": false,
  "heading": "In this Agreement, the following terms..."
}
```

| field | meaning |
|---|---|
| `identifier` | The clause's own label as it appears in the document, normalised: `"1"`, `"1.1"`, `"1.1(a)"`, `"article_4"`, `"recital_26"`, `"schedule_1"`, `"section_1.01"`. Unique within the file. |
| `level` | 1 for a top-level unit, 2 for its direct children, 3 for grandchildren. |
| `parent` | `identifier` of the parent clause, or `null` at level 1. Must appear earlier in `clauses`. |
| `char_start`, `char_end` | The clause's **own** span: from the first character of its number/heading up to (not including) the first character of the *next* clause at any level. A clause span therefore **excludes its descendants**. |
| `is_top_level` | `true` iff `level == 1` (kept explicit so consumers never re-derive it). |
| `heading` | The heading text, or the first line of the clause body when the numbering carries no separate heading. Informational only; never used by a metric. |

Invariants enforced by `tests/test_gold_annotations.py`:

* `0 <= char_start < char_end <= n_chars`;
* the list is sorted by `char_start`, and consecutive spans do not overlap
  (`clauses[i].char_end <= clauses[i+1].char_start`);
* every non-null `parent` resolves to an earlier clause whose `level` is exactly one less;
* `identifier` values are unique;
* front matter (title block, parties, recitals in a contract) that belongs to no numbered
  clause is **not** annotated — the partition covers the numbered body only, so gaps between
  spans are legal, overlaps are not.

The **subtree span** of a clause — its own span extended to the end of its last descendant —
is derived at load time (`GoldClause.subtree_end`). Because the list is ordered and
non-overlapping, a subtree is always contiguous. Metrics that need "the whole of clause 7"
use the subtree span; metrics that need atomic units use **leaf** clauses (those with no
children).

### `defined_terms`

```json
{
  "term": "Confidential Information",
  "definition_span": [4820, 5310]
}
```

`definition_span` is `[start, end)` over the sanitised text and covers the *definition itself*
— the sentence or lettered paragraph that says what the term means (typically the block
containing `"X" means ...`). Terms are recorded once, at the place they are defined.

### `cross_references`

```json
{
  "raw_text": "clause 12.3",
  "char_start": 9184,
  "target_identifier": "12.3",
  "kind": "internal_clause"
}
```

| field | meaning |
|---|---|
| `raw_text` | The exact reference text as it appears; `text[char_start:char_start+len(raw_text)]` must equal it. |
| `char_start` | Offset of `raw_text` in the sanitised text. |
| `target_identifier` | The `identifier` of the clause referred to, when the target is inside this document; `null` when the reference points outside the document. |
| `kind` | One of `internal_clause`, `internal_schedule`, `internal_definition`, `external_statute`, `external_document`. |

`target_identifier` must resolve to a clause in this file whenever `kind` starts with
`internal_`.

## What the gold annotations do **not** claim

* They are not a legal analysis. They record *where* the document's own numbering puts its
  boundaries, not whether a clause is well drafted.
* Sub-clause depth is capped at level 3. Deeper lettered romanettes are folded into their
  parent's span.
* Cross-references are annotated where a numbered target is named explicitly. Anaphoric
  references ("the foregoing", "this Section") are out of scope and are not annotated, so
  cross-reference recall is measured against explicit references only.
* Defined terms are those the document itself marks as defined (quoted-and-defined, or listed
  in a definitions clause). Terms used but never defined in the document are not annotated.
