# Gold annotations

This directory holds the gold-standard clause/defined-term/cross-reference annotations for
the five fixture documents in `src/legal_rag_eval/fixtures/documents/`. See `SCHEMA.md` for the
binding JSON contract.

## Where these come from

The annotations are produced **independently of LexiChunk**. `tools/build_gold.py` is a
standalone, stdlib-only regex seeder that reads each fixture's own numbering (recitals,
chapters/articles, clause/section decimals, schedules, lettered sub-paragraphs) and derives
clause spans, defined terms, and explicit cross-references directly from the text. No
LexiChunk output is consulted at any point.

The seeded files in this directory are **not** the final word: they are then corrected by
hand against the actual document text (fixing mis-seeded spans, adding references or
definitions the regex patterns can't reliably catch, reclassifying ambiguous cross-references,
etc.). Hand corrections are logged in `gold/CHANGES.md`.

## Regenerating

```
python tools/build_gold.py                 # seed gold/*.json for all five fixtures
python tools/build_gold.py --doc us_msa    # one document
python tools/build_gold.py --check         # re-seed in memory and diff against gold/*.json
python tools/build_gold.py --force         # overwrite hand corrections (destructive)
```

`--check` exits non-zero if the freshly-seeded output differs from what is committed.

**The seeder refuses to overwrite a hand-corrected file.** If `gold/<doc>.json` differs
from a fresh seed, that difference *is* the correction pass, so a plain run prints
`REFUSED` and exits non-zero rather than discarding it. Only `--force` overwrites. This
guard exists because the corrections were destroyed exactly once during development by a
stray `--doc` run, and restoring them cost a full re-annotation.

## Important: `--check` failures are expected

Because the committed `gold/*.json` files include hand corrections, running
`python tools/build_gold.py --check` after any hand correction **will** report drift —
that is expected and correct, not a bug. The seeder is a reproducible starting point, not
the source of truth. Do not "fix" a `--check` failure by re-running the seeder and
overwriting a hand-corrected file; that would discard the correction.

`--check` is useful for two things instead:
- Confirming the seeder is byte-for-byte reproducible (no non-determinism) between runs.
- Reviewing exactly what a hand correction changed relative to the raw seed, by diffing
  the committed file against a scratch re-seed.

If a fixture document itself changes, `text_sha256` in the corresponding gold file will no
longer match, and the seeder should be re-run and the diff manually reconciled with any
existing hand corrections before committing.
