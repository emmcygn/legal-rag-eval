# evals/

Label mappings for the external, public-dataset evaluations
(`make evals`, `python -m legal_rag_eval.evals`). Everything here is a hand-written
mapping between someone else's label set and LexiChunk's — never a result, never anything
derived from a chunker's output. The runs themselves land in
[`../results/external/`](../results/external/), and the methodology is in
[`../docs/external_evals.md`](../docs/external_evals.md).

| File | What it is |
|---|---|
| [`ledgar_label_map.yaml`](ledgar_label_map.yaml) | An explicit many-to-one mapping from LEDGAR's 100 clause-type labels (LexGLUE `ledgar` config, pinned to dataset revision `c23fdff`) onto LexiChunk's 31 `ClauseType` values, with a written rationale per label. |

## Why the mapping is a committed file

LEDGAR's taxonomy and LexiChunk's do not line up: LEDGAR is drawn overwhelmingly from
SEC-filed credit, M&A and employment agreements, and its vocabulary is narrower and more
finance-specific than a general contract taxonomy. 67 of the 100 labels map; the other 33
are marked `__out_of_scope__` and their rows are excluded from the score rather than
forced onto an approximate class. Nothing is ever mapped to `unknown` (a
classifier-failure sentinel, not a gold label) or to the positional `preamble`/`recitals`
classes.

That means the mapping is a judgement call that materially moves the reported accuracy,
so it is committed, versioned and reviewable rather than computed at run time. It is
loaded by `legal_rag_eval.evals.label_map`, which validates that every target names a real
`ClauseType`; `tests/test_evals_label_map.py` enforces that in CI. `results/external/ledgar.json`
records the coverage the run actually used (`label_coverage`, `mapped_labels`).

The file also ships with the wheel, under `share/legal-rag-eval/evals/`, so an installed
copy can reproduce the run.
