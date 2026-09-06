# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

There are **no git tags**. The version numbers below are assigned retrospectively to the
merged pull requests that produced them, so the history is readable; they were not
released under those numbers at the time, and nothing is published to PyPI. `version` in
`pyproject.toml` reads `1.0.0` and predates this file.

## [Unreleased]

### Changed

- **The Python package is now `legal_rag_eval`, not `scaffolder`.** The repository, the
  harness and the CLI were all called `legal-rag-eval`; only the import path still carried
  the name the code had before it was a benchmark.
  - `src/scaffolder/` → `src/legal_rag_eval/`, and every import with it.
  - Distribution name `scaffolder` → `legal-rag-eval`.
  - Console script `scaffolder` → `legal-rag-eval`, with the same five subcommands
    (`benchmark`, `benchmark-structural`, `benchmark-embed`, `benchmark-legacy`,
    `report`). `python -m legal_rag_eval` reaches the same entry point.
  - Wheel data-files prefix `share/scaffolder/` → `share/legal-rag-eval/`.
  - `scaffolder.yaml.example` → `legal-rag-eval.yaml.example`, and the config file the
    CLI looks for by default is `legal-rag-eval.yaml`.
- **Environment overrides use the `LEGAL_RAG_EVAL_` prefix.** The old `SCAFFOLDER_` prefix
  is still read for one release and raises a `DeprecationWarning`; a value under the new
  prefix always wins. This covers `SCAFFOLDER_LEXICHUNK_COMMIT` as well. See
  `legal_rag_eval.config.env_override`.
- **LexiChunk installs from PyPI** (`lexichunk>=0.9.0,<0.10`) instead of a git reference
  pinned to a commit. LexiChunk 0.9.0 is published, so the workaround for a PyPI 404 is no
  longer needed and a clean clone installs with no git access. Both releases the range
  admits, 0.9.0 and 0.9.1, were checked: each reproduces the committed `lexichunk_fixed`
  structural run on all 30 metric rows and the evidence run on all three aggregates,
  exactly.
- **One configuration file.** `scaffolder.yaml.example` and `evidence-benchmark.yaml.example`
  are merged into `legal-rag-eval.yaml.example`, with one commented section per benchmark
  (`evidence:`, `gold:`). Both loaders read their own section out of the same file and
  ignore the other's keys, each keeping its own strict unknown-key rejection. A file with
  no section headers is still read flat, so a config written against either old example
  keeps working; a file that mixes sections with loose top-level keys is rejected rather
  than guessed at. Precedence — CLI > env > file > defaults — is documented in the README
  and in the example itself.
- **The anchored-evidence benchmark reads `legal-rag-eval.yaml` from the working directory**
  when `--config` is not given, which the gold-scored benchmarks already did.
- **Top level: one job per directory.**
  - `streamlit_app.py` → `src/legal_rag_eval/dashboard/streamlit_app.py`, and it no longer
    prepends `src/` to `sys.path` — a hack that let a deployment run against a source tree
    that was never installed.
  - `requirements-dashboard.txt` → `src/legal_rag_eval/dashboard/requirements.txt`, reduced
    to `-e .[dashboard]`. It had restated the dependency list and drifted: it still pinned
    LexiChunk to a git commit after `pyproject.toml` had moved to the PyPI release.
    `make dashboard` and the `[dashboard]` extra are the entry points.
  - `scripts/` → `tools/`, with a README saying what each script is for and which `make`
    target runs it. Nothing was deleted; none of the three is superseded by a subcommand.
    They stay out of the CLI because each rewrites committed files, which a benchmark
    command should not be able to do as a side effect.
- **The README answers questions in the order a visitor asks them**: what this measures,
  the three benchmarks in one table with what each can and cannot conclude, how LexiChunk
  relates to this repository, install, results, reproduce. Methodology detail moved to
  `docs/` rather than being cut.

### Added

- `results/README.md` — every committed run, the LexiChunk version and commit that
  produced it, the harness commit it was committed in, and the command to regenerate it.
  It also names two gaps: no harness commit is recorded inside the result JSON, and the
  evidence report's evaluator package-tree and payload hashes no longer match a fresh run,
  because the rename changed the code those hashes fingerprint. No metric moved.
- `evals/README.md` — why the LEDGAR label mapping is a committed, reviewable judgement
  call, and what its coverage does to the reported accuracy.
- `tools/README.md` — what each maintenance script is for.
- `docs/ground-truth.md` — how the gold annotations and the anchored queries were made,
  moved out of the README.
- `docs/build-comparison.md` — the baseline-vs-fixed LexiChunk analysis and its caveats,
  moved out of the README.
- `docs/methodology.md` gains the pipeline, source layout, strategies compared, embedding
  models and metric inventory sections.
- `.gitattributes` marks `results/**/*.json` `linguist-generated`, so ~7 MB of
  machine-generated evidence no longer dominates the language statistics or diffs.
- Tests for the deprecated environment prefix and for the merged config file's section
  handling.

### Fixed

- Six numbers in the README prose described an older run than the committed one printed
  beside them. All were checked against `results/` and corrected: structural leaf
  fragmentation (0.064 → 0.030), top-level over-merge (0.236 → 0.020, which reverses the
  claim — the size-matched control is not ahead on that metric), sub-clause grouping
  (0.472 → 0.279), the MRR comparison against the size-matched control (LexiChunk is
  behind under both embedding models, not ahead under one), localisation (54% → 68.6%),
  cross-reference recall (0.350 → 0.774), and the CUAD containment and structure-recall
  figures, which had been regenerated in `docs/external_evals.md` but not in the README.
- `docs/limitations.md` said 34 of LEDGAR's 100 labels are out of scope; `ledgar.json`
  records 67 mapped, so it is 33.
- `make evals-smoke` wrote synthetic runs into `results/external/`, a directory
  `.gitignore`'s negations track. Following the documented smoke command and running
  `git add -A` would have committed meaningless numbers next to real evidence. Those
  filenames are now ignored.

### Removed

- No `scaffolder` compatibility alias package. Nothing outside this repository imports it,
  the distribution was never published, and a shim would be an import path with no user.
  Code that imports `scaffolder` must be updated; the environment-variable shim above
  covers the only interface that could plausibly be in someone's shell profile.
- `evidence-benchmark.yaml.example` and the top-level `requirements-dashboard.txt`,
  superseded as described above.

## [0.2.0] — 2026-09-06

Merged as [#3](https://github.com/emmcygn/legal-rag-eval/pull/3), "Harness v2".

### Added

- **Independent structural ground truth** in `gold/`: 365 hand-checked clause spans, 90
  defined terms and 171 cross-references across five fixture documents, seeded from each
  document's own numbering by `build_gold.py` and then corrected by hand, with every
  correction logged in `gold/CHANGES.md`. No chunker is consulted, so no chunker is graded
  against its own output.
- **Span-overlap metrics.** Every structural metric is computed by character-span overlap
  against the source, never from a chunker's self-reported offsets. A chunk that is not a
  contiguous span of the source is reported as unlocated rather than silently scored.
- **A size-matched control**, `rcts_1024`, so a retrieval comparison isolates what the
  chunking strategy contributes from what chunk size alone contributes.
- **Statistics**: paired bootstrap 95% intervals, Holm correction across the whole family
  of tests, Wilcoxon alongside the t-test, Cohen's *d*, and leave-one-document-out
  sensitivity.
- **A 30-query gold-anchored query set**, six per document, with relevant passages given
  by gold clause identifier rather than prose, and CI checks on every one of them.
- **External evaluations** against LEDGAR (LexGLUE) and CUAD (Atticus Project) —
  independent third-party labels neither this harness nor the SDK was tuned on — plus six
  defects they found in LexiChunk, filed with reproducers.
- **Generated README results tables**, rendered from a committed run, with a CI check that
  fails if they drift.

### Changed

- The original structural diagnostics scored strategies against LexiChunk's own parse.
  They are circular by construction and are now reachable only as `benchmark-legacy`,
  which warns loudly when it runs and is never treated as evidence.

## [0.1.0] — 2026-09-06

Merged as [#1](https://github.com/emmcygn/legal-rag-eval/pull/1) and
[#2](https://github.com/emmcygn/legal-rag-eval/pull/2).

### Added

- **The anchored-evidence benchmark**: a deterministic, offline evaluation in which every
  strategy ranks candidate source spans with the same lexical ranker and fills the same
  context budget, scored against 12 char-offset evidence spans authored independently of
  any chunker.
- A **strict versioned dataset interface** with SHA-256 document hashes, declared
  provenance and review status, and validation that fails loudly on a label whose stored
  text is not exactly `source[start:end]`.
- Negative controls: unanswerable queries, cross-document distractors and a zero-overlap
  negative, with abstention scored separately.
- Wheel/offline CI smoke tests, and Windows newline and hash validation.

### Changed

- Replaced unsupported headline gains with results that state their own limits. The bundled
  corpus is declared AI-authored, unreviewed and development-only.
- The dashboard became a viewer for committed report JSON; it does not recompute scoring.
- A present `VOYAGE_API_KEY` no longer enables paid API calls on its own; `--enable-voyage`
  is required as well.
