# tools/

Maintenance scripts. None of these is part of the harness: nothing in
`src/legal_rag_eval/` imports them, and none is needed to run a benchmark or to read a
result. They are here rather than as CLI subcommands because each one *changes committed
files* — the gold annotations, the README — and that is not something a benchmark command
should be able to do as a side effect.

Every one of them has a `make` target; the target is the supported way to run it.

| Script | What it is for | Run it with |
|---|---|---|
| [`build_gold.py`](build_gold.py) | Seeds `gold/*.json` from each fixture document's own clause numbering, with regexes. A **first pass only** — the committed annotations are hand-corrected on top of it, and every correction is logged in [`gold/CHANGES.md`](../gold/CHANGES.md). `--check` re-seeds in memory and diffs against the committed files. Stdlib-only, and deliberately imports neither `legal_rag_eval` nor `lexichunk`, so the structural ground truth cannot be derived from the thing it grades. | `make gold`, `make gold-check` |
| [`update_readme.py`](update_readme.py) | Rewrites the block between the `BEGIN/END GENERATED RESULTS` markers in `README.md` from an exported benchmark run. The only supported way to change those tables; `--check` exits 1 if they have drifted, which CI runs on every push. | `make readme`, `make readme-check` |
| [`compare_builds.py`](compare_builds.py) | Prints a Markdown table putting one strategy's numbers from two result files side by side, so a change in the *chunker* shows up as a number. The rest of the harness compares strategies within a run, which cannot answer "did this parser change help?". Writes nothing; paste the output if you want it published. | `make compare-builds` |

`build_gold.py --force` overwrites hand corrections and is destructive. `make gold`
refuses to overwrite a corrected file precisely so that a re-seed cannot quietly discard
annotation work; see [`gold/README.md`](../gold/README.md) for why
`make gold-check` is *expected* to report drift.
