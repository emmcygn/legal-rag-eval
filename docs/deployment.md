# Dashboard deployment

The Streamlit application is a viewer for `evidence_benchmark_report_v1` JSON produced by
the offline CLI. It does not run retrieval and does not recompute any score, so nothing it
displays can disagree with the committed evidence: if a number looks wrong there, the run
that produced the JSON is where to look.

## From a checkout

```bash
python -m pip install -e ".[dashboard]"
python -m legal_rag_eval benchmark
make dashboard
```

`make dashboard` runs `src/legal_rag_eval/dashboard/app.py`. The `[dashboard]` extra is the
only place the dashboard's dependencies are declared.

## On a host that runs a script path

Some hosts (Streamlit Cloud and similar) want a script to run and a requirements file
beside it. Both live in the package:

| | Path |
|---|---|
| Entry script | `src/legal_rag_eval/dashboard/streamlit_app.py` |
| Requirements | `src/legal_rag_eval/dashboard/requirements.txt` (installs `-e .[dashboard]`) |

```bash
pip install -r src/legal_rag_eval/dashboard/requirements.txt   # from the repository root
streamlit run src/legal_rag_eval/dashboard/streamlit_app.py
```

Both used to sit at the repository root, and the entry script prepended `src/` to
`sys.path` — which meant a deployment could run against a source tree that was never
installed, with the dependency list in `requirements-dashboard.txt` drifting from
`pyproject.toml`. It had, by the time it was removed.

No API keys are required. `.github/workflows/dashboard.yml` imports every dashboard module
and then runs a Streamlit health check against the entry script. Legacy live-retrieval
pages are not exposed in the application's navigation.
