# Contributing

## Development Setup

Use Python 3.10–3.12 and install the development extra:

```bash
python -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
make PYTHON=.venv/bin/python ci
```

On Windows, use `.venv\Scripts\python.exe` instead of `.venv/bin/python`.
Without Make, run that interpreter with `-m ruff check src/legal_rag_eval tests`,
`-m ruff format --check src/legal_rag_eval tests`, `-m mypy src/legal_rag_eval`, and
`-m pytest`.

## Dataset Contributions

Do not submit documents without lawful provenance and an accurate license declaration. Do not invent source URLs, licenses, human review, held-out status, or customer validation.

Every source document needs a stable UTF-8 byte representation, SHA-256, jurisdiction, provenance, license status, and review status. Every evidence and answer span must exactly match the source slice. Include answerable, multi-evidence, wrong-document, overlap, budget, and topical unanswerable controls where relevant.

Keep synthetic fixtures explicitly labelled. A human-reviewed corpus declaration requires every document in that corpus to carry the corresponding review status.

## Code Contributions

Keep the primary evaluator deterministic and offline. Retrieval models and paid services must remain opt-in and must never be selected merely because an API key exists. Avoid adding a second scoring implementation to reports or the dashboard.

Run Ruff, mypy, the full test suite, an sdist/wheel build, and the primary benchmark before opening a change.
