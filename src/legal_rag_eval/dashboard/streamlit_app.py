"""Streamlit entry script for hosted deployments.

`make dashboard` runs :mod:`legal_rag_eval.dashboard.app` directly and is the supported
way to start the viewer from a checkout. This script exists for hosts that want a file
path to run; it assumes the package is installed, which the `[dashboard]` extra does:

    pip install "legal-rag-eval[dashboard]"
    streamlit run src/legal_rag_eval/dashboard/streamlit_app.py

It used to live at the repository root and prepend `src/` to `sys.path`, which meant a
deployment could silently run against a source tree that was never installed.
"""

from legal_rag_eval.dashboard.app import main

main()
