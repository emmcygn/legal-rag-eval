"""Entry point for Streamlit Cloud deployment."""

import sys
from pathlib import Path

# Add src to path so legal_rag_eval package is importable
src_path = str(Path(__file__).parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from legal_rag_eval.dashboard.app import main  # noqa: E402

main()
