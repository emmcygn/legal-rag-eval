"""Streamlit viewer for anchored-evidence benchmark reports."""

from __future__ import annotations

import streamlit as st


def main() -> None:
    """Configure and launch the evidence report viewer."""
    st.set_page_config(
        page_title="Legal RAG Evidence Results",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.sidebar.title("Evidence Results")
    st.sidebar.caption("Viewer only: scoring is performed by the primary offline CLI.")

    from legal_rag_eval.dashboard.page_metrics import render_page

    render_page()


if __name__ == "__main__":
    main()
