"""Streamlit multi-page dashboard for scaffolder results."""

from __future__ import annotations

import os
from typing import Any

import streamlit as st

from scaffolder.chunking import DEFAULT_STRATEGIES
from scaffolder.models import StrategyName


def _invalidate_on_change(key: str, new_value: Any) -> bool:
    """Track a value and return True if it changed, clearing relevant caches."""
    prev_key = f"_prev_{key}"
    prev = st.session_state.get(prev_key)

    if prev != new_value:
        st.session_state[prev_key] = new_value
        if key == "global_model":
            keys_to_clear = [k for k in list(st.session_state) if str(k).startswith("index_")]
            for k in keys_to_clear:
                del st.session_state[k]
            return True
        if key == "strategies":
            keys_to_clear = [
                k
                for k in list(st.session_state)
                if str(k).startswith("index_") or str(k).startswith("ret_")
            ]
            for k in keys_to_clear:
                del st.session_state[k]
            return True
    return False


def main() -> None:
    """Configure and launch the Streamlit dashboard."""
    st.set_page_config(
        page_title="LexiChunk Scaffolder",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom CSS for polish
    st.markdown(
        """
    <style>
        @media (max-width: 768px) {
            .block-container { padding-left: 1rem; padding-right: 1rem; }
        }
        .streamlit-expanderHeader { font-size: 0.95em; }
        [data-testid="stMetricValue"] { font-size: 1.5rem; }
        .stDataFrame { font-size: 0.9em; }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Sidebar navigation
    st.sidebar.title("LexiChunk Scaffolder")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigate",
        options=["Compare Chunks", "Retrieval Demo", "Metrics Dashboard"],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Settings**")

    strategy_options = [s.value for s in DEFAULT_STRATEGIES]

    # Global settings in sidebar
    if "config" not in st.session_state:
        st.session_state.config = {
            "strategies": strategy_options,
            "embedding_model": "all-MiniLM-L6-v2",
        }

    # Strategy multi-select
    selected_strategies = st.sidebar.multiselect(
        "Strategies",
        options=strategy_options,
        default=[StrategyName.LEXICHUNK.value, StrategyName.RCTS_512.value],
    )
    st.session_state.config["strategies"] = selected_strategies
    _invalidate_on_change("strategies", tuple(selected_strategies))

    # Embedding model selection
    available_models = ["all-MiniLM-L6-v2", "bge-base-en-v1.5"]
    if os.getenv("VOYAGE_API_KEY"):
        available_models.append("voyage-law-2")

    selected_model = st.sidebar.selectbox(
        "Embedding Model",
        options=available_models,
        key="global_model",
        help="Voyage requires VOYAGE_API_KEY environment variable",
    )
    st.session_state.config["embedding_model"] = selected_model
    _invalidate_on_change("global_model", selected_model)

    if "voyage-law-2" in available_models:
        st.sidebar.success("Voyage API: Connected")
    else:
        st.sidebar.caption("Set VOYAGE_API_KEY to enable voyage-law-2")

    # Cache management section
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Cache**")

    index_count = sum(1 for k in st.session_state if str(k).startswith("index_"))
    chunk_count = sum(1 for k in st.session_state if str(k).startswith("chunks_"))
    st.sidebar.caption(f"Cached: {index_count} indexes, {chunk_count} chunk sets")

    if st.sidebar.button("Clear All Caches", type="secondary"):
        keys_to_clear = [
            k for k in list(st.session_state) if str(k).startswith(("index_", "chunks_", "ret_"))
        ]
        for k in keys_to_clear:
            del st.session_state[k]
        st.sidebar.success("Cache cleared!")
        st.rerun()

    # Route to pages
    if page == "Compare Chunks":
        from scaffolder.dashboard.page_compare import render_page

        render_page()
    elif page == "Retrieval Demo":
        from scaffolder.dashboard.page_retrieval import render_page

        render_page()
    elif page == "Metrics Dashboard":
        from scaffolder.dashboard.page_metrics import render_page as render_metrics_page

        render_metrics_page()

    # Export section in sidebar (available from any page)
    from scaffolder.dashboard.components import render_export_section

    render_export_section()


if __name__ == "__main__":
    main()
