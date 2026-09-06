"""Shared Streamlit components for the dashboard."""

from __future__ import annotations

import html
import re
from typing import TYPE_CHECKING

import streamlit as st

if TYPE_CHECKING:
    from legal_rag_eval.models import Chunk, ChunkSet

CLAUSE_TYPE_COLORS: dict[str, str] = {
    # Core commercial
    "definitions": "#4CAF50",
    "interpretation": "#66BB6A",
    "term": "#2196F3",
    "termination": "#F44336",
    "payment": "#FF9800",
    "fees": "#FFA726",
    # Obligations & liability
    "obligations": "#3F51B5",
    "warranties": "#9C27B0",
    "indemnification": "#E91E63",
    "limitation_of_liability": "#D32F2F",
    "exclusions": "#C62828",
    # IP & confidentiality
    "intellectual_property": "#00BCD4",
    "confidentiality": "#009688",
    "data_protection": "#00897B",
    # Dispute & governance
    "governing_law": "#795548",
    "dispute_resolution": "#8D6E63",
    "arbitration": "#A1887F",
    "jurisdiction": "#6D4C41",
    "force_majeure": "#607D8B",
    # Administrative
    "notices": "#78909C",
    "assignment": "#90A4AE",
    "entire_agreement": "#B0BEC5",
    "amendments": "#546E7A",
    "severability": "#455A64",
    "waiver": "#37474F",
    # Structure
    "preamble": "#8E24AA",
    "recitals": "#7B1FA2",
    "schedules": "#5E35B1",
    "general": "#757575",
}

DEFAULT_CHUNK_COLOR = "#757575"


def _highlight_terms(text: str, terms: list[str]) -> str:
    """Highlight defined terms in text using HTML bold + colour."""
    escaped = html.escape(text)
    escaped = escaped.replace("\n", "<br>")

    for term in terms:
        pattern = re.compile(re.escape(html.escape(str(term))), re.IGNORECASE)
        replacement = (
            f'<strong style="color: #1565C0; '
            f'background: #E3F2FD;">{html.escape(str(term))}</strong>'
        )

        def replace_match(_match: re.Match[str], value: str = replacement) -> str:
            return value

        escaped = pattern.sub(replace_match, escaped)

    return (
        f'<div style="font-family: monospace; white-space: pre-wrap; '
        f'font-size: 0.9em;">{escaped}</div>'
    )


def render_chunk_card(
    chunk: Chunk,
    index: int,
    show_metadata: bool = True,
    highlight_terms: bool = True,
) -> None:
    """Render a single chunk as a styled expandable card."""
    clause_type = str(chunk.metadata.get("clause_type", "general"))
    confidence = float(str(chunk.metadata.get("confidence", 0.0)))
    color = CLAUSE_TYPE_COLORS.get(clause_type, DEFAULT_CHUNK_COLOR)

    char_count = chunk.char_count
    token_estimate = char_count // 4
    header = f"Chunk {index + 1}"
    if clause_type and clause_type != "general" and show_metadata:
        header += f" | {clause_type.replace('_', ' ').title()}"
    header += f" | ~{token_estimate} tokens"

    with st.expander(header, expanded=index < 3):
        # Colour bar + hierarchy breadcrumb
        hierarchy_path = str(chunk.metadata.get("hierarchy_path", ""))
        if hierarchy_path and show_metadata:
            st.markdown(
                f'<div style="border-left: 4px solid {color}; padding-left: 8px; '
                f'color: #888; font-size: 0.85em; margin-bottom: 8px;">'
                f"{html.escape(hierarchy_path)}</div>",
                unsafe_allow_html=True,
            )
        elif show_metadata:
            st.markdown(
                f'<div style="border-left: 4px solid {color}; padding-left: 8px; '
                f'height: 4px; margin-bottom: 8px;"></div>',
                unsafe_allow_html=True,
            )

        # Defined terms as badges
        raw_terms = chunk.metadata.get("defined_terms", [])
        defined_terms = list(raw_terms) if isinstance(raw_terms, (list, tuple)) else []
        if defined_terms and show_metadata:
            badges = " ".join(
                f'<span style="background: #E3F2FD; color: #1565C0; '
                f"padding: 2px 8px; border-radius: 12px; font-size: 0.8em; "
                f'margin-right: 4px;">{html.escape(str(term))}</span>'
                for term in defined_terms[:10]
            )
            st.markdown(badges, unsafe_allow_html=True)
            st.markdown("")

        # Cross-references
        raw_refs = chunk.metadata.get("cross_refs", [])
        cross_refs = list(raw_refs) if isinstance(raw_refs, (list, tuple)) else []
        if cross_refs and show_metadata:
            ref_text = ", ".join(html.escape(str(ref)) for ref in cross_refs[:5])
            st.markdown(
                f'<span style="color: #FF8F00; font-size: 0.85em;">Cross-refs: {ref_text}</span>',
                unsafe_allow_html=True,
            )

        # Chunk text
        display_text = chunk.text
        if highlight_terms and defined_terms and show_metadata:
            display_text = _highlight_terms(display_text, [str(t) for t in defined_terms])
            st.markdown(display_text, unsafe_allow_html=True)
        else:
            st.text(display_text)

        # Footer metadata
        if show_metadata and confidence > 0:
            st.caption(
                f"Clause confidence: {confidence:.0%} | "
                f"Characters: {char_count:,} | "
                f"Index: {chunk.index}"
            )


def render_chunk_list(
    chunk_set: ChunkSet,
    strategy_label: str = "",
    show_metadata: bool = True,
) -> None:
    """Render a full chunk list with summary stats."""
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Chunks", chunk_set.count)
    with col2:
        st.metric("Avg Size", f"{chunk_set.avg_chunk_size:.0f} chars")
    with col3:
        st.metric("Time", f"{chunk_set.elapsed_seconds:.3f}s")

    for i, chunk in enumerate(chunk_set.chunks):
        render_chunk_card(
            chunk,
            i,
            show_metadata=show_metadata,
            highlight_terms=show_metadata,
        )


def render_chunk_size_chart(
    lexi_chunks: ChunkSet,
    baseline_chunks: ChunkSet,
    baseline_name: str,
) -> None:
    """Render a Plotly bar chart comparing chunk size distributions."""
    import plotly.graph_objects as go

    lexi_sizes = [c.char_count for c in lexi_chunks.chunks]
    base_sizes = [c.char_count for c in baseline_chunks.chunks]

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=lexi_sizes,
            name="LexiChunk",
            marker_color="#2196F3",
            opacity=0.7,
            nbinsx=20,
        )
    )
    fig.add_trace(
        go.Histogram(
            x=base_sizes,
            name=baseline_name,
            marker_color="#FF9800",
            opacity=0.7,
            nbinsx=20,
        )
    )

    fig.update_layout(
        title="Chunk Size Distribution (characters)",
        xaxis_title="Chunk size (chars)",
        yaxis_title="Count",
        barmode="overlay",
        height=300,
        margin={"l": 40, "r": 20, "t": 40, "b": 40},
        legend={"yanchor": "top", "y": 0.99, "xanchor": "right", "x": 0.99},
    )

    st.plotly_chart(fig, use_container_width=True)


def render_metric_card(
    label: str,
    value: float,
    delta: float | None = None,
    fmt: str = ".3f",
) -> None:
    """Render a metric card with optional delta."""
    formatted = f"{value:{fmt}}"
    if delta is not None:
        st.metric(label, formatted, delta=f"{delta:+{fmt}}")
    else:
        st.metric(label, formatted)


def render_error(title: str, detail: str, suggestion: str = "") -> None:
    """Render a structured error message with optional suggestion."""
    st.error(f"**{title}**")
    st.markdown(f"> {detail}")
    if suggestion:
        st.info(f"**Suggestion:** {suggestion}")


def render_dependency_error(package: str, extra: str) -> None:
    """Render a helpful error for missing optional dependencies."""
    render_error(
        title=f"Missing dependency: {package}",
        detail=f"The `{package}` package is required for this feature but not installed.",
        suggestion=f'Install with: `pip install "legal_rag_eval[{extra}]"`',
    )


def render_api_key_error(service: str, env_var: str, url: str) -> None:
    """Render a helpful error for missing API keys."""
    render_error(
        title=f"{service} API key not found",
        detail=f"The environment variable `{env_var}` is not set.",
        suggestion=f"Get a key at {url} and set `export {env_var}=your-key`",
    )


def render_export_section() -> None:
    """Render export buttons in the sidebar.

    Provides download buttons for JSON results and HTML report.
    Only active when benchmark results are available.
    """
    import json

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Export**")

    result_data = st.session_state.get("benchmark_result")

    if result_data is None:
        st.sidebar.caption("Run a benchmark to enable exports.")
        return

    # JSON download
    json_str = json.dumps(result_data, indent=2)
    st.sidebar.download_button(
        label="Download JSON",
        data=json_str,
        file_name="legal_rag_eval_results.json",
        mime="application/json",
        key="export_json",
    )

    # HTML report download
    try:
        html_str = _generate_html_report(result_data)
        st.sidebar.download_button(
            label="Download HTML Report",
            data=html_str,
            file_name="legal_rag_eval_report.html",
            mime="text/html",
            key="export_html",
        )
    except Exception as e:
        st.sidebar.caption(f"HTML export unavailable: {e}")


def _generate_html_report(data: dict[str, object]) -> str:
    """Generate HTML report from benchmark results.

    Uses Agent A's Jinja2 template if available, falls back to
    a minimal self-contained HTML report.
    """
    import json

    return (
        "<!DOCTYPE html>\n"
        "<html>\n<head>\n"
        "    <title>legal-rag-eval Benchmark Report</title>\n"
        "    <style>\n"
        "        body { font-family: -apple-system, sans-serif; max-width: 800px;"
        " margin: 2rem auto; padding: 0 1rem; }\n"
        "        h1 { color: #1a237e; }\n"
        "        pre { background: #f5f5f5; padding: 1rem; border-radius: 4px;"
        " overflow-x: auto; }\n"
        "        .metric { display: inline-block; padding: 1rem; margin: 0.5rem;"
        " background: #e3f2fd; border-radius: 8px; }\n"
        "        .metric .value { font-size: 2rem; font-weight: bold; color: #1565c0; }\n"
        "        .metric .label { font-size: 0.9rem; color: #666; }\n"
        "    </style>\n"
        "</head>\n<body>\n"
        "    <h1>LexiChunk legal-rag-eval Report</h1>\n"
        "    <p>Generated from benchmark results</p>\n"
        "    <h2>Raw Results</h2>\n"
        f"    <pre>{json.dumps(data, indent=2)}</pre>\n"
        "</body>\n</html>"
    )
