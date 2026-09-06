"""Evidence report viewer with no independent scoring implementation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st

EXPECTED_REPORT_VERSION = "evidence_benchmark_report_v1"


def _validate_report(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("report root must be an object")
    if value.get("report_version") != EXPECTED_REPORT_VERSION:
        raise ValueError(f"report_version must be {EXPECTED_REPORT_VERSION}")
    if not isinstance(value.get("aggregates"), dict):
        raise ValueError("report aggregates must be an object")
    if not isinstance(value.get("results"), dict):
        raise ValueError("report results must be an object")
    return value


def _load_local_report() -> dict[str, Any] | None:
    path = Path("results/evidence-benchmark.json")
    if not path.is_file():
        return None
    return _validate_report(json.loads(path.read_text(encoding="utf-8")))


def render_page() -> None:
    """Render a previously generated anchored-evidence report."""
    st.title("Anchored Evidence Benchmark")
    st.markdown(
        "View the exact JSON produced by `python -m scaffolder benchmark`. "
        "This dashboard does not rerun retrieval or recompute scores."
    )

    source = st.radio(
        "Report source",
        options=["Local results/evidence-benchmark.json", "Upload JSON"],
        horizontal=True,
    )
    try:
        if source.startswith("Local"):
            report = _load_local_report()
            if report is None:
                st.info("Run `python -m scaffolder benchmark` or upload a report.")
                return
        else:
            uploaded = st.file_uploader("Upload evidence benchmark JSON", type=["json"])
            if uploaded is None:
                return
            report = _validate_report(json.loads(uploaded.getvalue().decode("utf-8")))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        st.error(f"Cannot load evidence report: {error}")
        return

    _render_corpus(report)
    _render_aggregates(report)
    _render_comparisons(report)
    _render_queries(report)


def _render_corpus(report: dict[str, Any]) -> None:
    corpus = report.get("corpus", {})
    config = report.get("config", {})
    if not isinstance(corpus, dict) or not isinstance(config, dict):
        st.error("Report corpus/config metadata is malformed.")
        return
    st.subheader("Corpus Declaration")
    st.write(corpus.get("title", corpus.get("id", "unknown")))
    st.caption(
        f"Authorship: {corpus.get('authorship', 'unknown')} | "
        f"Legal validation: {corpus.get('legal_validation', 'unknown')} | "
        f"Held out: {corpus.get('held_out', 'unknown')} | "
        f"Customer proof: {corpus.get('customer_proof', 'unknown')}"
    )
    st.warning("These declarations come from the dataset and are not independently certified.")
    st.caption(
        f"Budget: {config.get('context_budget_tokens', 'unknown')} "
        f"{config.get('token_budget_unit', 'unknown')}"
    )


def _render_aggregates(report: dict[str, Any]) -> None:
    aggregates = report["aggregates"]
    rows = []
    for strategy, metrics in aggregates.items():
        if not isinstance(metrics, dict):
            continue
        rows.append(
            {
                "Strategy": strategy,
                "Answerable queries": metrics.get("answerable_query_count"),
                "Mean evidence recall": metrics.get("mean_evidence_recall"),
                "Mean evidence precision": metrics.get("mean_evidence_precision"),
                "Unanswerable queries": metrics.get("unanswerable_query_count"),
                "Abstention accuracy": metrics.get("abstention_accuracy"),
            }
        )
    st.subheader("Aggregates")
    st.dataframe(rows, use_container_width=True, hide_index=True)
    st.caption(
        "Evidence precision/recall cover answerable queries only. Abstention accuracy records "
        "whether retrieval emitted context for unanswerable queries; it does not grade a legal "
        "answer."
    )


def _render_comparisons(report: dict[str, Any]) -> None:
    comparisons = report.get("comparisons", [])
    if isinstance(comparisons, list) and comparisons:
        st.subheader("LexiChunk Comparisons")
        st.dataframe(comparisons, use_container_width=True, hide_index=True)
        st.caption("Win, loss, tie, and mixed outcomes are shown without filtering.")


def _render_queries(report: dict[str, Any]) -> None:
    results = report["results"]
    if not results:
        return
    strategy = st.selectbox("Query results strategy", options=list(results))
    records = results.get(strategy, [])
    if not isinstance(records, list):
        st.error("Selected strategy results are malformed.")
        return
    rows = []
    for record in records:
        if not isinstance(record, dict):
            continue
        rows.append(
            {
                "Query": record.get("query_id"),
                "Answerable": record.get("answerable"),
                "Evidence recall": record.get("evidence_recall"),
                "Evidence precision": record.get("evidence_precision"),
                "Abstention correct": record.get("abstention_correct"),
                "Selected tokens": record.get("selected_context_tokens"),
            }
        )
    st.subheader("Per-query Results")
    st.dataframe(rows, use_container_width=True, hide_index=True)
