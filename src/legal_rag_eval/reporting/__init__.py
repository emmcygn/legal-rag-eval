"""CLI, JSON, Markdown, and HTML report generation."""

from legal_rag_eval.reporting.cli import (
    render_benchmark,
    render_comparisons_table,
    render_legacy_structural_table,
    render_retrieval_table,
    render_structural_table,
)
from legal_rag_eval.reporting.html import render_html_report
from legal_rag_eval.reporting.json_export import (
    export_json,
    export_json_string,
    load_json,
    reconstruct_benchmark_result,
)
from legal_rag_eval.reporting.markdown import (
    render_results_markdown,
    update_readme,
    write_results_markdown,
)

__all__ = [
    "export_json",
    "export_json_string",
    "load_json",
    "reconstruct_benchmark_result",
    "render_benchmark",
    "render_comparisons_table",
    "render_html_report",
    "render_legacy_structural_table",
    "render_results_markdown",
    "render_retrieval_table",
    "render_structural_table",
    "update_readme",
    "write_results_markdown",
]
