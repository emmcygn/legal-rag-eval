"""HTML report generator using Jinja2 templates and Plotly charts.

Uses the same gold-scored `GoldStructuralMetrics` and `ComparisonResult`-shaped statistics
as the CLI (`scaffolder.reporting.cli`) — see that module's docstring for why this file also
never imports `scaffolder.metrics.statistical` directly and instead checks the `ComparisonLike`
shape structurally.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import plotly.graph_objects as go
import plotly.io as pio
from jinja2 import Environment, FileSystemLoader

from scaffolder.reporting.cli import (
    ComparisonLike,
    aggregate_gold_structural_metrics,
    aggregate_legacy_structural_metrics,
    aggregate_retrieval_metrics,
)

if TYPE_CHECKING:
    from scaffolder.models import BenchmarkResult, GoldStructuralMetrics

logger = logging.getLogger(__name__)

_TEMPLATES_DIR = Path(__file__).parent / "templates"


def _fmt_rate(value: float) -> str:
    return f"{value:.3f}"


def _fmt_optional(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _fmt_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _fmt_effect_size(value: float) -> str:
    if math.isinf(value):
        return "+inf" if value > 0 else "-inf"
    if math.isnan(value):
        return "n/a"
    return f"{value:.3f}"


def _fmt_ci_or_lodo(low: float, high: float) -> str:
    if math.isnan(low) or math.isnan(high):
        return "n/a"
    return f"[{low:+.3f}, {high:+.3f}]"


def render_html_report(
    result: BenchmarkResult,
    output_path: Path,
) -> None:
    """Render a full HTML report from benchmark results.

    The output is a self-contained HTML file with:
    - Inline CSS (no external dependencies)
    - Inline Plotly charts (plotly.js loaded via CDN)
    - Gold-scored structural metrics table and charts
    - Retrieval metrics tables and charts (per embedding model)
    - `ComparisonResult`-based statistics (bootstrap CI, Holm-adjusted p, Wilcoxon, effect
      sizes, leave-one-document-out), falling back to the legacy uncorrected table with a
      warning when only that is available
    - Methodology section

    Args:
        result: Complete BenchmarkResult from a benchmark run.
        output_path: Where to write the HTML file.
    """
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        autoescape=True,
    )
    template = env.get_template("report.html.j2")

    context = _build_context(result)

    html = template.render(**context)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    logger.info("HTML report written to %s", output_path)


def _build_context(result: BenchmarkResult) -> dict[str, Any]:
    """Build the template context from benchmark results."""
    context: dict[str, Any] = {
        "timestamp": result.timestamp,
        "seed": result.seed if result.seed is not None else "unknown",
        "lexichunk_version": result.lexichunk_version or "unknown",
        "lexichunk_commit": result.lexichunk_commit or "unknown",
        "strategies": [s.value for s in result.strategies],
        "documents": result.documents,
        "models": [m.value for m in result.models],
        "structural_chart_html": _structural_bar_chart(result),
        "structural_heatmap_html": _structural_heatmap(result),
        "has_retrieval": bool(result.retrieval_metrics),
        "retrieval_charts": {},
        "model_comparison_html": "",
        "drm_chart_html": "",
    }

    context.update(_structural_table_context(result))
    context.update(_comparisons_context(result))

    if result.retrieval_metrics:
        models = sorted({rm.embedding_model.value for rm in result.retrieval_metrics})
        context["retrieval_by_model"] = _retrieval_table_context(result)
        for model in models:
            context["retrieval_charts"][model] = _retrieval_bar_chart(result, model)
        context["model_comparison_html"] = _model_comparison_chart(result)
        context["drm_chart_html"] = _drm_chart(result)
    else:
        context["retrieval_by_model"] = {}

    return context


def _structural_table_context(result: BenchmarkResult) -> dict[str, Any]:
    structural_rows: list[dict[str, Any]] = []
    n_structural_documents = len({m.document_id for m in result.structural_metrics})
    for a in aggregate_gold_structural_metrics(result.structural_metrics):
        structural_rows.append(
            {
                "strategy": a.strategy.value,
                "located": _fmt_pct(a.localization_rate),
                "leaf_frag": _fmt_rate(a.clause_fragmentation_rate),
                "over_merge": _fmt_rate(a.top_level_over_merge_rate),
                "sub_clause_grp": _fmt_rate(a.sub_clause_grouping_rate),
                "head_r": _fmt_optional(a.heading_recall),
                "head_p": _fmt_optional(a.heading_precision),
                "def_attach": _fmt_rate(a.definition_attachment_recall),
                "xref_r": _fmt_optional(a.xref_target_recall),
                "xref_p": _fmt_optional(a.xref_target_precision),
                "size_cv": _fmt_rate(a.chunk_size_cv),
                "avg_chars": f"{a.avg_chunk_chars:.0f}",
                "chunks": f"{a.chunk_count:.1f}",
            }
        )

    legacy_rows: list[dict[str, Any]] = []
    n_legacy_documents = len({m.document_id for m in result.legacy_structural_metrics})
    for legacy_agg in aggregate_legacy_structural_metrics(result.legacy_structural_metrics):
        legacy_rows.append(
            {
                "strategy": legacy_agg.strategy.value,
                "clause_frag": _fmt_rate(legacy_agg.clause_fragmentation_rate),
                "def_preserv": _fmt_rate(legacy_agg.definition_preservation_rate),
                "xref_resol": _fmt_rate(legacy_agg.cross_ref_resolution_rate),
                "hierarchy": _fmt_rate(legacy_agg.hierarchy_depth_retained),
                "size_cv": _fmt_rate(legacy_agg.chunk_size_cv),
                "avg_chars": f"{legacy_agg.avg_chunk_chars:.0f}",
                "chunks": f"{legacy_agg.chunk_count:.1f}",
            }
        )

    return {
        "structural_rows": structural_rows,
        "n_structural_documents": n_structural_documents,
        "legacy_structural_rows": legacy_rows,
        "n_legacy_documents": n_legacy_documents,
        "has_legacy_structural": bool(legacy_rows),
    }


def _retrieval_table_context(result: BenchmarkResult) -> dict[str, Any]:
    by_model: dict[str, Any] = {}
    models = sorted({rm.embedding_model.value for rm in result.retrieval_metrics})
    for model in models:
        model_metrics = [rm for rm in result.retrieval_metrics if rm.embedding_model.value == model]
        n_queries = len({rm.query_id for rm in model_metrics})
        rows = [
            {
                "strategy": a.strategy.value,
                "p1": _fmt_rate(a.precision_at_1),
                "p3": _fmt_rate(a.precision_at_3),
                "p5": _fmt_rate(a.precision_at_5),
                "p10": _fmt_rate(a.precision_at_10),
                "mrr": _fmt_rate(a.mrr),
                "ndcg10": _fmt_rate(a.ndcg_at_10),
                "drm_rate": _fmt_rate(a.drm_rate),
            }
            for a in aggregate_retrieval_metrics(model_metrics)
        ]
        by_model[model] = {"n_queries": n_queries, "rows": rows}
    return by_model


def _comparisons_context(result: BenchmarkResult) -> dict[str, Any]:
    rows = [c for c in result.comparisons if isinstance(c, ComparisonLike)]
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in sorted(rows, key=lambda c: (c.embedding_model, c.metric_name, c.strategy_b.value)):
        by_model[c.embedding_model].append(
            {
                "metric": c.metric_name,
                "a": c.strategy_a.value,
                "b": c.strategy_b.value,
                "n": c.n,
                "mean_a": _fmt_rate(c.mean_a),
                "mean_b": _fmt_rate(c.mean_b),
                "delta": f"{c.delta:+.3f}",
                "ci": _fmt_ci_or_lodo(c.ci_low, c.ci_high),
                "p_t": _fmt_rate(c.p_value_t),
                "p_wilcoxon": _fmt_rate(c.p_value_wilcoxon),
                "p_holm": _fmt_rate(c.p_value_holm),
                "significant_holm": c.significant_holm,
                "cohens_d": _fmt_effect_size(c.cohens_d),
                "lodo": _fmt_ci_or_lodo(c.lodo_min_delta, c.lodo_max_delta),
            }
        )

    legacy_rows: list[dict[str, Any]] = []
    if not rows and result.significance_results:
        for sr in result.significance_results:
            legacy_rows.append(
                {
                    "metric": sr.metric_name,
                    "baseline": sr.strategy_b.value,
                    "mean_a": _fmt_rate(sr.mean_a),
                    "mean_b": _fmt_rate(sr.mean_b),
                    "p_value": f"{sr.p_value:.4f}",
                    "significant": sr.significant,
                    "effect_size": _fmt_effect_size(sr.effect_size),
                    "n_queries": sr.n_queries,
                }
            )

    return {
        "comparisons_by_model": dict(by_model),
        "has_comparisons": bool(by_model),
        "legacy_significance_rows": legacy_rows,
        "has_legacy_significance_fallback": bool(legacy_rows),
    }


def _structural_bar_chart(result: BenchmarkResult) -> str:
    """Grouped bar chart comparing gold-scored structural metrics across strategies."""
    if not result.structural_metrics:
        return ""

    aggregated = aggregate_gold_structural_metrics(result.structural_metrics)
    strategies = [a.strategy.value for a in aggregated]

    metrics_config = [
        ("Located", [a.localization_rate for a in aggregated]),
        ("1 - Leaf Frag.", [1.0 - a.clause_fragmentation_rate for a in aggregated]),
        ("1 - Over-merge", [1.0 - a.top_level_over_merge_rate for a in aggregated]),
        ("Def. Attachment", [a.definition_attachment_recall for a in aggregated]),
    ]

    fig = go.Figure()
    for label, values in metrics_config:
        fig.add_trace(
            go.Bar(
                name=label,
                x=strategies,
                y=values,
                text=[f"{v:.3f}" for v in values],
                textposition="auto",
            )
        )

    fig.update_layout(
        title="Gold-Scored Structural Quality by Strategy (higher = better)",
        barmode="group",
        yaxis_title="Score",
        yaxis_range=[0, 1.05],
        template="plotly_white",
        height=500,
    )

    return str(pio.to_html(fig, full_html=False, include_plotlyjs="cdn"))


def _structural_heatmap(result: BenchmarkResult) -> str:
    """Heatmap: strategies x documents, coloured by a composite of gold metrics.

    The composite only uses metrics that are never `None` (localization, fragmentation,
    over-merge, definition attachment) so it is always computable, unlike heading/cross-ref
    recall which are `None` for strategies that expose neither.
    """
    if not result.structural_metrics:
        return ""

    by_sd: dict[tuple[str, str], GoldStructuralMetrics] = {}
    for sm in result.structural_metrics:
        by_sd[(sm.strategy.value, sm.document_id)] = sm

    strategies = sorted({sm.strategy.value for sm in result.structural_metrics})
    documents = sorted({sm.document_id for sm in result.structural_metrics})

    z: list[list[float]] = []
    for strat in strategies:
        row = []
        for doc in documents:
            sm_entry = by_sd.get((strat, doc))
            if sm_entry:
                composite = (
                    sm_entry.localization_rate
                    + (1 - sm_entry.clause_fragmentation_rate)
                    + (1 - sm_entry.top_level_over_merge_rate)
                    + sm_entry.definition_attachment_recall
                ) / 4.0
                row.append(round(composite, 3))
            else:
                row.append(0.0)
        z.append(row)

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=documents,
            y=strategies,
            colorscale="RdYlGn",
            zmin=0,
            zmax=1,
            text=z,
            texttemplate="%{text:.3f}",
        )
    )

    fig.update_layout(
        title="Gold-Scored Structural Composite Score (strategy x document)",
        height=400,
        template="plotly_white",
    )

    return str(pio.to_html(fig, full_html=False, include_plotlyjs=False))


def _retrieval_bar_chart(result: BenchmarkResult, model_name: str) -> str:
    """Grouped bar chart: strategies x retrieval metrics for one model."""
    model_metrics = [
        rm for rm in result.retrieval_metrics if rm.embedding_model.value == model_name
    ]
    aggregated = aggregate_retrieval_metrics(model_metrics)
    strategies = [a.strategy.value for a in aggregated]

    metrics_config = [
        ("P@5", [a.precision_at_5 for a in aggregated]),
        ("R@10", [a.recall_at_10 for a in aggregated]),
        ("MRR", [a.mrr for a in aggregated]),
        ("NDCG@10", [a.ndcg_at_10 for a in aggregated]),
    ]

    fig = go.Figure()
    for label, values in metrics_config:
        fig.add_trace(
            go.Bar(
                name=label,
                x=strategies,
                y=values,
                text=[f"{v:.3f}" for v in values],
                textposition="auto",
            )
        )

    fig.update_layout(
        title=f"Retrieval Metrics by Strategy ({model_name})",
        barmode="group",
        yaxis_title="Score",
        yaxis_range=[0, 1.05],
        template="plotly_white",
        height=450,
    )
    return str(pio.to_html(fig, full_html=False, include_plotlyjs=False))


def _model_comparison_chart(result: BenchmarkResult) -> str:
    """Compare embedding models: for each strategy, show NDCG@10 across models."""
    by_sm: dict[tuple[str, str], list[Any]] = defaultdict(list)
    for rm in result.retrieval_metrics:
        key = (rm.strategy.value, rm.embedding_model.value)
        by_sm[key].append(rm)

    strategies = sorted({rm.strategy.value for rm in result.retrieval_metrics})
    models = sorted({rm.embedding_model.value for rm in result.retrieval_metrics})

    fig = go.Figure()
    for model in models:
        values = []
        for strat in strategies:
            items = by_sm.get((strat, model), [])
            avg = sum(m.ndcg_at_10 for m in items) / len(items) if items else 0
            values.append(avg)
        fig.add_trace(
            go.Bar(
                name=model,
                x=strategies,
                y=values,
                text=[f"{v:.3f}" for v in values],
                textposition="auto",
            )
        )

    fig.update_layout(
        title="NDCG@10 by Strategy and Embedding Model",
        barmode="group",
        yaxis_title="NDCG@10",
        yaxis_range=[0, 1.05],
        template="plotly_white",
        height=450,
    )
    return str(pio.to_html(fig, full_html=False, include_plotlyjs=False))


def _drm_chart(result: BenchmarkResult) -> str:
    """Bar chart showing mean Document Retrieval Mismatch rate per strategy.

    Uses `RetrievalMetrics.drm_rate` (the per-query mismatch fraction), not `drm_hit` — with
    one shared index across all documents, `drm_hit` is `True` for essentially every query
    and carries no information on its own (see the field's docstring in `scaffolder.models`).
    """
    by_strategy: dict[str, list[Any]] = defaultdict(list)
    for rm in result.retrieval_metrics:
        by_strategy[rm.strategy.value].append(rm)

    strategies = sorted(by_strategy.keys())
    drm_rates = []
    for strat in strategies:
        items = by_strategy[strat]
        rate = sum(m.drm_rate for m in items) / len(items) * 100 if items else 0
        drm_rates.append(rate)

    fig = go.Figure(
        go.Bar(
            x=strategies,
            y=drm_rates,
            text=[f"{v:.1f}%" for v in drm_rates],
            textposition="auto",
            marker_color=["#16a34a" if v < 10 else "#dc2626" for v in drm_rates],
        )
    )
    fig.update_layout(
        title="Document Retrieval Mismatch Rate (lower = better)",
        yaxis_title="DRM Rate (%)",
        template="plotly_white",
        height=350,
    )
    return str(pio.to_html(fig, full_html=False, include_plotlyjs=False))
