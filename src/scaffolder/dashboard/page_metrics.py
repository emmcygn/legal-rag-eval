"""Page 3: Metrics Dashboard — aggregate results and comparisons."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import streamlit as st

from scaffolder.models import StrategyName

if TYPE_CHECKING:
    from collections.abc import Callable

# Fixed per-strategy colours so the same strategy always draws in the same colour
# across the structural and retrieval charts. Keyed by StrategyName.value so a
# strategy not in this map (e.g. the legacy, unsized "rcts") still renders, just
# in the fallback grey used at each call site.
_STRATEGY_COLORS: dict[str, str] = {
    StrategyName.LEXICHUNK.value: "#2196F3",
    StrategyName.LEXICHUNK_CONTEXTUAL.value: "#03A9F4",
    StrategyName.RCTS_512.value: "#FF9800",
    StrategyName.RCTS_1024.value: "#F57C00",
    StrategyName.SENTENCE_SPLIT.value: "#4CAF50",
    StrategyName.FIXED_SIZE.value: "#9C27B0",
}


def _load_results() -> dict[str, Any] | None:
    """Load benchmark results from file or session state."""
    if "benchmark_result" in st.session_state:
        result: dict[str, Any] = st.session_state["benchmark_result"]
        return result

    results_dir = Path("results")
    if results_dir.exists():
        json_files = sorted(results_dir.glob("*.json"), reverse=True)
        if json_files:
            with open(json_files[0]) as f:
                data = json.load(f)
            st.session_state["benchmark_result"] = data
            return data  # type: ignore[no-any-return]

    return None


# Fields on GoldStructuralMetrics (scaffolder.models) that this page averages and
# charts. clause_fragmentation_rate replaces nothing (same name, new meaning: scored
# against gold clause spans, not LexiChunk's own parse); definition_attachment_recall
# and xref_target_recall replace the old definition_preservation_rate /
# cross_ref_resolution_rate; there is no gold equivalent of hierarchy_depth_retained,
# so localization_rate (how much of a strategy's output could be scored at all) takes
# its place as a headline number instead.
_GOLD_METRIC_KEYS = (
    "clause_fragmentation_rate",
    "definition_attachment_recall",
    "xref_target_recall",
    "localization_rate",
)


def _compute_strategy_averages(
    structural_results: list[dict[str, Any]],
) -> dict[str, dict[str, float | None]]:
    """Compute per-strategy averages across all documents.

    A metric that is ``None`` on a given document (for example ``xref_target_recall``
    when that document has no gold cross-references) is excluded from the average
    rather than treated as 0. A strategy with no valid values for a metric at all
    reports ``None`` for it -- render that as "n/a", never as 0.
    """
    totals: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for r in structural_results:
        s = r["strategy"]
        for key in _GOLD_METRIC_KEYS:
            value = r.get(key)
            if value is not None:
                totals[s][key].append(value)

    averages: dict[str, dict[str, float | None]] = {}
    for strategy, metrics in totals.items():
        strategy_avgs: dict[str, float | None] = {
            key: (sum(values) / len(values) if values else None) for key, values in metrics.items()
        }
        for key in _GOLD_METRIC_KEYS:
            strategy_avgs.setdefault(key, None)
        averages[strategy] = strategy_avgs

    return averages


def _format_rate(value: float | None, fmt: str = ".1%") -> str:
    """Format a metric value, rendering None as "n/a" rather than 0."""
    return "n/a" if value is None else f"{value:{fmt}}"


def render_page() -> None:
    """Render the metrics dashboard page."""
    st.title("Metrics Dashboard")
    st.markdown(
        "Aggregate benchmark results comparing LexiChunk against baselines. "
        "Load results from a previous benchmark run or upload a results JSON."
    )

    # --- Data Source ---
    col1, col2 = st.columns([3, 1])
    with col1:
        data_source = st.radio(
            "Data source",
            options=["Latest results file", "Upload JSON"],
            horizontal=True,
            key="metrics_source",
        )
    with col2:
        if st.button("Refresh", key="metrics_refresh"):
            st.session_state.pop("benchmark_result", None)
            st.rerun()

    result_data: dict[str, Any] | None = None

    if data_source == "Latest results file":
        result_data = _load_results()
        if result_data is None:
            st.warning(
                "No results found. Run a benchmark first: `make benchmark` "
                "or upload a results JSON file."
            )
            return
    else:
        uploaded = st.file_uploader("Upload results JSON", type=["json"])
        if uploaded:
            result_data = json.loads(uploaded.getvalue().decode("utf-8"))
            st.session_state["benchmark_result"] = result_data
        else:
            st.info("Upload a benchmark results JSON to view metrics.")
            return

    # --- Headline Metrics ---
    _render_headline_metrics(result_data)

    # --- Structural Comparison ---
    st.markdown("---")
    _render_structural_comparison(result_data)

    # --- Retrieval Comparison ---
    if result_data.get("retrieval_metrics"):
        st.markdown("---")
        _render_retrieval_comparison(result_data)

        # --- Embedding Model Comparison ---
        models = {r["embedding_model"] for r in result_data["retrieval_metrics"]}
        if len(models) > 1:
            st.markdown("---")
            _render_model_comparison(result_data)


def _best_baseline(
    baselines: dict[str, dict[str, float | None]],
    key: str,
    pick: Callable[[list[float]], float],
) -> float | None:
    """Return ``pick`` (``min`` or ``max``) of a metric across baselines, skipping any
    baseline where that metric is ``None`` ("n/a"). ``None`` if none have a value."""
    values = [value for m in baselines.values() if (value := m.get(key)) is not None]
    return pick(values) if values else None


def _diff(a: float | None, b: float | None) -> float | None:
    """``a - b``, or ``None`` if either operand is missing ("n/a")."""
    if a is None or b is None:
        return None
    return a - b


def _render_headline_metrics(data: dict[str, Any]) -> None:
    """Render headline metric cards showing LexiChunk's improvements."""
    st.subheader("Headline Results")

    structural = data.get("structural_metrics", [])
    if not structural:
        st.warning("No structural results found.")
        return

    avgs = _compute_strategy_averages(structural)
    lexi = avgs.get("lexichunk", {})

    baselines = {k: v for k, v in avgs.items() if k != "lexichunk"}
    if not baselines:
        st.warning("No baseline strategies found in results.")
        return

    # Clause fragmentation (lower is better): improvement = baseline - lexi.
    best_baseline_frag = _best_baseline(baselines, "clause_fragmentation_rate", min)
    frag_improvement = _diff(best_baseline_frag, lexi.get("clause_fragmentation_rate"))

    # Definition attachment recall (higher is better): improvement = lexi - baseline.
    best_baseline_def = _best_baseline(baselines, "definition_attachment_recall", max)
    def_improvement = _diff(lexi.get("definition_attachment_recall"), best_baseline_def)

    # Cross-ref target recall (higher is better; n/a for strategies emitting no refs).
    best_baseline_xref = _best_baseline(baselines, "xref_target_recall", max)
    xref_improvement = _diff(lexi.get("xref_target_recall"), best_baseline_xref)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Clause Fragmentation",
            _format_rate(lexi.get("clause_fragmentation_rate")),
            delta=f"-{frag_improvement:.1%}" if frag_improvement and frag_improvement > 0 else None,
            delta_color="normal",
            help="Lower is better. Shows how often gold clauses are split across chunks.",
        )
    with col2:
        st.metric(
            "Definition Attachment Recall",
            _format_rate(lexi.get("definition_attachment_recall")),
            delta=f"+{def_improvement:.1%}" if def_improvement and def_improvement > 0 else None,
            help="Higher is better. Shows how well defined terms are preserved.",
        )
    with col3:
        st.metric(
            "Cross-Ref Target Recall",
            _format_rate(lexi.get("xref_target_recall")),
            delta=f"+{xref_improvement:.1%}" if xref_improvement and xref_improvement > 0 else None,
            help="Higher is better; n/a for strategies that emit no cross-references.",
        )
    with col4:
        st.metric(
            "Localization Rate",
            _format_rate(lexi.get("localization_rate")),
            help="Share of this strategy's chunks that could be located in the document "
            "at all -- how much of its output the other metrics could even score.",
        )

    if data.get("retrieval_metrics"):
        _render_retrieval_headlines(data)


def _render_retrieval_headlines(data: dict[str, Any]) -> None:
    """Render retrieval metric headlines."""
    results = data["retrieval_metrics"]
    strategy_mrr: dict[str, list[float]] = defaultdict(list)
    strategy_ndcg: dict[str, list[float]] = defaultdict(list)
    strategy_drm: dict[str, list[bool]] = defaultdict(list)

    for r in results:
        strategy_mrr[r["strategy"]].append(r["mrr"])
        strategy_ndcg[r["strategy"]].append(r["ndcg_at_10"])
        strategy_drm[r["strategy"]].append(r.get("drm_hit", False))

    lc_mrr_vals = strategy_mrr.get("lexichunk", [])
    lc_ndcg_vals = strategy_ndcg.get("lexichunk", [])
    lc_drm_vals = strategy_drm.get("lexichunk", [])

    lexi_mrr = sum(lc_mrr_vals) / max(len(lc_mrr_vals), 1)
    lexi_ndcg = sum(lc_ndcg_vals) / max(len(lc_ndcg_vals), 1)
    lexi_drm = sum(lc_drm_vals) / max(len(lc_drm_vals), 1)

    st.markdown("")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("LexiChunk MRR", f"{lexi_mrr:.3f}")
    with col2:
        st.metric("LexiChunk NDCG@10", f"{lexi_ndcg:.3f}")
    with col3:
        st.metric(
            "LexiChunk DRM Rate",
            f"{lexi_drm:.1%}",
            help="Document Retrieval Mismatch — lower is better",
        )


def _render_structural_comparison(data: dict[str, Any]) -> None:
    """Render structural metrics comparison as a grouped bar chart."""
    import plotly.graph_objects as go

    st.subheader("Structural Metrics by Strategy")

    structural = data.get("structural_metrics", [])
    if not structural:
        st.info("No structural results available.")
        return

    avgs = _compute_strategy_averages(structural)
    strategies = sorted(avgs.keys())

    metrics = [
        ("Clause Preservation", "clause_fragmentation_rate", True),
        ("Definition Attachment Recall", "definition_attachment_recall", False),
        ("Cross-Ref Target Recall", "xref_target_recall", False),
    ]

    fig = go.Figure()
    for strategy in strategies:
        values = []
        labels = []
        for label, key, invert in metrics:
            val = avgs[strategy].get(key)
            if val is None:
                # No gold-scorable data for this metric/strategy -- leave a gap
                # instead of drawing a misleading zero-height bar.
                continue
            if invert:
                val = 1 - val
            values.append(val)
            labels.append(label)

        fig.add_trace(
            go.Bar(
                name=strategy,
                x=labels,
                y=values,
                marker_color=_STRATEGY_COLORS.get(strategy, "#757575"),
            )
        )

    fig.update_layout(
        barmode="group",
        yaxis={"range": [0, 1.05], "title": "Rate"},
        height=400,
        margin={"l": 40, "r": 20, "t": 20, "b": 40},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )

    st.plotly_chart(fig, use_container_width=True)


def _render_retrieval_comparison(data: dict[str, Any]) -> None:
    """Render retrieval metrics comparison."""
    import plotly.graph_objects as go

    st.subheader("Retrieval Metrics by Strategy")

    results = data["retrieval_metrics"]
    strategy_metrics: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for r in results:
        s = r["strategy"]
        strategy_metrics[s]["P@5"].append(r.get("precision_at_5", 0))
        strategy_metrics[s]["MRR"].append(r["mrr"])
        strategy_metrics[s]["NDCG@10"].append(r["ndcg_at_10"])

    strategies = sorted(strategy_metrics.keys())
    metric_names = ["P@5", "MRR", "NDCG@10"]

    fig = go.Figure()
    for strategy in strategies:
        avgs = [
            sum(strategy_metrics[strategy][m]) / max(len(strategy_metrics[strategy][m]), 1)
            for m in metric_names
        ]
        fig.add_trace(
            go.Bar(
                name=strategy,
                x=metric_names,
                y=avgs,
                marker_color=_STRATEGY_COLORS.get(strategy, "#757575"),
            )
        )

    fig.update_layout(
        barmode="group",
        yaxis={"range": [0, 1.05], "title": "Score"},
        height=400,
        margin={"l": 40, "r": 20, "t": 20, "b": 40},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )

    st.plotly_chart(fig, use_container_width=True)


def _render_model_comparison(data: dict[str, Any]) -> None:
    """Render embedding model comparison as a heatmap."""
    import plotly.graph_objects as go

    st.subheader("Embedding Model Comparison")
    st.markdown("Heatmap showing NDCG@10 for each strategy x model combination.")

    results = data["retrieval_metrics"]

    models = sorted({r["embedding_model"] for r in results})
    strategies = sorted({r["strategy"] for r in results})

    matrix: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        matrix[r["strategy"]][r["embedding_model"]].append(r["ndcg_at_10"])

    z = []
    for strategy in strategies:
        row = []
        for model in models:
            vals = matrix[strategy][model]
            row.append(sum(vals) / max(len(vals), 1) if vals else 0)
        z.append(row)

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=models,
            y=strategies,
            colorscale="Blues",
            text=[[f"{v:.3f}" for v in row] for row in z],
            texttemplate="%{text}",
            textfont={"size": 14},
            zmin=0,
            zmax=1,
            colorbar={"title": "NDCG@10"},
        )
    )

    fig.update_layout(
        height=300,
        margin={"l": 120, "r": 20, "t": 20, "b": 60},
        xaxis={"title": "Embedding Model"},
        yaxis={"title": "Strategy"},
    )

    st.plotly_chart(fig, use_container_width=True)
