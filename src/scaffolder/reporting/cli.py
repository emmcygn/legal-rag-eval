"""CLI output using rich tables."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.console import Console
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from scaffolder.models import (
        BenchmarkResult,
        RetrievalMetrics,
        SignificanceResult,
        StructuralMetrics,
    )


def _best_worst(values: list[float], higher_is_better: bool) -> tuple[float, float]:
    """Return (best_value, worst_value) from a list of floats."""
    if not values:
        return 0.0, 0.0
    if higher_is_better:
        return max(values), min(values)
    return min(values), max(values)


def _color_value(value: float, best: float, worst: float, fmt: str = ".3f") -> Text:
    """Color a value green if best, red if worst, default otherwise."""
    text = f"{value:{fmt}}"
    if abs(value - best) < 1e-9:
        return Text(text, style="bold green")
    if abs(value - worst) < 1e-9:
        return Text(text, style="bold red")
    return Text(text)


def render_structural_table(
    results: list[StructuralMetrics],
    console: Console | None = None,
) -> None:
    """Render structural metrics as a colour-coded rich table.

    Green = best value per metric, Red = worst value per metric.
    """
    if console is None:
        console = Console()

    if not results:
        console.print("[yellow]No structural results to display.[/yellow]")
        return

    docs = sorted({r.document_id for r in results})

    for doc_id in docs:
        doc_results = [r for r in results if r.document_id == doc_id]
        if not doc_results:
            continue

        table = Table(
            title=f"Structural Metrics — {doc_id}",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Strategy", style="bold")
        table.add_column("Clause Frag.", justify="right")
        table.add_column("Def. Preserv.", justify="right")
        table.add_column("XRef Resol.", justify="right")
        table.add_column("Hierarchy", justify="right")
        table.add_column("Size CV", justify="right")
        table.add_column("Avg Chars", justify="right")
        table.add_column("Chunks", justify="right")

        frag_vals = [r.clause_fragmentation_rate for r in doc_results]
        def_vals = [r.definition_preservation_rate for r in doc_results]
        xref_vals = [r.cross_ref_resolution_rate for r in doc_results]
        hier_vals = [r.hierarchy_depth_retained for r in doc_results]

        frag_best, frag_worst = _best_worst(frag_vals, higher_is_better=False)
        def_best, def_worst = _best_worst(def_vals, higher_is_better=True)
        xref_best, xref_worst = _best_worst(xref_vals, higher_is_better=True)
        hier_best, hier_worst = _best_worst(hier_vals, higher_is_better=True)

        for r in doc_results:
            table.add_row(
                r.strategy.value,
                _color_value(r.clause_fragmentation_rate, frag_best, frag_worst),
                _color_value(r.definition_preservation_rate, def_best, def_worst),
                _color_value(r.cross_ref_resolution_rate, xref_best, xref_worst),
                _color_value(r.hierarchy_depth_retained, hier_best, hier_worst, ".2f"),
                f"{r.chunk_size_cv:.3f}",
                f"{r.avg_chunk_chars:.0f}",
                str(r.chunk_count),
            )

        console.print(table)
        console.print()


def render_summary_header(
    result: BenchmarkResult,
    console: Console | None = None,
) -> None:
    """Render a summary header with config and timestamp."""
    if console is None:
        console = Console()

    console.print()
    console.rule("[bold blue]Scaffolder Benchmark Results[/bold blue]")
    console.print(f"  Timestamp: {result.timestamp}")
    console.print(f"  Strategies: {', '.join(s.value for s in result.strategies)}")
    console.print(f"  Documents: {len(result.documents)}")
    if result.models:
        console.print(f"  Embedding models: {', '.join(m.value for m in result.models)}")
    console.rule()
    console.print()


def render_aggregate_summary(
    result: BenchmarkResult,
    console: Console | None = None,
) -> None:
    """Render aggregate summary comparing strategies across all documents."""
    if console is None:
        console = Console()

    if not result.structural_metrics:
        return

    # Compute per-strategy averages
    from collections import defaultdict

    strategy_sums: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    strategy_counts: dict[str, int] = defaultdict(int)

    for m in result.structural_metrics:
        key = m.strategy.value
        strategy_sums[key]["frag"] += m.clause_fragmentation_rate
        strategy_sums[key]["def"] += m.definition_preservation_rate
        strategy_sums[key]["xref"] += m.cross_ref_resolution_rate
        strategy_sums[key]["hier"] += m.hierarchy_depth_retained
        strategy_counts[key] += 1

    table = Table(
        title="Aggregate Summary (averaged across all documents)",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Strategy", style="bold")
    table.add_column("Avg Clause Frag.", justify="right")
    table.add_column("Avg Def. Preserv.", justify="right")
    table.add_column("Avg XRef Resol.", justify="right")
    table.add_column("Avg Hierarchy", justify="right")

    for strategy in sorted(strategy_sums.keys()):
        n = strategy_counts[strategy]
        sums = strategy_sums[strategy]
        table.add_row(
            strategy,
            f"{sums['frag'] / n:.3f}",
            f"{sums['def'] / n:.3f}",
            f"{sums['xref'] / n:.3f}",
            f"{sums['hier'] / n:.2f}",
        )

    console.print(table)
    console.print()


def _get_significance_marker(
    strategy: str,
    metric_attr: str,
    significance_results: list[SignificanceResult],
) -> str:
    """Get significance marker for a strategy/metric pair.

    Returns "**" if p < 0.01, "*" if p < 0.05, "" otherwise.
    Compares the given strategy against LexiChunk (the reference).
    """
    if not significance_results:
        return ""

    # No marker for the reference strategy
    if strategy == "lexichunk":
        return ""

    for sr in significance_results:
        if sr.strategy_b.value == strategy and sr.metric_name == metric_attr:
            if sr.p_value < 0.01:
                return "**"
            if sr.p_value < 0.05:
                return "*"
    return ""


def render_retrieval_table(
    metrics: list[RetrievalMetrics],
    significance_results: list[SignificanceResult] | None = None,
    console: Console | None = None,
) -> None:
    """Render retrieval metrics as a colour-coded rich table.

    Shows per-strategy aggregate retrieval metrics (averaged across all queries).
    Adds significance markers when statistical tests are provided.

    Significance markers:
    - * p < 0.05 (significant)
    - ** p < 0.01 (highly significant)
    """
    if console is None:
        console = Console()

    if not metrics:
        console.print("[yellow]No retrieval results to display (run with --embed).[/yellow]")
        return

    sig = significance_results or []
    metric_versions = ", ".join(sorted({metric.ndcg_metric for metric in metrics}))
    console.print(f"NDCG compatibility field uses: {metric_versions}")
    console.print(
        "Do not compare different metric versions or interpret assignment scores as chunk NDCG."
    )

    # Group by embedding model
    models = sorted({m.embedding_model.value for m in metrics})

    for model in models:
        model_metrics = [m for m in metrics if m.embedding_model.value == model]

        # Group by strategy and compute averages
        strategies = sorted({m.strategy.value for m in model_metrics})
        strategy_avgs: dict[str, dict[str, float]] = {}

        for strategy in strategies:
            strat_metrics = [m for m in model_metrics if m.strategy.value == strategy]
            n = len(strat_metrics)
            if n == 0:
                continue

            strategy_avgs[strategy] = {
                "P@1": sum(m.precision_at_1 for m in strat_metrics) / n,
                "P@3": sum(m.precision_at_3 for m in strat_metrics) / n,
                "P@5": sum(m.precision_at_5 for m in strat_metrics) / n,
                "P@10": sum(m.precision_at_10 for m in strat_metrics) / n,
                "MRR": sum(m.mrr for m in strat_metrics) / n,
                "NDCG@10": sum(m.ndcg_at_10 for m in strat_metrics) / n,
                "DRM%": sum(1 for m in strat_metrics if m.drm_hit) / n,
                "n_queries": float(n),
            }

        # Build table
        table = Table(
            title=f"Retrieval Metrics — {model}",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Strategy", style="bold")
        table.add_column("P@1", justify="right")
        table.add_column("P@3", justify="right")
        table.add_column("P@5", justify="right")
        table.add_column("P@10", justify="right")
        table.add_column("MRR", justify="right")
        table.add_column("NDCG@10", justify="right")
        table.add_column("DRM%", justify="right")
        table.add_column("Queries", justify="right")

        # Metric attr map for significance lookups
        metric_attr_map = {
            "P@1": "precision_at_1",
            "P@3": "precision_at_3",
            "P@5": "precision_at_5",
            "P@10": "precision_at_10",
            "MRR": "mrr",
            "NDCG@10": "ndcg_at_10",
        }

        # Compute best/worst for colour coding
        metric_keys = ["P@1", "P@3", "P@5", "P@10", "MRR", "NDCG@10"]
        best_worst: dict[str, tuple[float, float]] = {}
        for mk in metric_keys:
            vals = [a[mk] for a in strategy_avgs.values()]
            best_worst[mk] = _best_worst(vals, higher_is_better=True)

        # DRM is lower-is-better
        drm_vals = [a["DRM%"] for a in strategy_avgs.values()]
        best_worst["DRM%"] = _best_worst(drm_vals, higher_is_better=False)

        for strategy in strategies:
            avgs = strategy_avgs.get(strategy)
            if avgs is None:
                continue

            row: list[Text | str] = [strategy]
            for mk in metric_keys:
                value = avgs[mk]
                best, worst = best_worst[mk]
                text = _color_value(value, best, worst)

                # Add significance marker
                marker = _get_significance_marker(strategy, metric_attr_map[mk], sig)
                if marker:
                    text.append(f" {marker}", style="bold yellow")
                row.append(text)

            # DRM rate (lower is better)
            drm_best, drm_worst = best_worst["DRM%"]
            drm_text = _color_value(avgs["DRM%"], drm_best, drm_worst)
            row.append(drm_text)

            row.append(str(int(avgs["n_queries"])))
            table.add_row(*row)

        console.print(table)
        console.print()


def render_benchmark(
    result: BenchmarkResult,
    console: Console | None = None,
) -> None:
    """Full CLI benchmark output: header + structural + retrieval + summary."""
    if console is None:
        console = Console()

    render_summary_header(result, console)
    render_structural_table(result.structural_metrics, console)

    # Retrieval metrics (only if embedding was run)
    if result.retrieval_metrics:
        render_retrieval_table(
            result.retrieval_metrics,
            significance_results=result.significance_results,
            console=console,
        )

        # Significance legend
        if result.significance_results:
            console.print(
                "[dim]Significance: * p<0.05, ** p<0.01 "
                "(paired t-test, LexiChunk vs baseline)[/dim]"
            )
            console.print()

    render_aggregate_summary(result, console)
