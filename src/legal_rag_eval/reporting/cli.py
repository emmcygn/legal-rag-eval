"""CLI output using rich tables.

Reports the metrics the audit says are defensible: :class:`~legal_rag_eval.models.
GoldStructuralMetrics` (scored against hand-checked span annotations, not LexiChunk's own
parse) and, when present, :class:`~legal_rag_eval.metrics.statistical.ComparisonResult`
(bootstrap CI, Holm-corrected p, Wilcoxon, effect sizes, leave-one-document-out) instead of
a bare uncorrected paired t-test.

This module deliberately does NOT import :mod:`legal_rag_eval.metrics.statistical` — the
``ComparisonResult`` shape is checked structurally via :class:`ComparisonLike` below, since
``BenchmarkResult.comparisons`` is typed as ``list[object]`` and reporting has no need to
depend on the concrete statistics implementation.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from rich.console import Console
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from legal_rag_eval.models import (
        BenchmarkResult,
        GoldStructuralMetrics,
        LegacyStructuralMetrics,
        RetrievalMetrics,
        SignificanceResult,
        StrategyName,
    )


@runtime_checkable
class ComparisonLike(Protocol):
    """Structural shape of :class:`legal_rag_eval.metrics.statistical.ComparisonResult`.

    ``BenchmarkResult.comparisons`` is ``list[object]`` on purpose — reporting only needs
    this shape, not the concrete dataclass, so it never has to import the statistics
    module. Any object with these attributes (the real ``ComparisonResult``, or a
    reconstruction of one) renders correctly.
    """

    metric_name: str
    strategy_a: StrategyName
    strategy_b: StrategyName
    embedding_model: str
    n: int
    mean_a: float
    mean_b: float
    delta: float
    ci_low: float
    ci_high: float
    t_statistic: float
    p_value_t: float
    p_value_wilcoxon: float
    p_value_holm: float
    significant_holm: bool
    cohens_d: float
    rank_biserial: float
    n_documents: int
    lodo_min_delta: float
    lodo_max_delta: float
    lodo_worst_document: str


# -- Small numeric helpers ---------------------------------------------------


def _mean(values: Iterable[float]) -> float:
    """Arithmetic mean of ``values``, or ``0.0`` for an empty iterable."""
    materialized = list(values)
    return sum(materialized) / len(materialized) if materialized else 0.0


def _mean_optional(values: Iterable[float | None]) -> float | None:
    """Mean of the non-``None`` values, or ``None`` if every value is ``None``."""
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else None


def _best_worst(values: list[float], higher_is_better: bool) -> tuple[float, float]:
    """Return (best_value, worst_value) from a list of floats."""
    if not values:
        return 0.0, 0.0
    if higher_is_better:
        return max(values), min(values)
    return min(values), max(values)


def _best_worst_optional(
    values: list[float | None], higher_is_better: bool
) -> tuple[float | None, float | None]:
    """Like `_best_worst`, ignoring ``None`` entries; returns ``(None, None)`` if all are."""
    present = [v for v in values if v is not None]
    if not present:
        return None, None
    return _best_worst(present, higher_is_better)


def _color_value(value: float, best: float, worst: float, fmt: str = ".3f") -> Text:
    """Color a value green if best, red if worst, default otherwise."""
    text = f"{value:{fmt}}"
    if abs(value - best) < 1e-9:
        return Text(text, style="bold green")
    if abs(value - worst) < 1e-9:
        return Text(text, style="bold red")
    return Text(text)


def _color_optional(
    value: float | None,
    best: float | None,
    worst: float | None,
    fmt: str = ".3f",
) -> Text:
    """Like `_color_value`, but renders ``None`` as a plain, never-coloured "n/a"."""
    if value is None:
        return Text("n/a", style="dim")
    if best is None or worst is None:
        return Text(f"{value:{fmt}}")
    return _color_value(value, best, worst, fmt)


def _fmt_effect_size(value: float) -> str:
    """Format an effect size that may legitimately be +/-inf (never silently 0.0)."""
    if math.isinf(value):
        return "+inf" if value > 0 else "-inf"
    return f"{value:.3f}"


# -- Aggregation (macro-average over documents / queries, one row per strategy) ---------


@dataclass(frozen=True, slots=True)
class AggregatedGoldStructural:
    """One strategy's `GoldStructuralMetrics`, macro-averaged over documents."""

    strategy: StrategyName
    n_documents: int
    localization_rate: float
    clause_fragmentation_rate: float
    top_level_over_merge_rate: float
    sub_clause_grouping_rate: float
    heading_recall: float | None
    heading_precision: float | None
    definition_attachment_recall: float
    xref_target_recall: float | None
    xref_target_precision: float | None
    chunk_size_cv: float
    avg_chunk_chars: float
    chunk_count: float


def aggregate_gold_structural_metrics(
    results: Sequence[GoldStructuralMetrics],
) -> list[AggregatedGoldStructural]:
    """Macro-average `GoldStructuralMetrics` across documents, one row per strategy.

    "Macro-averaged" means the mean of each per-document rate, not a pooled/micro-average
    over raw counts — the same convention the generated README results table uses. A
    strategy whose per-document ``heading_recall``/``xref_target_*`` are all ``None`` (it
    exposes no headings or cross-references) aggregates to ``None``, never ``0.0``.
    """
    by_strategy: dict[StrategyName, list[GoldStructuralMetrics]] = defaultdict(list)
    for r in results:
        by_strategy[r.strategy].append(r)

    aggregated: list[AggregatedGoldStructural] = []
    for strategy in sorted(by_strategy, key=lambda s: s.value):
        rows = by_strategy[strategy]
        aggregated.append(
            AggregatedGoldStructural(
                strategy=strategy,
                n_documents=len(rows),
                localization_rate=_mean(r.localization_rate for r in rows),
                clause_fragmentation_rate=_mean(r.clause_fragmentation_rate for r in rows),
                top_level_over_merge_rate=_mean(r.top_level_over_merge_rate for r in rows),
                sub_clause_grouping_rate=_mean(r.sub_clause_grouping_rate for r in rows),
                heading_recall=_mean_optional(r.heading_recall for r in rows),
                heading_precision=_mean_optional(r.heading_precision for r in rows),
                definition_attachment_recall=_mean(r.definition_attachment_recall for r in rows),
                xref_target_recall=_mean_optional(r.xref_target_recall for r in rows),
                xref_target_precision=_mean_optional(r.xref_target_precision for r in rows),
                chunk_size_cv=_mean(r.chunk_size_cv for r in rows),
                avg_chunk_chars=_mean(r.avg_chunk_chars for r in rows),
                chunk_count=_mean(float(r.chunk_count) for r in rows),
            )
        )
    return aggregated


@dataclass(frozen=True, slots=True)
class AggregatedLegacyStructural:
    """One strategy's `LegacyStructuralMetrics`, macro-averaged over documents."""

    strategy: StrategyName
    n_documents: int
    clause_fragmentation_rate: float
    definition_preservation_rate: float
    cross_ref_resolution_rate: float
    hierarchy_depth_retained: float
    chunk_size_cv: float
    avg_chunk_chars: float
    chunk_count: float


def aggregate_legacy_structural_metrics(
    results: Sequence[LegacyStructuralMetrics],
) -> list[AggregatedLegacyStructural]:
    """Macro-average `LegacyStructuralMetrics` across documents, one row per strategy."""
    by_strategy: dict[StrategyName, list[LegacyStructuralMetrics]] = defaultdict(list)
    for r in results:
        by_strategy[r.strategy].append(r)

    aggregated: list[AggregatedLegacyStructural] = []
    for strategy in sorted(by_strategy, key=lambda s: s.value):
        rows = by_strategy[strategy]
        aggregated.append(
            AggregatedLegacyStructural(
                strategy=strategy,
                n_documents=len(rows),
                clause_fragmentation_rate=_mean(r.clause_fragmentation_rate for r in rows),
                definition_preservation_rate=_mean(r.definition_preservation_rate for r in rows),
                cross_ref_resolution_rate=_mean(r.cross_ref_resolution_rate for r in rows),
                hierarchy_depth_retained=_mean(r.hierarchy_depth_retained for r in rows),
                chunk_size_cv=_mean(r.chunk_size_cv for r in rows),
                avg_chunk_chars=_mean(r.avg_chunk_chars for r in rows),
                chunk_count=_mean(float(r.chunk_count) for r in rows),
            )
        )
    return aggregated


@dataclass(frozen=True, slots=True)
class AggregatedRetrieval:
    """One strategy's `RetrievalMetrics` (for one embedding model), meaned over queries."""

    strategy: StrategyName
    n_queries: int
    precision_at_1: float
    precision_at_3: float
    precision_at_5: float
    precision_at_10: float
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    recall_at_10: float
    mrr: float
    ndcg_at_10: float
    drm_rate: float


def aggregate_retrieval_metrics(
    results: Sequence[RetrievalMetrics],
) -> list[AggregatedRetrieval]:
    """Mean `RetrievalMetrics` over queries, one row per strategy.

    Callers should pre-filter ``results`` to a single embedding model — mixing models in
    one call would average across incomparable embedding spaces.
    """
    by_strategy: dict[StrategyName, list[RetrievalMetrics]] = defaultdict(list)
    for r in results:
        by_strategy[r.strategy].append(r)

    aggregated: list[AggregatedRetrieval] = []
    for strategy in sorted(by_strategy, key=lambda s: s.value):
        rows = by_strategy[strategy]
        aggregated.append(
            AggregatedRetrieval(
                strategy=strategy,
                n_queries=len(rows),
                precision_at_1=_mean(r.precision_at_1 for r in rows),
                precision_at_3=_mean(r.precision_at_3 for r in rows),
                precision_at_5=_mean(r.precision_at_5 for r in rows),
                precision_at_10=_mean(r.precision_at_10 for r in rows),
                recall_at_1=_mean(r.recall_at_1 for r in rows),
                recall_at_3=_mean(r.recall_at_3 for r in rows),
                recall_at_5=_mean(r.recall_at_5 for r in rows),
                recall_at_10=_mean(r.recall_at_10 for r in rows),
                mrr=_mean(r.mrr for r in rows),
                ndcg_at_10=_mean(r.ndcg_at_10 for r in rows),
                drm_rate=_mean(r.drm_rate for r in rows),
            )
        )
    return aggregated


# -- Structural tables --------------------------------------------------------


def render_structural_table(
    results: list[GoldStructuralMetrics],
    console: Console | None = None,
) -> None:
    """Render gold-scored structural metrics: one macro-averaged table, one row per strategy.

    Ground truth is hand-checked span annotations in ``gold/`` (see
    :class:`~legal_rag_eval.models.GoldStructuralMetrics`) — this is the metric set that should
    be cited as evidence. ``None`` fields (heading/cross-reference metrics for a strategy
    that exposes neither) render as "n/a" and are never colour-coded; ``Sub-clause Grp`` and
    ``Size CV``/``Avg Chars``/``Chunks`` are informational and are never colour-coded either.
    """
    if console is None:
        console = Console()

    if not results:
        console.print("[yellow]No structural results to display.[/yellow]")
        return

    aggregated = aggregate_gold_structural_metrics(results)
    n_documents = len({r.document_id for r in results})

    table = Table(
        title=f"Structural Metrics — gold-scored (n={n_documents} documents)",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Strategy", style="bold")
    table.add_column("Located", justify="right")
    table.add_column("Leaf Frag ↓", justify="right")
    table.add_column("Over-merge ↓", justify="right")
    table.add_column("Sub-clause Grp", justify="right")
    table.add_column("Head R", justify="right")
    table.add_column("Head P", justify="right")
    table.add_column("Def Attach ↑", justify="right")
    table.add_column("XRef R", justify="right")
    table.add_column("XRef P", justify="right")
    table.add_column("Size CV", justify="right")
    table.add_column("Avg Chars", justify="right")
    table.add_column("Chunks", justify="right")

    frag_best, frag_worst = _best_worst(
        [a.clause_fragmentation_rate for a in aggregated], higher_is_better=False
    )
    merge_best, merge_worst = _best_worst(
        [a.top_level_over_merge_rate for a in aggregated], higher_is_better=False
    )
    hr_best, hr_worst = _best_worst_optional(
        [a.heading_recall for a in aggregated], higher_is_better=True
    )
    hp_best, hp_worst = _best_worst_optional(
        [a.heading_precision for a in aggregated], higher_is_better=True
    )
    def_best, def_worst = _best_worst(
        [a.definition_attachment_recall for a in aggregated], higher_is_better=True
    )
    xr_best, xr_worst = _best_worst_optional(
        [a.xref_target_recall for a in aggregated], higher_is_better=True
    )
    xp_best, xp_worst = _best_worst_optional(
        [a.xref_target_precision for a in aggregated], higher_is_better=True
    )

    for a in aggregated:
        table.add_row(
            a.strategy.value,
            f"{a.localization_rate * 100:.1f}%",
            _color_value(a.clause_fragmentation_rate, frag_best, frag_worst),
            _color_value(a.top_level_over_merge_rate, merge_best, merge_worst),
            f"{a.sub_clause_grouping_rate:.3f}",
            _color_optional(a.heading_recall, hr_best, hr_worst),
            _color_optional(a.heading_precision, hp_best, hp_worst),
            _color_value(a.definition_attachment_recall, def_best, def_worst),
            _color_optional(a.xref_target_recall, xr_best, xr_worst),
            _color_optional(a.xref_target_precision, xp_best, xp_worst),
            f"{a.chunk_size_cv:.3f}",
            f"{a.avg_chunk_chars:.0f}",
            f"{a.chunk_count:.1f}",
        )

    console.print(table)
    console.print()


def render_legacy_structural_table(
    results: list[LegacyStructuralMetrics],
    console: Console | None = None,
) -> None:
    """Render the LEGACY, LexiChunk-self-graded structural metrics — only when non-empty.

    These are superseded and circular: LexiChunk's own parse defines the ground truth they
    are scored against, so LexiChunk cannot help but score well by construction, and the
    baselines' scores move whenever LexiChunk's parser changes even though the baselines
    themselves have not. They are not evidence about LexiChunk and are excluded from the
    default report — this table only appears when the caller explicitly opted in
    (``--legacy-metrics``) and ``result.legacy_structural_metrics`` is non-empty.
    """
    if console is None:
        console = Console()

    if not results:
        return

    aggregated = aggregate_legacy_structural_metrics(results)
    n_documents = len({r.document_id for r in results})

    console.print(
        "[bold yellow]SUPERSEDED[/bold yellow] — legacy structural metrics. LexiChunk's own "
        "parse defines this ground truth, so these rows are not evidence about LexiChunk, "
        "and the baselines' scores move whenever LexiChunk's parser changes even though the "
        "baselines have not."
    )
    table = Table(
        title=f"Legacy Structural Metrics — DO NOT cite as evidence (n={n_documents} documents)",
        show_header=True,
        header_style="bold red",
    )
    table.add_column("Strategy", style="bold")
    table.add_column("Clause Frag.", justify="right")
    table.add_column("Def. Preserv.", justify="right")
    table.add_column("XRef Resol.", justify="right")
    table.add_column("Hierarchy", justify="right")
    table.add_column("Size CV", justify="right")
    table.add_column("Avg Chars", justify="right")
    table.add_column("Chunks", justify="right")

    for a in aggregated:
        table.add_row(
            a.strategy.value,
            f"{a.clause_fragmentation_rate:.3f}",
            f"{a.definition_preservation_rate:.3f}",
            f"{a.cross_ref_resolution_rate:.3f}",
            f"{a.hierarchy_depth_retained:.3f}",
            f"{a.chunk_size_cv:.3f}",
            f"{a.avg_chunk_chars:.0f}",
            f"{a.chunk_count:.1f}",
        )

    console.print(table)
    console.print()


# -- Summary header -----------------------------------------------------------


def render_summary_header(
    result: BenchmarkResult,
    console: Console | None = None,
) -> None:
    """Render the run's provenance: seed, LexiChunk build, models, strategy parameters.

    A baseline whose chunk size is not stated cannot be compared to anything, so the
    resolved ``strategy_parameters`` (from ``result.config``) are always printed here.
    """
    if console is None:
        console = Console()

    console.print()
    console.rule("[bold blue]legal-rag-eval Benchmark Results[/bold blue]")
    # highlight=False: Rich's automatic highlighter otherwise splices ANSI codes into the
    # middle of version/commit/timestamp strings that merely look numeric (e.g. "0.8.0b1"
    # becomes "0.8" + escape codes + ".0b1"), corrupting them as substrings for any
    # downstream consumer that greps this plain-text output.
    console.print(f"  Timestamp: {result.timestamp}", highlight=False)
    console.print(
        f"  Seed: {result.seed if result.seed is not None else 'unknown'}", highlight=False
    )
    console.print(f"  LexiChunk version: {result.lexichunk_version or 'unknown'}", highlight=False)
    console.print(f"  LexiChunk commit: {result.lexichunk_commit or 'unknown'}", highlight=False)
    console.print(f"  Strategies: {', '.join(s.value for s in result.strategies)}", highlight=False)
    console.print(f"  Documents: {len(result.documents)}", highlight=False)
    if result.models:
        console.print(
            f"  Embedding models: {', '.join(m.value for m in result.models)}", highlight=False
        )

    strategy_parameters = result.config.get("strategy_parameters")
    if isinstance(strategy_parameters, dict) and strategy_parameters:
        console.print("  Strategy parameters:", highlight=False)
        for name in sorted(strategy_parameters):
            console.print(f"    {name}: {strategy_parameters[name]}", highlight=False)

    console.rule()
    console.print()


# -- Retrieval table ------------------------------------------------------------


def _get_significance_marker(
    strategy: str,
    metric_attr: str,
    significance_results: list[SignificanceResult],
) -> str:
    """Get significance marker for a strategy/metric pair.

    Returns "**" if p < 0.01, "*" if p < 0.05, "" otherwise. Compares the given strategy
    against LexiChunk (the reference). When ``significance_results`` was produced from
    `~legal_rag_eval.metrics.statistical.ComparisonResult.to_significance_result`, this ``p``
    is already Holm-adjusted; on older data it may be a raw, uncorrected p-value.
    """
    if not significance_results:
        return ""

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
    """Render retrieval metrics as a colour-coded rich table, one per embedding model.

    Shows per-strategy mean retrieval metrics over queries, with the query count (``n``)
    stated in the table title. Adds significance markers when statistical results are
    provided (``*`` p<0.05, ``**`` p<0.01 against LexiChunk).
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

    metric_attr_map = {
        "P@1": "precision_at_1",
        "P@3": "precision_at_3",
        "P@5": "precision_at_5",
        "P@10": "precision_at_10",
        "MRR": "mrr",
        "NDCG@10": "ndcg_at_10",
    }
    metric_keys = list(metric_attr_map)

    for model in models:
        model_metrics = [m for m in metrics if m.embedding_model.value == model]
        aggregated = aggregate_retrieval_metrics(model_metrics)
        n_queries = len({m.query_id for m in model_metrics})

        table = Table(
            title=f"Retrieval Metrics — {model} (n={n_queries} queries)",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Strategy", style="bold")
        for col in ("P@1", "P@3", "P@5", "P@10", "MRR", "NDCG@10", "DRM rate", "Queries"):
            table.add_column(col, justify="right")

        best_worst = {
            mk: _best_worst(
                [getattr(a, metric_attr_map[mk]) for a in aggregated], higher_is_better=True
            )
            for mk in metric_keys
        }
        drm_best, drm_worst = _best_worst([a.drm_rate for a in aggregated], higher_is_better=False)

        for a in aggregated:
            row: list[Text | str] = [a.strategy.value]
            for mk in metric_keys:
                value = getattr(a, metric_attr_map[mk])
                best, worst = best_worst[mk]
                text = _color_value(value, best, worst)
                marker = _get_significance_marker(a.strategy.value, metric_attr_map[mk], sig)
                if marker:
                    text.append(f" {marker}", style="bold yellow")
                row.append(text)
            row.append(_color_value(a.drm_rate, drm_best, drm_worst))
            row.append(str(a.n_queries))
            table.add_row(*row)

        console.print(table)
        console.print()


# -- Significance / comparisons table -------------------------------------------


def render_comparisons_table(
    comparisons: Sequence[object],
    console: Console | None = None,
) -> None:
    """Render the `ComparisonResult`-based statistics table, one table per embedding model.

    The absolute delta and its bootstrap 95% CI are always the headline column — never a
    percentage improvement. Rows are marked significant only via the Holm-adjusted p-value
    (``p_value_holm`` / ``significant_holm``), never the raw t/Wilcoxon p-values, which are
    shown for reference only.
    """
    if console is None:
        console = Console()

    rows = [c for c in comparisons if isinstance(c, ComparisonLike)]
    if not rows:
        console.print("[yellow]No comparisons to display.[/yellow]")
        return

    models = sorted({r.embedding_model for r in rows})
    for model in models:
        model_rows = sorted(
            (r for r in rows if r.embedding_model == model),
            key=lambda r: (r.metric_name, r.strategy_a.value, r.strategy_b.value),
        )

        table = Table(
            title=f"Strategy Comparisons — {model}",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Metric")
        table.add_column("A")
        table.add_column("B")
        table.add_column("n", justify="right")
        table.add_column("Mean A", justify="right")
        table.add_column("Mean B", justify="right")
        table.add_column("Δ", justify="right")
        table.add_column("95% CI", justify="right")
        table.add_column("p (t)", justify="right")
        table.add_column("p (Wilcoxon)", justify="right")
        table.add_column("p (Holm)", justify="right")
        table.add_column("Holm sig")
        table.add_column("Cohen's d", justify="right")
        table.add_column("LODO range", justify="right")

        for r in model_rows:
            ci = f"[{r.ci_low:+.3f}, {r.ci_high:+.3f}]"
            lodo = (
                "n/a"
                if math.isnan(r.lodo_min_delta)
                else f"[{r.lodo_min_delta:+.3f}, {r.lodo_max_delta:+.3f}]"
            )
            sig_text = (
                Text("YES", style="bold green") if r.significant_holm else Text("no", style="dim")
            )
            table.add_row(
                r.metric_name,
                r.strategy_a.value,
                r.strategy_b.value,
                str(r.n),
                f"{r.mean_a:.3f}",
                f"{r.mean_b:.3f}",
                f"{r.delta:+.3f}",
                ci,
                f"{r.p_value_t:.3f}",
                f"{r.p_value_wilcoxon:.3f}",
                f"{r.p_value_holm:.3f}",
                sig_text,
                _fmt_effect_size(r.cohens_d),
                lodo,
            )

        console.print(table)
        console.print()


def render_legacy_significance_table(
    results: list[SignificanceResult],
    console: Console | None = None,
) -> None:
    """Render the old single-test, uncorrected significance table.

    Kept only as a fallback for result objects that predate `ComparisonResult`
    (``result.comparisons`` is empty but ``result.significance_results`` is not) — always
    called from `render_significance_section`, which prints the uncorrected-p warning.
    """
    if console is None:
        console = Console()

    if not results:
        return

    table = Table(
        title="Significance (legacy, uncorrected p-values)",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Metric", style="bold")
    table.add_column("Baseline")
    table.add_column("Mean A", justify="right")
    table.add_column("Mean B", justify="right")
    table.add_column("p-value", justify="right")
    table.add_column("Significant?", justify="right")
    table.add_column("Effect size (d)", justify="right")
    table.add_column("n", justify="right")

    for sr in results:
        table.add_row(
            sr.metric_name,
            sr.strategy_b.value,
            f"{sr.mean_a:.3f}",
            f"{sr.mean_b:.3f}",
            f"{sr.p_value:.4f}",
            "YES" if sr.significant else "no",
            _fmt_effect_size(sr.effect_size),
            str(sr.n_queries),
        )

    console.print(table)
    console.print()


def render_significance_section(
    result: BenchmarkResult,
    console: Console | None = None,
) -> None:
    """Render the run's statistics: the rich comparisons table, or a warned legacy fallback."""
    if console is None:
        console = Console()

    if result.comparisons:
        render_comparisons_table(result.comparisons, console)
    elif result.significance_results:
        console.print(
            "[yellow]Warning: this run has no ComparisonResult data — showing single-test, "
            "uncorrected p-values with no bootstrap CI, no multiple-comparison correction, "
            "and no leave-one-document-out sensitivity.[/yellow]"
        )
        render_legacy_significance_table(result.significance_results, console)


# -- Top-level entry point -----------------------------------------------------


def render_benchmark(
    result: BenchmarkResult,
    console: Console | None = None,
) -> None:
    """Full CLI benchmark output: header + structural (+ legacy) + retrieval + significance."""
    if console is None:
        console = Console()

    render_summary_header(result, console)
    render_structural_table(result.structural_metrics, console)

    if result.legacy_structural_metrics:
        render_legacy_structural_table(result.legacy_structural_metrics, console)

    if result.retrieval_metrics:
        render_retrieval_table(
            result.retrieval_metrics,
            significance_results=result.significance_results,
            console=console,
        )

    if result.comparisons or result.significance_results:
        render_significance_section(result, console)
