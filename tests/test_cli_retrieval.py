"""Tests for CLI retrieval metrics rendering."""

from __future__ import annotations

from io import StringIO

from rich.console import Console

from scaffolder.models import (
    EmbeddingModelName,
    RetrievalMetrics,
    SignificanceResult,
    StrategyName,
)
from scaffolder.reporting.cli import render_retrieval_table


def _make_retrieval_metrics(
    strategy: StrategyName = StrategyName.LEXICHUNK,
    model: EmbeddingModelName = EmbeddingModelName.MINILM,
    query_id: str = "q1",
    mrr: float = 1.0,
    ndcg: float = 0.85,
    drm: bool = False,
    drm_rate: float = 0.0,
) -> RetrievalMetrics:
    return RetrievalMetrics(
        query_id=query_id,
        strategy=strategy,
        embedding_model=model,
        precision_at_1=1.0,
        precision_at_3=0.67,
        precision_at_5=0.4,
        precision_at_10=0.2,
        recall_at_1=0.5,
        recall_at_3=0.75,
        recall_at_5=1.0,
        recall_at_10=1.0,
        mrr=mrr,
        ndcg_at_10=ndcg,
        drm_hit=drm,
        drm_rate=drm_rate,
    )


class TestRenderRetrievalTable:
    def test_renders_without_error(self) -> None:
        console = Console(file=StringIO())
        metrics = [
            _make_retrieval_metrics(strategy=StrategyName.LEXICHUNK),
            _make_retrieval_metrics(strategy=StrategyName.FIXED_SIZE, mrr=0.5, ndcg=0.4),
        ]
        render_retrieval_table(metrics, console=console)

    def test_with_significance_markers(self) -> None:
        console = Console(file=StringIO())
        metrics = [
            _make_retrieval_metrics(strategy=StrategyName.LEXICHUNK),
            _make_retrieval_metrics(strategy=StrategyName.FIXED_SIZE),
        ]
        sig_results = [
            SignificanceResult(
                metric_name="mrr",
                strategy_a=StrategyName.LEXICHUNK,
                strategy_b=StrategyName.FIXED_SIZE,
                mean_a=1.0,
                mean_b=0.5,
                improvement_pct=100.0,
                t_statistic=3.5,
                p_value=0.003,
                significant=True,
                effect_size=1.2,
                n_queries=10,
            ),
            SignificanceResult(
                metric_name="ndcg_at_10",
                strategy_a=StrategyName.LEXICHUNK,
                strategy_b=StrategyName.FIXED_SIZE,
                mean_a=0.85,
                mean_b=0.4,
                improvement_pct=112.5,
                t_statistic=2.3,
                p_value=0.04,
                significant=True,
                effect_size=0.8,
                n_queries=10,
            ),
        ]
        render_retrieval_table(metrics, significance_results=sig_results, console=console)
        output = console.file.getvalue()  # type: ignore[union-attr]
        assert "**" in output  # p=0.003 -> **
        assert "*" in output  # p=0.04 -> *

    def test_empty_results(self) -> None:
        console = Console(file=StringIO())
        render_retrieval_table([], console=console)
        output = console.file.getvalue()  # type: ignore[union-attr]
        assert "No retrieval results" in output

    def test_multiple_models(self) -> None:
        console = Console(file=StringIO())
        metrics = [
            _make_retrieval_metrics(model=EmbeddingModelName.MINILM),
            _make_retrieval_metrics(model=EmbeddingModelName.BGE_BASE, mrr=0.9),
        ]
        render_retrieval_table(metrics, console=console)
        output = console.file.getvalue()  # type: ignore[union-attr]
        assert "all-MiniLM-L6-v2" in output
        assert "bge-base-en-v1.5" in output

    def test_shows_n_queries_in_title(self) -> None:
        console = Console(file=StringIO())
        metrics = [
            _make_retrieval_metrics(query_id="q1"),
            _make_retrieval_metrics(query_id="q2"),
        ]
        render_retrieval_table(metrics, console=console)
        output = console.file.getvalue()  # type: ignore[union-attr]
        assert "n=2 queries" in output

    def test_drm_rate_displayed(self) -> None:
        # Wide enough that the "DRM rate" header doesn't wrap onto two lines.
        console = Console(file=StringIO(), width=150)
        metrics = [
            _make_retrieval_metrics(strategy=StrategyName.LEXICHUNK, drm=False, drm_rate=0.05),
            _make_retrieval_metrics(strategy=StrategyName.RCTS, drm=True, drm_rate=0.4),
        ]
        render_retrieval_table(metrics, console=console)
        output = console.file.getvalue()  # type: ignore[union-attr]
        assert "DRM rate" in output
