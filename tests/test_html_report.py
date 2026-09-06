"""Tests for HTML report generation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from legal_rag_eval.metrics.statistical import ComparisonResult
from legal_rag_eval.models import (
    BenchmarkResult,
    EmbeddingModelName,
    GoldStructuralMetrics,
    LegacyStructuralMetrics,
    RetrievalMetrics,
    StrategyName,
)
from legal_rag_eval.reporting.html import render_html_report

if TYPE_CHECKING:
    from pathlib import Path


def _gold_structural(
    strategy: StrategyName,
    document_id: str,
    *,
    clause_fragmentation_rate: float = 0.05,
    heading_recall: float | None = 0.9,
    heading_precision: float | None = 0.85,
    xref_target_recall: float | None = 0.9,
    xref_target_precision: float | None = 0.88,
) -> GoldStructuralMetrics:
    return GoldStructuralMetrics(
        strategy=strategy,
        document_id=document_id,
        chunk_count=10,
        avg_chunk_chars=400.0,
        chunk_size_cv=0.3,
        located_chunks=10,
        localization_rate=1.0,
        localization_coverage=1.0,
        n_leaf_clauses=20,
        clause_fragmentation_rate=clause_fragmentation_rate,
        n_top_level_clauses=8,
        top_level_over_merge_rate=0.02,
        sub_clause_grouping_rate=0.1,
        heading_recall=heading_recall,
        heading_precision=heading_precision,
        n_gold_headings=8,
        n_definition_uses=12,
        definition_attachment_recall=0.95,
        n_gold_cross_refs=5,
        xref_target_recall=xref_target_recall,
        xref_target_precision=xref_target_precision,
    )


def _minimal_result() -> BenchmarkResult:
    """Create a minimal BenchmarkResult for testing."""
    return BenchmarkResult(
        timestamp="2026-03-18T00:00:00",
        strategies=[StrategyName.LEXICHUNK, StrategyName.RCTS],
        documents=["doc_a", "doc_b"],
        seed=42,
        lexichunk_version="0.8.0b1",
        lexichunk_commit="abc1234",
        structural_metrics=[
            _gold_structural(StrategyName.LEXICHUNK, "doc_a", clause_fragmentation_rate=0.05),
            _gold_structural(
                StrategyName.RCTS,
                "doc_a",
                clause_fragmentation_rate=0.45,
                heading_recall=None,
                heading_precision=None,
                xref_target_recall=None,
                xref_target_precision=None,
            ),
            _gold_structural(StrategyName.LEXICHUNK, "doc_b", clause_fragmentation_rate=0.08),
            _gold_structural(
                StrategyName.RCTS,
                "doc_b",
                clause_fragmentation_rate=0.50,
                heading_recall=None,
                heading_precision=None,
                xref_target_recall=None,
                xref_target_precision=None,
            ),
        ],
    )


def _result_with_retrieval() -> BenchmarkResult:
    """Create a BenchmarkResult that includes retrieval and comparison data."""
    result = _minimal_result()
    result.models = [EmbeddingModelName.MINILM]
    result.retrieval_metrics = [
        RetrievalMetrics(
            query_id="q1",
            strategy=StrategyName.LEXICHUNK,
            embedding_model=EmbeddingModelName.MINILM,
            precision_at_1=1.0,
            precision_at_3=0.67,
            precision_at_5=0.6,
            precision_at_10=0.4,
            recall_at_1=0.33,
            recall_at_3=0.67,
            recall_at_5=1.0,
            recall_at_10=1.0,
            mrr=1.0,
            ndcg_at_10=0.9,
            drm_hit=False,
            drm_rate=0.05,
            n_relevant_sections=3,
        ),
        RetrievalMetrics(
            query_id="q1",
            strategy=StrategyName.RCTS,
            embedding_model=EmbeddingModelName.MINILM,
            precision_at_1=0.0,
            precision_at_3=0.33,
            precision_at_5=0.4,
            precision_at_10=0.3,
            recall_at_1=0.0,
            recall_at_3=0.33,
            recall_at_5=0.67,
            recall_at_10=1.0,
            mrr=0.5,
            ndcg_at_10=0.6,
            drm_hit=True,
            drm_rate=0.2,
            n_relevant_sections=3,
        ),
    ]
    result.comparisons = [
        ComparisonResult(
            metric_name="ndcg_at_10",
            strategy_a=StrategyName.LEXICHUNK,
            strategy_b=StrategyName.RCTS,
            embedding_model="all-MiniLM-L6-v2",
            n=10,
            mean_a=0.9,
            mean_b=0.6,
            delta=0.3,
            ci_low=0.1,
            ci_high=0.5,
            t_statistic=3.5,
            p_value_t=0.01,
            p_value_wilcoxon=0.02,
            p_value_holm=0.02,
            significant_holm=True,
            cohens_d=1.2,
            rank_biserial=0.8,
            n_documents=2,
            lodo_min_delta=0.2,
            lodo_max_delta=0.4,
            lodo_worst_document="doc_a",
        ),
    ]
    result.significance_results = [c.to_significance_result() for c in result.comparisons]
    return result


class TestRenderHtmlReport:
    def test_produces_html_file(self, tmp_path: Path) -> None:
        path = tmp_path / "report.html"
        render_html_report(_minimal_result(), path)
        assert path.exists()
        assert path.stat().st_size > 0

    def test_contains_title(self, tmp_path: Path) -> None:
        path = tmp_path / "report.html"
        render_html_report(_minimal_result(), path)
        html = path.read_text(encoding="utf-8")
        assert "LexiChunk Benchmark Report" in html

    def test_contains_strategy_names(self, tmp_path: Path) -> None:
        path = tmp_path / "report.html"
        render_html_report(_minimal_result(), path)
        html = path.read_text(encoding="utf-8")
        assert "lexichunk" in html
        assert "rcts" in html

    def test_contains_structural_table_rows(self, tmp_path: Path) -> None:
        path = tmp_path / "report.html"
        render_html_report(_minimal_result(), path)
        html = path.read_text(encoding="utf-8")
        assert "doc_a" in html or "n=2" in html
        # macro-averaged fragmentation for lexichunk is (0.05 + 0.08) / 2 = 0.065
        assert "0.065" in html

    def test_contains_na_for_none_optional_fields(self, tmp_path: Path) -> None:
        path = tmp_path / "report.html"
        render_html_report(_minimal_result(), path)
        html = path.read_text(encoding="utf-8")
        assert "n/a" in html

    def test_contains_plotly_chart(self, tmp_path: Path) -> None:
        path = tmp_path / "report.html"
        render_html_report(_minimal_result(), path)
        html = path.read_text(encoding="utf-8")
        assert "plotly" in html.lower()

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        path = tmp_path / "deep" / "nested" / "report.html"
        render_html_report(_minimal_result(), path)
        assert path.exists()

    def test_contains_timestamp(self, tmp_path: Path) -> None:
        path = tmp_path / "report.html"
        render_html_report(_minimal_result(), path)
        html = path.read_text(encoding="utf-8")
        assert "2026-03-18" in html

    def test_contains_provenance(self, tmp_path: Path) -> None:
        path = tmp_path / "report.html"
        render_html_report(_minimal_result(), path)
        html = path.read_text(encoding="utf-8")
        assert "0.8.0b1" in html
        assert "abc1234" in html

    def test_methodology_section(self, tmp_path: Path) -> None:
        path = tmp_path / "report.html"
        render_html_report(_minimal_result(), path)
        html = path.read_text(encoding="utf-8")
        assert "Methodology" in html
        assert "hand-checked span annotations" in html
        assert "NDCG" in html

    def test_no_legacy_structural_section_when_absent(self, tmp_path: Path) -> None:
        path = tmp_path / "report.html"
        render_html_report(_minimal_result(), path)
        html = path.read_text(encoding="utf-8")
        assert "Legacy Structural Metrics" not in html


class TestRenderWithLegacyStructural:
    def test_legacy_section_present_and_labelled_superseded(self, tmp_path: Path) -> None:
        result = _minimal_result()
        result.legacy_structural_metrics = [
            LegacyStructuralMetrics(
                strategy=StrategyName.LEXICHUNK,
                document_id="doc_a",
                clause_fragmentation_rate=0.0,
                definition_preservation_rate=1.0,
                cross_ref_resolution_rate=1.0,
                hierarchy_depth_retained=1.0,
                chunk_size_cv=0.2,
                chunk_count=10,
                avg_chunk_chars=400.0,
            ),
        ]
        path = tmp_path / "report.html"
        render_html_report(result, path)
        html = path.read_text(encoding="utf-8")
        assert "Legacy Structural Metrics" in html
        assert "SUPERSEDED" in html


class TestRenderWithRetrieval:
    def test_retrieval_section_present(self, tmp_path: Path) -> None:
        path = tmp_path / "report.html"
        render_html_report(_result_with_retrieval(), path)
        html = path.read_text(encoding="utf-8")
        assert "Retrieval Quality Metrics" in html

    def test_comparisons_section_present(self, tmp_path: Path) -> None:
        path = tmp_path / "report.html"
        render_html_report(_result_with_retrieval(), path)
        html = path.read_text(encoding="utf-8")
        assert "Statistical Comparisons" in html
        assert "badge-sig" in html
        # only an rcts_1024 row gets the bold per-row control label
        assert "<strong>(size-matched control)</strong>" not in html

    def test_drm_chart_present(self, tmp_path: Path) -> None:
        path = tmp_path / "report.html"
        render_html_report(_result_with_retrieval(), path)
        html = path.read_text(encoding="utf-8")
        assert "Document Retrieval Mismatch" in html


class TestRenderWithoutRetrieval:
    def test_no_retrieval_section(self, tmp_path: Path) -> None:
        path = tmp_path / "report.html"
        render_html_report(_minimal_result(), path)
        html = path.read_text(encoding="utf-8")
        assert "Retrieval Quality Metrics" not in html

    def test_no_comparisons_section(self, tmp_path: Path) -> None:
        path = tmp_path / "report.html"
        render_html_report(_minimal_result(), path)
        html = path.read_text(encoding="utf-8")
        assert "Statistical Comparisons" not in html
