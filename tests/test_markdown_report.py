"""Tests for the generated-from-JSON README results section."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pytest

from legal_rag_eval.metrics.statistical import ComparisonResult
from legal_rag_eval.models import (
    BenchmarkResult,
    EmbeddingModelName,
    GoldStructuralMetrics,
    RetrievalMetrics,
    StrategyName,
)
from legal_rag_eval.reporting.markdown import (
    render_results_markdown,
    update_readme,
    write_results_markdown,
)

if TYPE_CHECKING:
    from pathlib import Path


def _gold_structural(
    strategy: StrategyName,
    document_id: str,
    *,
    avg_chunk_chars: float = 400.0,
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
        avg_chunk_chars=avg_chunk_chars,
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


def _retrieval_metric(
    strategy: StrategyName,
    query_id: str,
    *,
    model: EmbeddingModelName = EmbeddingModelName.MINILM,
    mrr: float = 0.8,
    ndcg: float = 0.75,
) -> RetrievalMetrics:
    return RetrievalMetrics(
        query_id=query_id,
        strategy=strategy,
        embedding_model=model,
        precision_at_1=1.0,
        precision_at_3=0.67,
        precision_at_5=0.6,
        precision_at_10=0.4,
        recall_at_1=0.33,
        recall_at_3=0.67,
        recall_at_5=1.0,
        recall_at_10=1.0,
        mrr=mrr,
        ndcg_at_10=ndcg,
        drm_hit=False,
        drm_rate=0.05,
        n_relevant_sections=2,
    )


def _comparison(
    metric_name: str,
    strategy_b: StrategyName,
    *,
    delta: float = 0.2,
    significant_holm: bool = True,
) -> ComparisonResult:
    return ComparisonResult(
        metric_name=metric_name,
        strategy_a=StrategyName.LEXICHUNK,
        strategy_b=strategy_b,
        embedding_model="all-MiniLM-L6-v2",
        n=6,
        mean_a=0.8,
        mean_b=0.8 - delta,
        delta=delta,
        ci_low=delta - 0.1,
        ci_high=delta + 0.1,
        t_statistic=2.5,
        p_value_t=0.02,
        p_value_wilcoxon=0.03,
        p_value_holm=0.04,
        significant_holm=significant_holm,
        cohens_d=0.9,
        rank_biserial=0.6,
        n_documents=2,
        lodo_min_delta=delta - 0.05,
        lodo_max_delta=delta + 0.05,
        lodo_worst_document="doc1",
    )


def _make_full_result() -> BenchmarkResult:
    """Two strategies, two documents, a few queries, and two ComparisonResults."""
    result = BenchmarkResult(
        timestamp="2026-09-06T12:00:00Z",
        strategies=[StrategyName.LEXICHUNK, StrategyName.RCTS_1024],
        documents=["doc1", "doc2"],
        models=[EmbeddingModelName.MINILM],
        seed=42,
        lexichunk_version="0.8.0b1",
        lexichunk_commit="abc1234",
        config={
            "strategy_parameters": {
                "lexichunk": {"target_chunk_chars": 790},
                "rcts_1024": {"chunk_size": 1024},
            }
        },
    )
    result.structural_metrics = [
        _gold_structural(StrategyName.LEXICHUNK, "doc1", avg_chunk_chars=790.0),
        _gold_structural(StrategyName.LEXICHUNK, "doc2", avg_chunk_chars=810.0),
        _gold_structural(
            StrategyName.RCTS_1024,
            "doc1",
            avg_chunk_chars=1024.0,
            clause_fragmentation_rate=0.40,
            heading_recall=None,
            heading_precision=None,
            xref_target_recall=None,
            xref_target_precision=None,
        ),
        _gold_structural(
            StrategyName.RCTS_1024,
            "doc2",
            avg_chunk_chars=1024.0,
            clause_fragmentation_rate=0.44,
            heading_recall=None,
            heading_precision=None,
            xref_target_recall=None,
            xref_target_precision=None,
        ),
    ]
    result.retrieval_metrics = [
        _retrieval_metric(StrategyName.LEXICHUNK, "doc1_q1", mrr=0.9, ndcg=0.85),
        _retrieval_metric(StrategyName.LEXICHUNK, "doc2_q1", mrr=0.7, ndcg=0.65),
        _retrieval_metric(StrategyName.RCTS_1024, "doc1_q1", mrr=0.5, ndcg=0.45),
        _retrieval_metric(StrategyName.RCTS_1024, "doc2_q1", mrr=0.4, ndcg=0.35),
    ]
    result.comparisons = [
        _comparison("mrr", StrategyName.RCTS_1024, delta=0.35, significant_holm=True),
        _comparison("ndcg_at_10", StrategyName.RCTS_1024, delta=0.30, significant_holm=False),
    ]
    return result


class TestRenderResultsMarkdown:
    def test_contains_provenance(self) -> None:
        md = render_results_markdown(_make_full_result())
        assert "0.8.0b1" in md
        assert "abc1234" in md
        assert "all-MiniLM-L6-v2" in md
        assert "42" in md

    def test_structural_table_has_n_documents(self) -> None:
        md = render_results_markdown(_make_full_result())
        assert "Macro-averaged over documents (n = 2)" in md

    def test_structural_table_contains_macro_averaged_fragmentation(self) -> None:
        md = render_results_markdown(_make_full_result())
        # lexichunk: (0.05 + 0.05) / 2 = 0.050
        assert "0.050" in md

    def test_retrieval_table_has_n_queries(self) -> None:
        md = render_results_markdown(_make_full_result())
        assert "mean over queries, n = 2" in md  # 2 distinct queries for the MiniLM model

    def test_retrieval_table_contains_mean_mrr(self) -> None:
        md = render_results_markdown(_make_full_result())
        # lexichunk mrr mean: (0.9 + 0.7) / 2 = 0.800
        assert "0.800" in md

    def test_comparisons_table_contains_delta_and_ci(self) -> None:
        md = render_results_markdown(_make_full_result())
        assert "+0.350" in md
        assert "+0.250" in md and "+0.450" in md  # ci_low/high for the mrr comparison

    def test_comparisons_table_flags_rcts_1024_as_control(self) -> None:
        md = render_results_markdown(_make_full_result())
        assert "size-matched control" in md
        assert "rcts_1024" in md

    def test_how_to_read_section_states_mean_chunk_length(self) -> None:
        md = render_results_markdown(_make_full_result())
        assert "How to read this" in md
        # lexichunk mean chunk length: (790 + 810) / 2 = 800
        assert "800" in md
        # rcts_1024 mean chunk length: 1024
        assert "1024" in md

    def test_none_renders_as_na(self) -> None:
        md = render_results_markdown(_make_full_result())
        assert "n/a" in md

    def test_no_percentage_improvement_string(self) -> None:
        md = render_results_markdown(_make_full_result())
        assert "improvement" not in md.lower()
        # No "+NN.N% ..." style figure anywhere (deltas are absolute, not percentages).
        assert not re.search(r"[+-]?\d+(\.\d+)?%\s*(improvement|better|faster|gain)", md.lower())

    def test_empty_result_does_not_raise(self) -> None:
        result = BenchmarkResult(timestamp="now")
        md = render_results_markdown(result)
        assert isinstance(md, str)
        assert "No structural metrics" in md
        assert "No retrieval metrics" in md
        assert "No `ComparisonResult` data" in md


class TestWriteResultsMarkdown:
    def test_writes_file(self, tmp_path: Path) -> None:
        path = write_results_markdown(_make_full_result(), tmp_path / "results.md")
        assert path.exists()
        assert "0.8.0b1" in path.read_text(encoding="utf-8")

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        path = write_results_markdown(_make_full_result(), tmp_path / "nested" / "results.md")
        assert path.exists()


class TestUpdateReadme:
    def test_replaces_between_markers(self, tmp_path: Path) -> None:
        readme = tmp_path / "README.md"
        readme.write_text(
            "# My Project\n\n"
            "Some intro text.\n\n"
            "<!-- BEGIN GENERATED RESULTS -->\n"
            "old stale content\n"
            "<!-- END GENERATED RESULTS -->\n\n"
            "Footer text.\n",
            encoding="utf-8",
        )
        update_readme(readme, _make_full_result())
        text = readme.read_text(encoding="utf-8")

        assert "# My Project" in text
        assert "Some intro text." in text
        assert "Footer text." in text
        assert "old stale content" not in text
        assert "0.8.0b1" in text
        assert text.index("<!-- BEGIN GENERATED RESULTS -->") < text.index("0.8.0b1")
        assert text.index("0.8.0b1") < text.index("<!-- END GENERATED RESULTS -->")

    def test_raises_when_start_marker_missing(self, tmp_path: Path) -> None:
        readme = tmp_path / "README.md"
        readme.write_text("# Project\n\n<!-- END GENERATED RESULTS -->\n", encoding="utf-8")
        with pytest.raises(ValueError, match="BEGIN GENERATED RESULTS"):
            update_readme(readme, _make_full_result())

    def test_raises_when_end_marker_missing(self, tmp_path: Path) -> None:
        readme = tmp_path / "README.md"
        readme.write_text("# Project\n\n<!-- BEGIN GENERATED RESULTS -->\n", encoding="utf-8")
        with pytest.raises(ValueError, match="END GENERATED RESULTS"):
            update_readme(readme, _make_full_result())

    def test_raises_when_both_markers_missing(self, tmp_path: Path) -> None:
        readme = tmp_path / "README.md"
        readme.write_text("# Project\n\nNo markers here.\n", encoding="utf-8")
        with pytest.raises(ValueError):
            update_readme(readme, _make_full_result())

    def test_custom_markers(self, tmp_path: Path) -> None:
        readme = tmp_path / "README.md"
        readme.write_text(
            "<!-- CUSTOM START -->\nSTALE_PLACEHOLDER_CONTENT\n<!-- CUSTOM END -->\n",
            encoding="utf-8",
        )
        update_readme(
            readme,
            _make_full_result(),
            start_marker="<!-- CUSTOM START -->",
            end_marker="<!-- CUSTOM END -->",
        )
        text = readme.read_text(encoding="utf-8")
        assert "STALE_PLACEHOLDER_CONTENT" not in text
        assert "0.8.0b1" in text


class TestJsonRoundTripWithComparisons:
    def test_export_load_reconstruct_does_not_raise(self, tmp_path: Path) -> None:
        from legal_rag_eval.reporting.json_export import (
            export_json,
            load_json,
            reconstruct_benchmark_result,
        )

        result = _make_full_result()
        path = tmp_path / "result.json"
        export_json(result, path)
        data = load_json(path)
        reconstructed = reconstruct_benchmark_result(data)

        assert reconstructed.timestamp == result.timestamp
        assert len(reconstructed.structural_metrics) == len(result.structural_metrics)
