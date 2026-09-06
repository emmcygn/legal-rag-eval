"""Tests for CLI and JSON reporting modules."""

from __future__ import annotations

import io
import json
from typing import TYPE_CHECKING

from rich.console import Console

from scaffolder.models import (
    BenchmarkResult,
    EmbeddingModelName,
    GoldStructuralMetrics,
    LegacyStructuralMetrics,
    RetrievalMetrics,
    SignificanceResult,
    StrategyName,
)
from scaffolder.reporting.cli import (
    render_benchmark,
    render_legacy_structural_table,
    render_significance_section,
    render_structural_table,
    render_summary_header,
)
from scaffolder.reporting.json_export import (
    export_json,
    export_json_string,
    load_json,
)

if TYPE_CHECKING:
    from pathlib import Path


def _gold_structural(
    strategy: StrategyName,
    document_id: str,
    *,
    chunk_count: int = 10,
    avg_chunk_chars: float = 400.0,
    chunk_size_cv: float = 0.3,
    localization_rate: float = 1.0,
    clause_fragmentation_rate: float = 0.05,
    top_level_over_merge_rate: float = 0.02,
    sub_clause_grouping_rate: float = 0.1,
    heading_recall: float | None = 0.9,
    heading_precision: float | None = 0.85,
    definition_attachment_recall: float = 0.95,
    xref_target_recall: float | None = 0.9,
    xref_target_precision: float | None = 0.88,
) -> GoldStructuralMetrics:
    return GoldStructuralMetrics(
        strategy=strategy,
        document_id=document_id,
        chunk_count=chunk_count,
        avg_chunk_chars=avg_chunk_chars,
        chunk_size_cv=chunk_size_cv,
        located_chunks=chunk_count,
        localization_rate=localization_rate,
        localization_coverage=localization_rate,
        n_leaf_clauses=20,
        clause_fragmentation_rate=clause_fragmentation_rate,
        n_top_level_clauses=8,
        top_level_over_merge_rate=top_level_over_merge_rate,
        sub_clause_grouping_rate=sub_clause_grouping_rate,
        heading_recall=heading_recall,
        heading_precision=heading_precision,
        n_gold_headings=8,
        n_definition_uses=12,
        definition_attachment_recall=definition_attachment_recall,
        n_gold_cross_refs=5,
        xref_target_recall=xref_target_recall,
        xref_target_precision=xref_target_precision,
    )


def _make_structural_result() -> BenchmarkResult:
    return BenchmarkResult(
        timestamp="2026-03-18T12:00:00",
        strategies=[StrategyName.LEXICHUNK, StrategyName.RCTS],
        documents=["doc1", "doc2"],
        seed=42,
        lexichunk_version="0.8.0b1",
        lexichunk_commit="abc1234",
        structural_metrics=[
            _gold_structural(StrategyName.LEXICHUNK, "doc1", clause_fragmentation_rate=0.05),
            _gold_structural(
                StrategyName.RCTS,
                "doc1",
                clause_fragmentation_rate=0.45,
                heading_recall=None,
                heading_precision=None,
                xref_target_recall=None,
                xref_target_precision=None,
            ),
            _gold_structural(StrategyName.LEXICHUNK, "doc2", clause_fragmentation_rate=0.08),
            _gold_structural(
                StrategyName.RCTS,
                "doc2",
                clause_fragmentation_rate=0.50,
                heading_recall=None,
                heading_precision=None,
                xref_target_recall=None,
                xref_target_precision=None,
            ),
        ],
    )


class TestCLIRendering:
    def _console(self) -> Console:
        return Console(file=io.StringIO(), force_terminal=True, width=160)

    def test_render_structural_table_runs(self) -> None:
        console = self._console()
        result = _make_structural_result()
        render_structural_table(result.structural_metrics, console)
        output = console.file.getvalue()  # type: ignore[union-attr]
        assert "lexichunk" in output
        assert "rcts" in output
        assert "n=2 documents" in output

    def test_render_structural_table_na_for_none(self) -> None:
        console = self._console()
        result = _make_structural_result()
        render_structural_table(result.structural_metrics, console)
        output = console.file.getvalue()  # type: ignore[union-attr]
        # RCTS has heading_recall/precision and xref_* as None on every document.
        assert "n/a" in output

    def test_render_structural_table_empty(self) -> None:
        console = self._console()
        render_structural_table([], console)
        output = console.file.getvalue()  # type: ignore[union-attr]
        assert "No structural results" in output

    def test_render_legacy_structural_table_only_when_nonempty(self) -> None:
        console = self._console()
        render_legacy_structural_table([], console)
        output = console.file.getvalue()  # type: ignore[union-attr]
        assert output == ""

    def test_render_legacy_structural_table_says_superseded(self) -> None:
        console = self._console()
        legacy = [
            LegacyStructuralMetrics(
                strategy=StrategyName.LEXICHUNK,
                document_id="doc1",
                clause_fragmentation_rate=0.0,
                definition_preservation_rate=1.0,
                cross_ref_resolution_rate=1.0,
                hierarchy_depth_retained=1.0,
                chunk_size_cv=0.2,
                chunk_count=10,
                avg_chunk_chars=400.0,
            ),
        ]
        render_legacy_structural_table(legacy, console)
        output = console.file.getvalue()  # type: ignore[union-attr]
        assert "SUPERSEDED" in output

    def test_render_summary_header(self) -> None:
        console = self._console()
        result = _make_structural_result()
        render_summary_header(result, console)
        output = console.file.getvalue()  # type: ignore[union-attr]
        assert "Benchmark Results" in output
        assert "lexichunk" in output
        assert "42" in output  # seed
        assert "0.8.0b1" in output  # lexichunk_version
        assert "abc1234" in output  # lexichunk_commit

    def test_render_summary_header_unknown_provenance(self) -> None:
        console = self._console()
        result = BenchmarkResult(timestamp="now")
        render_summary_header(result, console)
        output = console.file.getvalue()  # type: ignore[union-attr]
        assert "unknown" in output

    def test_render_summary_header_strategy_parameters(self) -> None:
        console = self._console()
        result = _make_structural_result()
        result.config["strategy_parameters"] = {"rcts": {"chunk_size": 1024}}
        render_summary_header(result, console)
        output = console.file.getvalue()  # type: ignore[union-attr]
        assert "chunk_size" in output

    def test_render_benchmark_full(self) -> None:
        console = self._console()
        result = _make_structural_result()
        render_benchmark(result, console)
        output = console.file.getvalue()  # type: ignore[union-attr]
        assert "Benchmark Results" in output
        assert "lexichunk" in output

    def test_render_benchmark_default_console(self) -> None:
        """render_benchmark works when called without a console."""
        result = _make_structural_result()
        # Just verify no exception; output goes to real stdout
        render_benchmark(result)

    def test_render_summary_header_with_models(self) -> None:
        console = self._console()
        result = _make_structural_result()
        result.models = [EmbeddingModelName.MINILM]
        render_summary_header(result, console)
        output = console.file.getvalue()  # type: ignore[union-attr]
        assert "all-MiniLM-L6-v2" in output

    def test_render_significance_section_empty(self) -> None:
        console = self._console()
        result = BenchmarkResult(timestamp="now")
        render_significance_section(result, console)
        output = console.file.getvalue()  # type: ignore[union-attr]
        assert output == ""

    def test_render_significance_section_legacy_fallback_warns(self) -> None:
        console = self._console()
        result = BenchmarkResult(timestamp="now")
        result.significance_results = [
            SignificanceResult(
                metric_name="ndcg_at_10",
                strategy_a=StrategyName.LEXICHUNK,
                strategy_b=StrategyName.RCTS,
                mean_a=0.9,
                mean_b=0.6,
                improvement_pct=50.0,
                t_statistic=3.5,
                p_value=0.01,
                significant=True,
                effect_size=1.2,
                n_queries=10,
            ),
        ]
        render_significance_section(result, console)
        output = console.file.getvalue()  # type: ignore[union-attr]
        assert "uncorrected" in output.lower()
        assert "ndcg_at_10" in output


class TestJSONExport:
    def test_export_creates_file(self, tmp_path: Path) -> None:
        result = _make_structural_result()
        path = tmp_path / "test.json"
        export_json(result, path)
        assert path.exists()

    def test_export_creates_parent_dirs(self, tmp_path: Path) -> None:
        result = _make_structural_result()
        path = tmp_path / "deep" / "nested" / "result.json"
        export_json(result, path)
        assert path.exists()

    def test_export_roundtrip(self, tmp_path: Path) -> None:
        result = _make_structural_result()
        path = tmp_path / "test.json"
        export_json(result, path)
        loaded = load_json(path)
        assert loaded["timestamp"] == result.timestamp
        assert len(loaded["structural_metrics"]) == 4

    def test_export_json_string(self) -> None:
        result = _make_structural_result()
        s = export_json_string(result)
        data = json.loads(s)
        assert data["timestamp"] == "2026-03-18T12:00:00"

    def test_export_with_retrieval_metrics(self, tmp_path: Path) -> None:
        result = _make_structural_result()
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
                drm_rate=0.1,
                n_relevant_sections=3,
            ),
        ]
        result.significance_results = [
            SignificanceResult(
                metric_name="ndcg_at_10",
                strategy_a=StrategyName.LEXICHUNK,
                strategy_b=StrategyName.RCTS,
                mean_a=0.9,
                mean_b=0.6,
                improvement_pct=50.0,
                t_statistic=3.5,
                p_value=0.01,
                significant=True,
                effect_size=1.2,
                n_queries=10,
            ),
        ]
        path = tmp_path / "full.json"
        export_json(result, path)
        loaded = load_json(path)
        assert len(loaded["retrieval_metrics"]) == 1
        assert len(loaded["significance_results"]) == 1
