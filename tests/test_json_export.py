"""Tests for JSON export and reconstruction."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import TYPE_CHECKING

from legal_rag_eval.models import (
    BenchmarkResult,
    EmbeddingModelName,
    GoldStructuralMetrics,
    LegacyStructuralMetrics,
    RetrievalMetrics,
    SignificanceResult,
    StrategyName,
)

if TYPE_CHECKING:
    from pathlib import Path
from legal_rag_eval.reporting.cli import ComparisonLike
from legal_rag_eval.reporting.json_export import (
    export_json,
    export_json_string,
    load_json,
    reconstruct_benchmark_result,
)


def _gold_structural(strategy: StrategyName, document_id: str) -> GoldStructuralMetrics:
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
        clause_fragmentation_rate=0.05,
        n_top_level_clauses=8,
        top_level_over_merge_rate=0.02,
        sub_clause_grouping_rate=0.1,
        heading_recall=0.9,
        heading_precision=0.85,
        n_gold_headings=8,
        n_definition_uses=12,
        definition_attachment_recall=0.95,
        n_gold_cross_refs=5,
        xref_target_recall=0.9,
        xref_target_precision=0.88,
    )


def _make_result() -> BenchmarkResult:
    """Create a minimal BenchmarkResult for testing."""
    return BenchmarkResult(
        timestamp="2026-03-18T12:00:00",
        strategies=[StrategyName.LEXICHUNK, StrategyName.RCTS],
        documents=["uk_service_agreement"],
        models=[EmbeddingModelName.MINILM],
        seed=42,
        lexichunk_version="0.8.0b1",
        lexichunk_commit="abc1234",
        config={"strategies": ["lexichunk", "rcts"], "strategy_parameters": {"rcts": {"n": 1}}},
    )


class TestExportJson:
    """Test JSON file export."""

    def test_export_creates_file(self, tmp_path: Path) -> None:
        result = _make_result()
        out = export_json(result, tmp_path / "test.json")
        assert out.exists()

    def test_export_valid_json(self, tmp_path: Path) -> None:
        result = _make_result()
        out = export_json(result, tmp_path / "test.json")
        data = json.loads(out.read_text())
        assert data["timestamp"] == "2026-03-18T12:00:00"

    def test_export_creates_parent_dirs(self, tmp_path: Path) -> None:
        result = _make_result()
        out = export_json(result, tmp_path / "nested" / "dir" / "test.json")
        assert out.exists()

    def test_export_strategies(self, tmp_path: Path) -> None:
        result = _make_result()
        out = export_json(result, tmp_path / "test.json")
        data = json.loads(out.read_text())
        assert data["strategies"] == ["lexichunk", "rcts"]

    def test_export_gold_structural_metrics(self, tmp_path: Path) -> None:
        result = _make_result()
        result.structural_metrics = [_gold_structural(StrategyName.LEXICHUNK, "doc1")]
        out = export_json(result, tmp_path / "test.json")
        data = json.loads(out.read_text())
        assert data["structural_metrics"][0]["localization_rate"] == 1.0
        assert data["structural_metrics"][0]["heading_recall"] == 0.9

    def test_export_gold_structural_metrics_none_fields(self, tmp_path: Path) -> None:
        result = _make_result()
        sm = _gold_structural(StrategyName.SENTENCE_SPLIT, "doc1")
        sm_no_headings = replace(sm, heading_recall=None, heading_precision=None)
        result.structural_metrics = [sm_no_headings]
        out = export_json(result, tmp_path / "test.json")
        data = json.loads(out.read_text())
        assert data["structural_metrics"][0]["heading_recall"] is None


class TestExportJsonString:
    """Test JSON string export."""

    def test_returns_string(self) -> None:
        result = _make_result()
        s = export_json_string(result)
        assert isinstance(s, str)

    def test_valid_json_string(self) -> None:
        result = _make_result()
        s = export_json_string(result)
        data = json.loads(s)
        assert data["timestamp"] == "2026-03-18T12:00:00"


class TestLoadJson:
    """Test loading JSON results."""

    def test_roundtrip(self, tmp_path: Path) -> None:
        result = _make_result()
        out = export_json(result, tmp_path / "test.json")
        data = load_json(out)
        assert data["timestamp"] == "2026-03-18T12:00:00"
        assert data["documents"] == ["uk_service_agreement"]


class TestReconstructBenchmarkResult:
    def test_empty_data(self) -> None:
        result = reconstruct_benchmark_result({})
        assert result.timestamp == ""
        assert result.strategies == []
        assert result.documents == []

    def test_provenance_fields(self) -> None:
        result = reconstruct_benchmark_result(
            {
                "timestamp": "2026-03-18T00:00:00",
                "seed": 42,
                "lexichunk_version": "0.8.0b1",
                "lexichunk_commit": "abc1234",
                "config": {"strategy_parameters": {"rcts": {"n": 1}}},
            }
        )
        assert result.seed == 42
        assert result.lexichunk_version == "0.8.0b1"
        assert result.lexichunk_commit == "abc1234"
        assert result.config["strategy_parameters"] == {"rcts": {"n": 1}}

    def test_invalid_strategy_skipped(self) -> None:
        data = {"strategies": ["lexichunk", "invalid_strategy"]}
        result = reconstruct_benchmark_result(data)
        assert len(result.strategies) == 1
        assert result.strategies[0] == StrategyName.LEXICHUNK

    def test_gold_structural_metrics_roundtrip(self, tmp_path: Path) -> None:
        original = _make_result()
        original.structural_metrics = [_gold_structural(StrategyName.LEXICHUNK, "doc1")]
        path = tmp_path / "test.json"
        export_json(original, path)
        reconstructed = reconstruct_benchmark_result(load_json(path))
        assert len(reconstructed.structural_metrics) == 1
        sm = reconstructed.structural_metrics[0]
        assert isinstance(sm, GoldStructuralMetrics)
        assert sm.localization_rate == 1.0
        assert sm.heading_recall == 0.9
        assert len(reconstructed.legacy_structural_metrics) == 0

    def test_gold_structural_metrics_optional_none_roundtrip(self, tmp_path: Path) -> None:
        original = _make_result()
        sm = _gold_structural(StrategyName.SENTENCE_SPLIT, "doc1")
        sm_no_xref = replace(sm, xref_target_recall=None, xref_target_precision=None)
        original.structural_metrics = [sm_no_xref]
        path = tmp_path / "test.json"
        export_json(original, path)
        reconstructed = reconstruct_benchmark_result(load_json(path))
        assert reconstructed.structural_metrics[0].xref_target_recall is None

    def test_malformed_gold_structural_metric_skipped(self) -> None:
        data = {
            "structural_metrics": [
                {"strategy": "lexichunk", "document_id": "doc1"},  # missing required fields
            ],
        }
        result = reconstruct_benchmark_result(data)
        assert len(result.structural_metrics) == 0
        assert len(result.legacy_structural_metrics) == 0

    def test_legacy_structural_metrics_field_reconstructed(self) -> None:
        data = {
            "legacy_structural_metrics": [
                {
                    "strategy": "lexichunk",
                    "document_id": "doc1",
                    "clause_fragmentation_rate": 0.0,
                    "definition_preservation_rate": 1.0,
                    "cross_ref_resolution_rate": 1.0,
                    "hierarchy_depth_retained": 1.0,
                    "chunk_size_cv": 0.2,
                    "chunk_count": 10,
                    "avg_chunk_chars": 400.0,
                },
            ],
        }
        result = reconstruct_benchmark_result(data)
        assert len(result.legacy_structural_metrics) == 1
        assert isinstance(result.legacy_structural_metrics[0], LegacyStructuralMetrics)

    def test_old_style_structural_metrics_routed_to_legacy(self) -> None:
        """Older JSON exports put the legacy shape under the "structural_metrics" key."""
        data = {
            "structural_metrics": [
                {
                    "strategy": "lexichunk",
                    "document_id": "doc1",
                    "clause_fragmentation_rate": 0.05,
                    "definition_preservation_rate": 0.95,
                    "cross_ref_resolution_rate": 0.90,
                    "hierarchy_depth_retained": 1.0,
                    "chunk_size_cv": 0.3,
                    "chunk_count": 10,
                    "avg_chunk_chars": 400.0,
                },
            ],
        }
        result = reconstruct_benchmark_result(data)
        assert len(result.structural_metrics) == 0
        assert len(result.legacy_structural_metrics) == 1
        assert result.legacy_structural_metrics[0].definition_preservation_rate == 0.95

    def test_retrieval_metrics(self) -> None:
        data = {
            "retrieval_metrics": [
                {
                    "query_id": "q1",
                    "strategy": "lexichunk",
                    "embedding_model": "all-MiniLM-L6-v2",
                    "precision_at_1": 1.0,
                    "precision_at_3": 0.67,
                    "precision_at_5": 0.6,
                    "precision_at_10": 0.4,
                    "recall_at_1": 0.33,
                    "recall_at_3": 0.67,
                    "recall_at_5": 1.0,
                    "recall_at_10": 1.0,
                    "mrr": 1.0,
                    "ndcg_at_10": 0.9,
                    "drm_hit": False,
                    "drm_rate": 0.1,
                    "n_relevant_sections": 3,
                },
            ],
        }
        result = reconstruct_benchmark_result(data)
        assert len(result.retrieval_metrics) == 1
        assert result.retrieval_metrics[0].mrr == 1.0
        assert result.retrieval_metrics[0].drm_rate == 0.1

    def test_retrieval_metrics_defaults_missing_new_fields(self) -> None:
        """Older JSON without drm_rate/n_relevant_sections still loads, with defaults."""
        data = {
            "retrieval_metrics": [
                {
                    "query_id": "q1",
                    "strategy": "lexichunk",
                    "embedding_model": "all-MiniLM-L6-v2",
                    "precision_at_1": 1.0,
                    "precision_at_3": 0.67,
                    "precision_at_5": 0.6,
                    "precision_at_10": 0.4,
                    "recall_at_1": 0.33,
                    "recall_at_3": 0.67,
                    "recall_at_5": 1.0,
                    "recall_at_10": 1.0,
                    "mrr": 1.0,
                    "ndcg_at_10": 0.9,
                    "drm_hit": False,
                },
            ],
        }
        result = reconstruct_benchmark_result(data)
        assert result.retrieval_metrics[0].drm_rate == 0.0
        assert result.retrieval_metrics[0].n_relevant_sections == 0

    def test_significance_results(self) -> None:
        data = {
            "significance_results": [
                {
                    "metric_name": "ndcg_at_10",
                    "strategy_a": "lexichunk",
                    "strategy_b": "rcts",
                    "mean_a": 0.9,
                    "mean_b": 0.6,
                    "improvement_pct": 50.0,
                    "t_statistic": 3.5,
                    "p_value": 0.01,
                    "significant": True,
                    "effect_size": 1.2,
                    "n_queries": 10,
                },
            ],
        }
        result = reconstruct_benchmark_result(data)
        assert len(result.significance_results) == 1
        assert result.significance_results[0].significant is True

    def test_malformed_significance_result_skipped(self) -> None:
        data = {"significance_results": [{"metric_name": "mrr"}]}
        result = reconstruct_benchmark_result(data)
        assert len(result.significance_results) == 0

    def test_partial_comparison_skipped(self) -> None:
        """An entry missing most of the shape is dropped with a warning, not half-built."""
        data = {
            "comparisons": [
                {"metric_name": "mrr", "strategy_a": "lexichunk", "strategy_b": "rcts_1024"}
            ],
        }
        result = reconstruct_benchmark_result(data)
        assert result.comparisons == []

    def test_roundtrip_preserves_comparisons(self, tmp_path: Path) -> None:
        """A round trip must keep the bootstrap CI, Holm p-value and LODO range.

        `scripts/update_readme.py` renders the README from an exported JSON, so dropping
        `comparisons` on reconstruction silently published tables with no uncertainty in
        them at all. Reconstruction yields `ReconstructedComparison`, not `ComparisonResult`
        — the reporting layer reads the shape, not the class."""
        from legal_rag_eval.metrics.statistical import ComparisonResult

        original = _make_result()
        original.comparisons = [
            ComparisonResult(
                metric_name="mrr",
                strategy_a=StrategyName.LEXICHUNK,
                strategy_b=StrategyName.RCTS_1024,
                embedding_model="all-MiniLM-L6-v2",
                n=8,
                mean_a=0.8,
                mean_b=0.6,
                delta=0.2,
                ci_low=0.05,
                ci_high=0.35,
                t_statistic=2.9,
                p_value_t=0.02,
                p_value_wilcoxon=0.03,
                p_value_holm=0.04,
                significant_holm=True,
                cohens_d=0.9,
                rank_biserial=0.6,
                n_documents=2,
                lodo_min_delta=0.1,
                lodo_max_delta=0.3,
                lodo_worst_document="doc1",
            ),
        ]
        original.significance_results = [c.to_significance_result() for c in original.comparisons]

        path = tmp_path / "test.json"
        export_json(original, path)
        data = load_json(path)
        reconstructed = reconstruct_benchmark_result(data)

        assert len(reconstructed.significance_results) == 1
        assert len(reconstructed.comparisons) == 1
        rebuilt = reconstructed.comparisons[0]
        assert isinstance(rebuilt, ComparisonLike)
        assert rebuilt.metric_name == "mrr"
        assert rebuilt.strategy_a is StrategyName.LEXICHUNK
        assert rebuilt.strategy_b is StrategyName.RCTS_1024
        assert (rebuilt.ci_low, rebuilt.ci_high) == (0.05, 0.35)
        assert rebuilt.p_value_holm == 0.04
        assert rebuilt.significant_holm is True
        assert (rebuilt.lodo_min_delta, rebuilt.lodo_max_delta) == (0.1, 0.3)

    def test_roundtrip_via_json_full(self, tmp_path: Path) -> None:
        """Export a result to JSON, then reconstruct it."""
        original = _make_result()
        original.structural_metrics = [_gold_structural(StrategyName.LEXICHUNK, "doc1")]
        original.retrieval_metrics = [
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
            ),
        ]
        original.significance_results = [
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
        path = tmp_path / "test.json"
        export_json(original, path)
        data = load_json(path)
        reconstructed = reconstruct_benchmark_result(data)
        assert len(reconstructed.structural_metrics) == 1
        assert reconstructed.structural_metrics[0].localization_rate == 1.0
        assert len(reconstructed.retrieval_metrics) == 1
        assert len(reconstructed.significance_results) == 1
