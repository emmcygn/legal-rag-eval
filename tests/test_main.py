"""Tests for benchmark result reconstruction used by `scaffolder.__main__`'s `report` command.

`reconstruct_benchmark_result` used to live in `scaffolder.__main__` as a private function; it
now lives in `scaffolder.reporting.json_export` (the reporting package owns JSON round-tripping,
and `__main__` just calls it) — see that module for the implementation and its docstring for
what is (and, for `comparisons`, deliberately is not) reconstructed.
"""

from __future__ import annotations

import json
import subprocess
import sys
from typing import TYPE_CHECKING

from scaffolder.models import StrategyName
from scaffolder.reporting.json_export import reconstruct_benchmark_result

if TYPE_CHECKING:
    from pathlib import Path


class TestReconstructBenchmarkResult:
    def test_empty_data(self) -> None:
        result = reconstruct_benchmark_result({})
        assert result.timestamp == ""
        assert result.strategies == []
        assert result.documents == []

    def test_gold_structural_metrics(self) -> None:
        data = {
            "timestamp": "2026-03-18T00:00:00",
            "strategies": ["lexichunk", "rcts"],
            "documents": ["doc1"],
            "models": [],
            "structural_metrics": [
                {
                    "strategy": "lexichunk",
                    "document_id": "doc1",
                    "chunk_count": 10,
                    "avg_chunk_chars": 400.0,
                    "chunk_size_cv": 0.3,
                    "located_chunks": 10,
                    "localization_rate": 1.0,
                    "localization_coverage": 1.0,
                    "n_leaf_clauses": 20,
                    "clause_fragmentation_rate": 0.05,
                    "n_top_level_clauses": 8,
                    "top_level_over_merge_rate": 0.02,
                    "sub_clause_grouping_rate": 0.1,
                    "heading_recall": 0.9,
                    "heading_precision": 0.85,
                    "n_gold_headings": 8,
                    "n_definition_uses": 12,
                    "definition_attachment_recall": 0.95,
                    "n_gold_cross_refs": 5,
                    "xref_target_recall": 0.9,
                    "xref_target_precision": 0.88,
                },
            ],
            "retrieval_metrics": [],
            "significance_results": [],
        }
        result = reconstruct_benchmark_result(data)
        assert len(result.strategies) == 2
        assert result.strategies[0] == StrategyName.LEXICHUNK
        assert len(result.structural_metrics) == 1
        assert result.structural_metrics[0].clause_fragmentation_rate == 0.05
        assert result.structural_metrics[0].heading_recall == 0.9

    def test_invalid_strategy_skipped(self) -> None:
        data = {
            "strategies": ["lexichunk", "invalid_strategy"],
        }
        result = reconstruct_benchmark_result(data)
        assert len(result.strategies) == 1

    def test_malformed_metric_skipped(self) -> None:
        data = {
            "structural_metrics": [
                {"strategy": "lexichunk"},  # Missing fields
            ],
        }
        result = reconstruct_benchmark_result(data)
        assert len(result.structural_metrics) == 0

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
                },
            ],
        }
        result = reconstruct_benchmark_result(data)
        assert len(result.retrieval_metrics) == 1
        assert result.retrieval_metrics[0].mrr == 1.0

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

    def test_roundtrip_via_json(self, tmp_path: Path) -> None:
        """Export a result to JSON, then reconstruct it."""
        from scaffolder.models import BenchmarkResult, GoldStructuralMetrics
        from scaffolder.reporting.json_export import export_json, load_json

        original = BenchmarkResult(
            timestamp="2026-03-18T00:00:00",
            strategies=[StrategyName.LEXICHUNK],
            documents=["doc1"],
            structural_metrics=[
                GoldStructuralMetrics(
                    strategy=StrategyName.LEXICHUNK,
                    document_id="doc1",
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
                ),
            ],
        )
        path = tmp_path / "test.json"
        export_json(original, path)
        data = load_json(path)
        reconstructed = reconstruct_benchmark_result(data)
        assert len(reconstructed.structural_metrics) == 1
        sm = reconstructed.structural_metrics[0]
        assert sm.clause_fragmentation_rate == 0.05


def test_primary_cli_runs_offline_anchored_benchmark(tmp_path: Path) -> None:
    output_path = tmp_path / "evidence.json"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scaffolder",
            "benchmark",
            "--output",
            str(output_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["report_version"] == "evidence_benchmark_report_v1"
    assert report["scope"]["retrieval"].startswith("deterministic lexical")
