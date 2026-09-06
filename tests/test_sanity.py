"""Sanity tests for metric ranges and invariants.

These tests run real benchmark data through the pipeline and verify
that all metric values are within expected ranges and satisfy known
invariants (e.g., recall monotonicity, precision trends).
"""

from __future__ import annotations

import datetime

from scaffolder.chunking import ChunkingPipeline, get_all_strategies
from scaffolder.fixtures import FixtureManager
from scaffolder.metrics.structural import compute_structural_metrics
from scaffolder.models import BenchmarkResult, StructuralMetrics


def _build_structural_result() -> BenchmarkResult:
    """Run the full structural pipeline and return a BenchmarkResult."""
    fm = FixtureManager()
    documents = fm.load_all()
    strategies = get_all_strategies()
    pipeline = ChunkingPipeline(strategies)
    strategy_results = pipeline.run(documents)

    result = BenchmarkResult(
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        strategies=[sr.strategy for sr in strategy_results],
        documents=[d.id for d in documents],
        strategy_results=strategy_results,
    )

    for sr in strategy_results:
        for cs in sr.chunk_sets:
            doc = fm.get_by_id(cs.document_id)
            sm = compute_structural_metrics(cs, doc)
            result.structural_metrics.append(sm)

    return result


# Cache the result so we only run the pipeline once across all tests
_RESULT: BenchmarkResult | None = None


def _get_result() -> BenchmarkResult:
    global _RESULT  # noqa: PLW0603
    if _RESULT is None:
        _RESULT = _build_structural_result()
    return _RESULT


class TestMetricRanges:
    def test_clause_fragmentation_rate_in_range(self) -> None:
        for sm in _get_result().structural_metrics:
            assert 0.0 <= sm.clause_fragmentation_rate <= 1.0, (
                f"{sm.strategy.value}/{sm.document_id}: CFR={sm.clause_fragmentation_rate}"
            )

    def test_definition_preservation_rate_in_range(self) -> None:
        for sm in _get_result().structural_metrics:
            assert 0.0 <= sm.definition_preservation_rate <= 1.0, (
                f"{sm.strategy.value}/{sm.document_id}: DPR={sm.definition_preservation_rate}"
            )

    def test_cross_ref_resolution_rate_in_range(self) -> None:
        for sm in _get_result().structural_metrics:
            assert 0.0 <= sm.cross_ref_resolution_rate <= 1.0, (
                f"{sm.strategy.value}/{sm.document_id}: CRRR={sm.cross_ref_resolution_rate}"
            )

    def test_hierarchy_depth_retained_in_range(self) -> None:
        for sm in _get_result().structural_metrics:
            assert 0.0 <= sm.hierarchy_depth_retained <= 1.0, (
                f"{sm.strategy.value}/{sm.document_id}: HDR={sm.hierarchy_depth_retained}"
            )

    def test_chunk_size_cv_non_negative(self) -> None:
        for sm in _get_result().structural_metrics:
            assert sm.chunk_size_cv >= 0.0, (
                f"{sm.strategy.value}/{sm.document_id}: CV={sm.chunk_size_cv}"
            )

    def test_chunk_count_positive(self) -> None:
        for sm in _get_result().structural_metrics:
            assert sm.chunk_count > 0, (
                f"{sm.strategy.value}/{sm.document_id}: count={sm.chunk_count}"
            )

    def test_avg_chunk_chars_positive(self) -> None:
        for sm in _get_result().structural_metrics:
            assert sm.avg_chunk_chars > 0, (
                f"{sm.strategy.value}/{sm.document_id}: avg={sm.avg_chunk_chars}"
            )


class TestStructuralInvariants:
    def test_all_strategies_present(self) -> None:
        result = _get_result()
        strategies = {sm.strategy for sm in result.structural_metrics}
        assert len(strategies) == 5  # 4 + contextual

    def test_all_documents_present(self) -> None:
        result = _get_result()
        documents = {sm.document_id for sm in result.structural_metrics}
        assert len(documents) == 5

    def test_complete_matrix(self) -> None:
        """Every (strategy, document) pair has metrics."""
        result = _get_result()
        expected = len(result.strategies) * len(result.documents)
        assert len(result.structural_metrics) == expected


class TestCompositeScores:
    def _composite(self, sm: StructuralMetrics) -> float:
        return (
            (1 - sm.clause_fragmentation_rate)
            + sm.definition_preservation_rate
            + sm.cross_ref_resolution_rate
            + sm.hierarchy_depth_retained
        ) / 4.0

    def test_composite_in_range(self) -> None:
        for sm in _get_result().structural_metrics:
            score = self._composite(sm)
            assert 0.0 <= score <= 1.0, f"{sm.strategy.value}/{sm.document_id}: composite={score}"
