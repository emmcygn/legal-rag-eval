"""Tests for statistical significance testing."""

from __future__ import annotations

import math

import pytest

from legal_rag_eval.metrics.statistical import (
    SIGNIFICANCE_METRICS,
    ComparisonResult,
    cohens_d_paired,
    compute_all_comparisons,
    compute_all_significance,
    compute_significance,
    holm_correction,
    leave_one_document_out,
    paired_bootstrap_ci,
    paired_t_test,
    rank_biserial_correlation,
    wilcoxon_signed_rank,
)
from legal_rag_eval.models import (
    EmbeddingModelName,
    RetrievalMetrics,
    StrategyName,
)


class TestPairedTTest:
    def test_significant_difference(self) -> None:
        a = [0.8, 0.9, 0.7, 0.85, 0.75]
        b = [0.5, 0.6, 0.4, 0.55, 0.45]
        t_stat, p_value, significant, cohens_d = paired_t_test(a, b)
        assert significant
        assert p_value < 0.05
        assert cohens_d > 0.8  # large effect

    def test_identical_values_not_significant(self) -> None:
        a = [0.5, 0.5, 0.5, 0.5, 0.5]
        b = [0.5, 0.5, 0.5, 0.5, 0.5]
        t_stat, p_value, significant, cohens_d = paired_t_test(a, b)
        assert not significant
        assert cohens_d == 0.0

    def test_single_value_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            paired_t_test([0.5], [0.5])

    def test_mismatched_lengths_raises(self) -> None:
        with pytest.raises(ValueError, match="lengths must match"):
            paired_t_test([0.5, 0.6], [0.5])


class TestCohensDPaired:
    def test_zero_variance_nonzero_mean_returns_positive_inf(self) -> None:
        # Every pair improves by exactly 0.1 -- a perfectly consistent,
        # non-zero effect. The old implementation silently returned 0.0 here.
        a = [0.6, 0.6, 0.6, 0.6]
        b = [0.5, 0.5, 0.5, 0.5]
        assert cohens_d_paired(a, b) == math.inf

    def test_zero_variance_nonzero_mean_returns_negative_inf(self) -> None:
        a = [0.5, 0.5, 0.5, 0.5]
        b = [0.6, 0.6, 0.6, 0.6]
        assert cohens_d_paired(a, b) == -math.inf

    def test_zero_variance_zero_mean_returns_zero(self) -> None:
        a = [0.5, 0.5, 0.5, 0.5]
        b = [0.5, 0.5, 0.5, 0.5]
        assert cohens_d_paired(a, b) == 0.0

    def test_matches_paired_t_test(self) -> None:
        a = [0.8, 0.9, 0.7, 0.85, 0.75]
        b = [0.5, 0.6, 0.4, 0.55, 0.45]
        _, _, _, cohens_d = paired_t_test(a, b)
        assert cohens_d == cohens_d_paired(a, b)


class TestHolmCorrection:
    def test_hand_computed_example(self) -> None:
        # m=4, ascending p-values already sorted.
        # ranks (1-indexed) 1..4 -> factors 4,3,2,1
        # raw: 0.04, 0.06, 0.06, 0.04 -> cumulative max: 0.04, 0.06, 0.06, 0.06
        p_values = [0.01, 0.02, 0.03, 0.04]
        adjusted = holm_correction(p_values)
        assert adjusted == pytest.approx([0.04, 0.06, 0.06, 0.06])

    def test_empty_list(self) -> None:
        assert holm_correction([]) == []

    def test_single_value_passes_through(self) -> None:
        assert holm_correction([0.03]) == pytest.approx([0.03])

    def test_monotone_in_sorted_order(self) -> None:
        p_values = [0.2, 0.001, 0.04, 0.03, 0.5]
        adjusted = holm_correction(p_values)
        order = sorted(range(len(p_values)), key=lambda i: p_values[i])
        adjusted_in_rank_order = [adjusted[i] for i in order]
        assert adjusted_in_rank_order == sorted(adjusted_in_rank_order)

    def test_clamped_to_one(self) -> None:
        adjusted = holm_correction([0.9, 0.9, 0.9])
        assert all(p <= 1.0 for p in adjusted)

    def test_order_independent_per_original_position(self) -> None:
        ascending = [0.01, 0.02, 0.03, 0.04]
        descending = list(reversed(ascending))
        adjusted_ascending = holm_correction(ascending)
        adjusted_descending = holm_correction(descending)
        assert adjusted_descending == pytest.approx(list(reversed(adjusted_ascending)))


class TestRankBiserialCorrelation:
    def test_all_positive_differences(self) -> None:
        a = [0.9, 0.8, 0.7, 0.6]
        b = [0.5, 0.4, 0.3, 0.2]
        assert rank_biserial_correlation(a, b) == pytest.approx(1.0)

    def test_all_negative_differences(self) -> None:
        a = [0.5, 0.4, 0.3, 0.2]
        b = [0.9, 0.8, 0.7, 0.6]
        assert rank_biserial_correlation(a, b) == pytest.approx(-1.0)

    def test_all_ties_returns_zero(self) -> None:
        a = [0.5, 0.5, 0.5]
        b = [0.5, 0.5, 0.5]
        assert rank_biserial_correlation(a, b) == 0.0


class TestWilcoxonSignedRank:
    def test_agrees_in_direction_with_t_test(self) -> None:
        # n=5 caps the Wilcoxon exact two-sided p-value at 0.0625 even when
        # every difference has the same sign, so use n=6 to allow p < 0.05.
        a = [0.8, 0.9, 0.7, 0.85, 0.75, 0.82]
        b = [0.5, 0.6, 0.4, 0.55, 0.45, 0.52]
        _, t_p_value, _, _ = paired_t_test(a, b)
        _, wilcoxon_p_value = wilcoxon_signed_rank(a, b)
        assert t_p_value < 0.05
        assert wilcoxon_p_value < 0.05

    def test_all_ties_returns_zero_one_without_raising(self) -> None:
        a = [0.5, 0.5, 0.5, 0.5]
        b = [0.5, 0.5, 0.5, 0.5]
        statistic, p_value = wilcoxon_signed_rank(a, b)
        assert (statistic, p_value) == (0.0, 1.0)


class TestPairedBootstrapCI:
    def test_known_signal_excludes_zero(self) -> None:
        a = [0.80, 0.85, 0.75, 0.90, 0.83, 0.77, 0.88, 0.73, 0.82, 0.79]
        b = [0.55] * 10
        ci_low, ci_high = paired_bootstrap_ci(a, b, seed=0)
        assert ci_low > 0.0
        assert ci_high > 0.0

    def test_all_zero_difference(self) -> None:
        a = [0.5] * 10
        b = [0.5] * 10
        ci_low, ci_high = paired_bootstrap_ci(a, b, seed=0)
        assert ci_low == pytest.approx(0.0)
        assert ci_high == pytest.approx(0.0)

    def test_deterministic_for_same_seed(self) -> None:
        a = [0.80, 0.85, 0.75, 0.90, 0.83, 0.77, 0.88, 0.73, 0.82, 0.79]
        b = [0.55] * 10
        result_1 = paired_bootstrap_ci(a, b, seed=42)
        result_2 = paired_bootstrap_ci(a, b, seed=42)
        assert result_1 == result_2

    def test_different_seeds_can_differ(self) -> None:
        a = [0.80, 0.85, 0.75, 0.90, 0.83, 0.77, 0.88, 0.73, 0.82, 0.79]
        b = [0.55] * 10
        result_seed_0 = paired_bootstrap_ci(a, b, seed=0)
        result_seed_1 = paired_bootstrap_ci(a, b, seed=1)
        assert result_seed_0 != result_seed_1

    def test_ci_brackets_observed_mean_delta(self) -> None:
        a = [0.80, 0.85, 0.75, 0.90, 0.83, 0.77, 0.88, 0.73, 0.82, 0.79]
        b = [0.55] * 10
        mean_delta = sum(x - y for x, y in zip(a, b, strict=True)) / len(a)
        ci_low, ci_high = paired_bootstrap_ci(a, b, seed=0)
        assert ci_low <= mean_delta <= ci_high

    def test_fewer_than_two_pairs_returns_nan(self) -> None:
        ci_low, ci_high = paired_bootstrap_ci([0.5], [0.4])
        assert math.isnan(ci_low)
        assert math.isnan(ci_high)


class TestLeaveOneDocumentOut:
    def test_single_document_carries_whole_effect(self) -> None:
        # d1's queries all show a strong +0.3 improvement; d2's queries show
        # no improvement at all. The overall effect is entirely due to d1.
        values_a = [0.9, 0.9, 0.9, 0.5, 0.5, 0.5]
        values_b = [0.6, 0.6, 0.6, 0.5, 0.5, 0.5]
        document_ids = ["d1", "d1", "d1", "d2", "d2", "d2"]

        result = leave_one_document_out(values_a, values_b, document_ids)

        assert result["d1"] == pytest.approx(0.0)  # remove the whole effect
        assert result["d2"] == pytest.approx(0.3)  # effect fully preserved

    def test_document_leaving_fewer_than_two_pairs_is_skipped(self) -> None:
        values_a = [0.9, 0.8, 0.5]
        values_b = [0.6, 0.5, 0.5]
        document_ids = ["d1", "d1", "d2"]

        result = leave_one_document_out(values_a, values_b, document_ids)

        # removing d1 leaves only 1 pair (d2) -> skipped
        assert "d1" not in result
        # removing d2 leaves 2 pairs (d1) -> kept
        assert "d2" in result

    def test_mismatched_lengths_raises(self) -> None:
        with pytest.raises(ValueError, match="equal length"):
            leave_one_document_out([0.5, 0.6], [0.5], ["d1", "d2"])


def _make_metrics(
    strategy: StrategyName,
    model: EmbeddingModelName,
    query_ids: list[str],
    p5_values: list[float],
) -> list[RetrievalMetrics]:
    """Create mock RetrievalMetrics with specified P@5 values."""
    return [
        RetrievalMetrics(
            query_id=qid,
            strategy=strategy,
            embedding_model=model,
            precision_at_1=v,
            precision_at_3=v,
            precision_at_5=v,
            precision_at_10=v * 0.8,
            recall_at_1=v * 0.5,
            recall_at_3=v * 0.7,
            recall_at_5=v * 0.9,
            recall_at_10=v,
            mrr=v,
            ndcg_at_10=v,
            drm_hit=False,
        )
        for qid, v in zip(query_ids, p5_values, strict=True)
    ]


class TestComputeSignificance:
    def test_produces_one_result_per_metric(self) -> None:
        queries = ["q1", "q2", "q3", "q4", "q5"]
        lc = _make_metrics(
            StrategyName.LEXICHUNK, EmbeddingModelName.MINILM, queries, [0.8, 0.9, 0.7, 0.85, 0.75]
        )
        bl = _make_metrics(
            StrategyName.RCTS, EmbeddingModelName.MINILM, queries, [0.5, 0.6, 0.4, 0.55, 0.45]
        )

        results = compute_significance(lc, bl, StrategyName.RCTS)
        assert len(results) == len(SIGNIFICANCE_METRICS)

    def test_too_few_queries_returns_empty(self) -> None:
        queries = ["q1"]
        lc = _make_metrics(StrategyName.LEXICHUNK, EmbeddingModelName.MINILM, queries, [0.8])
        bl = _make_metrics(StrategyName.RCTS, EmbeddingModelName.MINILM, queries, [0.5])

        results = compute_significance(lc, bl, StrategyName.RCTS)
        assert results == []


class TestComputeAllSignificance:
    def test_groups_by_model(self) -> None:
        queries = ["q1", "q2", "q3"]
        lc = _make_metrics(
            StrategyName.LEXICHUNK, EmbeddingModelName.MINILM, queries, [0.8, 0.9, 0.7]
        )
        rcts = _make_metrics(StrategyName.RCTS, EmbeddingModelName.MINILM, queries, [0.5, 0.6, 0.4])

        results = compute_all_significance(lc + rcts)
        assert len(results) == len(SIGNIFICANCE_METRICS)  # 1 baseline x 1 model x N metrics


class TestComputeAllComparisons:
    def test_lexichunk_contextual_is_tested(self) -> None:
        # Regression test: the old compute_all_significance silently excluded
        # LEXICHUNK_CONTEXTUAL from ever being tested as a treatment.
        queries = ["q1", "q2", "q3", "q4", "q5"]
        lc = _make_metrics(
            StrategyName.LEXICHUNK,
            EmbeddingModelName.MINILM,
            queries,
            [0.8, 0.9, 0.7, 0.85, 0.75],
        )
        lc_ctx = _make_metrics(
            StrategyName.LEXICHUNK_CONTEXTUAL,
            EmbeddingModelName.MINILM,
            queries,
            [0.82, 0.92, 0.72, 0.87, 0.77],
        )
        rcts = _make_metrics(
            StrategyName.RCTS, EmbeddingModelName.MINILM, queries, [0.5, 0.6, 0.4, 0.55, 0.45]
        )

        results = compute_all_comparisons(lc + lc_ctx + rcts)

        strategies_a = {r.strategy_a for r in results}
        assert StrategyName.LEXICHUNK in strategies_a
        assert StrategyName.LEXICHUNK_CONTEXTUAL in strategies_a
        assert all(r.strategy_b == StrategyName.RCTS for r in results)

    def test_holm_applied_across_whole_family_not_per_baseline(self) -> None:
        queries = ["q1", "q2", "q3", "q4", "q5"]
        lc = _make_metrics(
            StrategyName.LEXICHUNK,
            EmbeddingModelName.MINILM,
            queries,
            [0.8, 0.9, 0.7, 0.85, 0.75],
        )
        rcts = _make_metrics(
            StrategyName.RCTS, EmbeddingModelName.MINILM, queries, [0.5, 0.6, 0.4, 0.55, 0.45]
        )
        fixed = _make_metrics(
            StrategyName.FIXED_SIZE,
            EmbeddingModelName.MINILM,
            queries,
            [0.55, 0.65, 0.45, 0.6, 0.5],
        )

        results = compute_all_comparisons(lc + rcts + fixed)

        # There are 2 baselines x len(SIGNIFICANCE_METRICS) metrics in this family.
        assert len(results) == 2 * len(SIGNIFICANCE_METRICS)

        expected_holm = holm_correction([r.p_value_t for r in results])
        actual_holm = [r.p_value_holm for r in results]
        assert actual_holm == pytest.approx(expected_holm)

    def test_n_equals_number_of_common_queries(self) -> None:
        lc = _make_metrics(
            StrategyName.LEXICHUNK,
            EmbeddingModelName.MINILM,
            ["q1", "q2", "q3", "q4", "q5"],
            [0.8, 0.9, 0.7, 0.85, 0.75],
        )
        rcts = _make_metrics(
            StrategyName.RCTS,
            EmbeddingModelName.MINILM,
            ["q1", "q2", "q3"],  # only 3 queries overlap
            [0.5, 0.6, 0.4],
        )

        results = compute_all_comparisons(lc + rcts)
        assert results
        assert all(r.n == 3 for r in results)

    def test_deterministic_for_same_seed(self) -> None:
        queries = ["q1", "q2", "q3", "q4", "q5"]
        lc = _make_metrics(
            StrategyName.LEXICHUNK,
            EmbeddingModelName.MINILM,
            queries,
            [0.8, 0.9, 0.7, 0.85, 0.75],
        )
        rcts = _make_metrics(
            StrategyName.RCTS, EmbeddingModelName.MINILM, queries, [0.5, 0.6, 0.4, 0.55, 0.45]
        )

        results_1 = compute_all_comparisons(lc + rcts, seed=7)
        results_2 = compute_all_comparisons(lc + rcts, seed=7)
        assert results_1 == results_2

    def test_too_few_common_queries_skipped(self) -> None:
        lc = _make_metrics(StrategyName.LEXICHUNK, EmbeddingModelName.MINILM, ["q1"], [0.8])
        rcts = _make_metrics(StrategyName.RCTS, EmbeddingModelName.MINILM, ["q1"], [0.5])
        assert compute_all_comparisons(lc + rcts) == []

    def test_query_documents_mapping_used_for_lodo(self) -> None:
        # docA (3 queries) carries almost the entire effect (diff=0.4 each);
        # docB (2 queries) barely moves (diff=0.05 each). Unequal group sizes
        # so removing docA moves the delta further than removing docB does.
        queries = ["q1", "q2", "q3", "q4", "q5"]
        lc = _make_metrics(
            StrategyName.LEXICHUNK,
            EmbeddingModelName.MINILM,
            queries,
            [0.9, 0.9, 0.9, 0.6, 0.6],
        )
        rcts = _make_metrics(
            StrategyName.RCTS,
            EmbeddingModelName.MINILM,
            queries,
            [0.5, 0.5, 0.5, 0.55, 0.55],
        )
        query_documents = {
            "q1": "docA",
            "q2": "docA",
            "q3": "docA",
            "q4": "docB",
            "q5": "docB",
        }

        results = compute_all_comparisons(lc + rcts, query_documents=query_documents)

        precision_at_5 = next(r for r in results if r.metric_name == "precision_at_5")
        assert precision_at_5.n_documents == 2
        assert precision_at_5.lodo_worst_document == "docA"

    def test_to_significance_result_adapter(self) -> None:
        queries = ["q1", "q2", "q3", "q4", "q5"]
        lc = _make_metrics(
            StrategyName.LEXICHUNK,
            EmbeddingModelName.MINILM,
            queries,
            [0.8, 0.9, 0.7, 0.85, 0.75],
        )
        rcts = _make_metrics(
            StrategyName.RCTS, EmbeddingModelName.MINILM, queries, [0.5, 0.6, 0.4, 0.55, 0.45]
        )

        comparison = compute_all_comparisons(lc + rcts)[0]
        legacy = comparison.to_significance_result()

        assert isinstance(comparison, ComparisonResult)
        assert legacy.metric_name == comparison.metric_name
        assert legacy.p_value == comparison.p_value_holm
        assert legacy.significant == comparison.significant_holm
        assert legacy.effect_size == comparison.cohens_d
        assert legacy.n_queries == comparison.n
