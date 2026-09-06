"""Metric maths for the external evaluations, checked against hand-worked cases."""

from __future__ import annotations

import math

import pytest

from scaffolder.evals.classification import (
    accuracy,
    calibration_curve,
    confusion_pairs,
    macro_f1,
    per_class_scores,
    spearman,
)
from scaffolder.evals.common import (
    Interval,
    bootstrap_ci,
    deterministic_sample,
    markdown_table,
    mean,
    proportion_ci,
    write_json,
)


class TestAccuracy:
    def test_hand_worked(self) -> None:
        assert accuracy(["a", "b", "c", "d"], ["a", "b", "c", "x"]) == 0.75

    def test_perfect_and_zero(self) -> None:
        assert accuracy(["a", "b"], ["a", "b"]) == 1.0
        assert accuracy(["a", "b"], ["b", "a"]) == 0.0

    def test_empty(self) -> None:
        assert accuracy([], []) == 0.0

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="length mismatch"):
            accuracy(["a"], ["a", "b"])


class TestPerClassScores:
    def test_hand_worked_precision_recall(self) -> None:
        # gold:  a a a b b
        # pred:  a a b b x
        # class a: tp=2, predicted=2 -> P=1.0; support=3 -> R=2/3
        # class b: tp=1, predicted=2 -> P=0.5; support=2 -> R=0.5
        scores = {s.label: s for s in per_class_scores(
            ["a", "a", "a", "b", "b"], ["a", "a", "b", "b", "x"]
        )}
        assert scores["a"].precision == 1.0
        assert scores["a"].recall == pytest.approx(2 / 3)
        assert scores["a"].f1 == pytest.approx(2 * 1.0 * (2 / 3) / (1.0 + 2 / 3))
        assert scores["b"].precision == 0.5
        assert scores["b"].recall == 0.5
        assert scores["b"].f1 == 0.5

    def test_labels_default_to_gold_classes_only(self) -> None:
        # "z" is predicted but never gold: it must not dilute the macro average.
        labels = [s.label for s in per_class_scores(["a", "a"], ["a", "z"])]
        assert labels == ["a"]

    def test_zero_division_is_zero_not_nan(self) -> None:
        scores = per_class_scores(["a"], ["b"], labels=["a", "b"])
        for score in scores:
            assert not math.isnan(score.f1)
            assert score.f1 == 0.0

    def test_sorted_by_descending_support(self) -> None:
        scores = per_class_scores(["a", "b", "b", "b", "c", "c"], ["a", "b", "b", "b", "c", "c"])
        assert [s.label for s in scores] == ["b", "c", "a"]

    def test_predicted_count_is_reported(self) -> None:
        scores = {s.label: s for s in per_class_scores(["a", "a"], ["a", "b"], labels=["a", "b"])}
        assert scores["a"].predicted == 1
        assert scores["b"].predicted == 1
        assert scores["b"].support == 0


class TestMacroF1:
    def test_is_unweighted_mean_of_class_f1(self) -> None:
        gold = ["a", "a", "a", "a", "b"]
        pred = ["a", "a", "a", "a", "a"]
        # a: P=4/5 R=1.0 -> F1=8/9 ; b: P=0 R=0 -> F1=0
        assert macro_f1(gold, pred) == pytest.approx((8 / 9) / 2)

    def test_penalises_majority_predictor_more_than_accuracy(self) -> None:
        gold = ["a"] * 9 + ["b"]
        pred = ["a"] * 10
        assert accuracy(gold, pred) == pytest.approx(0.9)
        assert macro_f1(gold, pred) < 0.5

    def test_empty(self) -> None:
        assert macro_f1([], []) == 0.0


class TestConfusionPairs:
    def test_only_disagreements_counted_and_ranked(self) -> None:
        gold = ["a", "a", "a", "b"]
        pred = ["b", "b", "a", "a"]
        assert confusion_pairs(gold, pred) == [("a", "b", 2), ("b", "a", 1)]

    def test_top_n_truncates(self) -> None:
        gold = ["a", "b", "c"]
        pred = ["x", "y", "z"]
        assert len(confusion_pairs(gold, pred, top_n=2)) == 2

    def test_no_errors(self) -> None:
        assert confusion_pairs(["a"], ["a"]) == []


class TestCalibration:
    def test_bins_and_ece_on_a_perfectly_calibrated_case(self) -> None:
        # 10 items at confidence 0.95, 9 correct -> bin accuracy 0.9
        confidences = [0.95] * 10
        correct = [True] * 9 + [False]
        cal = calibration_curve(confidences, correct, n_bins=10)
        assert len(cal.bins) == 1
        assert cal.bins[0].accuracy == pytest.approx(0.9)
        assert cal.bins[0].mean_confidence == pytest.approx(0.95)
        assert cal.expected_calibration_error == pytest.approx(0.05)

    def test_confidence_of_one_lands_in_the_top_bin(self) -> None:
        cal = calibration_curve([1.0], [True], n_bins=10)
        assert cal.bins[0].low == pytest.approx(0.9)
        assert cal.bins[0].high == pytest.approx(1.0)

    def test_monotonic_detection(self) -> None:
        confidences = [0.1] * 10 + [0.9] * 10
        rising = [False] * 9 + [True] + [True] * 10
        falling = [True] * 10 + [False] * 9 + [True]
        assert calibration_curve(confidences, rising).monotonic
        assert not calibration_curve(confidences, falling).monotonic

    def test_spearman_is_positive_when_confidence_tracks_correctness(self) -> None:
        # Correctness is binary, so ties cap the coefficient below 1.0 even for
        # a perfectly ordered case; it must still be strongly positive.
        rising = calibration_curve([0.1, 0.2, 0.8, 0.9], [False, False, True, True])
        falling = calibration_curve([0.1, 0.2, 0.8, 0.9], [True, True, False, False])
        assert rising.spearman > 0.85
        assert falling.spearman == pytest.approx(-rising.spearman)

    def test_empty_bins_are_dropped(self) -> None:
        cal = calibration_curve([0.05, 0.95], [True, True], n_bins=10)
        assert len(cal.bins) == 2

    def test_bad_bin_count_raises(self) -> None:
        with pytest.raises(ValueError, match="n_bins"):
            calibration_curve([0.5], [True], n_bins=0)


class TestSpearman:
    def test_perfect_positive_and_negative(self) -> None:
        assert spearman([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)
        assert spearman([1, 2, 3], [30, 20, 10]) == pytest.approx(-1.0)

    def test_constant_variable_returns_zero(self) -> None:
        assert spearman([1, 1, 1], [1, 2, 3]) == 0.0

    def test_handles_ties(self) -> None:
        assert spearman([1, 1, 2, 2], [1, 1, 2, 2]) == pytest.approx(1.0)

    def test_too_short(self) -> None:
        assert spearman([1], [1]) == 0.0


class TestBootstrap:
    def test_is_deterministic_for_a_fixed_seed(self) -> None:
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        first = bootstrap_ci(data, mean, n_boot=200, seed=7)
        second = bootstrap_ci(data, mean, n_boot=200, seed=7)
        assert first == second

    def test_different_seeds_give_different_intervals(self) -> None:
        data = [float(i) for i in range(50)]
        assert bootstrap_ci(data, mean, n_boot=200, seed=1) != bootstrap_ci(
            data, mean, n_boot=200, seed=2
        )

    def test_interval_brackets_the_point_estimate(self) -> None:
        data = [float(i) for i in range(100)]
        interval = bootstrap_ci(data, mean, n_boot=500, seed=0)
        assert interval.low <= interval.point <= interval.high

    def test_degenerate_inputs(self) -> None:
        assert bootstrap_ci([], mean) == Interval(0.0, 0.0, 0.0, 0)
        single = bootstrap_ci([3.0], mean)
        assert single.low == single.point == single.high == 3.0

    def test_wider_interval_for_smaller_samples(self) -> None:
        wide = bootstrap_ci([0.0, 1.0] * 5, mean, n_boot=500, seed=0)
        narrow = bootstrap_ci([0.0, 1.0] * 200, mean, n_boot=500, seed=0)
        assert (wide.high - wide.low) > (narrow.high - narrow.low)

    def test_proportion_ci(self) -> None:
        interval = proportion_ci([True] * 80 + [False] * 20, n_boot=300, seed=0)
        assert interval.point == pytest.approx(0.8)
        assert 0.7 < interval.low < 0.8 < interval.high < 0.9


class TestIntervalFormatting:
    def test_percent_rendering(self) -> None:
        assert Interval(0.8123, 0.79, 0.83, 100).format() == "81.2% [79.0, 83.0]"

    def test_raw_rendering(self) -> None:
        assert Interval(0.5, 0.4, 0.6, 10).format(pct=False, places=2) == "0.50 [0.40, 0.60]"


class TestHelpers:
    def test_markdown_table_shape(self) -> None:
        table = markdown_table(["a", "bb"], [["1", "2"], ["3", "4"]])
        lines = table.splitlines()
        assert len(lines) == 4
        assert all(line.startswith("| ") and line.endswith(" |") for line in lines)
        assert set(lines[1].replace("|", "").replace(" ", "")) == {"-"}

    def test_markdown_table_no_rows(self) -> None:
        assert len(markdown_table(["a"], []).splitlines()) == 2

    def test_deterministic_sample_is_stable_and_ordered(self) -> None:
        population = list(range(100))
        first = deterministic_sample(population, 10, seed=3)
        assert first == deterministic_sample(population, 10, seed=3)
        assert first == sorted(first)
        assert len(first) == 10

    def test_deterministic_sample_returns_all_when_k_is_large_or_zero(self) -> None:
        population = [1, 2, 3]
        assert deterministic_sample(population, 0, seed=0) == population
        assert deterministic_sample(population, 99, seed=0) == population

    def test_write_json_round_trips(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        import json

        target = write_json({"b": 1, "a": 2}, tmp_path / "nested" / "out.json")
        assert json.loads(target.read_text(encoding="utf-8")) == {"a": 2, "b": 1}

    def test_mean_of_empty(self) -> None:
        assert mean([]) == 0.0
