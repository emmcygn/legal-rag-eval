"""Classification metrics used by the LEDGAR evaluation.

Implemented from scratch (rather than pulled from scikit-learn) so the metric
definitions are auditable and so the numbers reported in
``docs/external_evals.md`` do not depend on a library version.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True, slots=True)
class ClassScore:
    """Precision / recall / F1 and support for a single class."""

    label: str
    precision: float
    recall: float
    f1: float
    support: int
    predicted: int

    def as_dict(self) -> dict[str, float | int | str]:
        """Return a JSON-friendly mapping."""
        return {
            "label": self.label,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "support": self.support,
            "predicted": self.predicted,
        }


def accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    """Fraction of positions where prediction equals truth."""
    _check_lengths(y_true, y_pred)
    if not y_true:
        return 0.0
    hits = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == p)
    return hits / len(y_true)


def per_class_scores(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str] | None = None,
) -> list[ClassScore]:
    """Per-class precision, recall and F1.

    A class with no predictions gets precision 0.0 (not NaN) and a class with
    no gold instances gets recall 0.0, matching scikit-learn's
    ``zero_division=0`` convention.

    Args:
        y_true: Gold labels.
        y_pred: Predicted labels, aligned with ``y_true``.
        labels: Classes to score. Defaults to every label appearing in
            ``y_true`` (so classes the model hallucinates but that never occur
            in the gold data do not dilute the macro average).

    Returns:
        Scores sorted by descending support, then label.
    """
    _check_lengths(y_true, y_pred)
    if labels is None:
        labels = sorted(set(y_true))

    support = Counter(y_true)
    predicted = Counter(y_pred)
    correct: Counter[str] = Counter(t for t, p in zip(y_true, y_pred, strict=True) if t == p)

    scores: list[ClassScore] = []
    for label in labels:
        tp = correct[label]
        precision = tp / predicted[label] if predicted[label] else 0.0
        recall = tp / support[label] if support[label] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        scores.append(
            ClassScore(
                label=label,
                precision=precision,
                recall=recall,
                f1=f1,
                support=support[label],
                predicted=predicted[label],
            )
        )
    scores.sort(key=lambda s: (-s.support, s.label))
    return scores


def macro_f1(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str] | None = None,
) -> float:
    """Unweighted mean of per-class F1 over ``labels``."""
    scores = per_class_scores(y_true, y_pred, labels)
    if not scores:
        return 0.0
    return sum(s.f1 for s in scores) / len(scores)


def confusion_pairs(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    top_n: int = 15,
) -> list[tuple[str, str, int]]:
    """Most frequent ``(gold, predicted)`` pairs where the two disagree."""
    _check_lengths(y_true, y_pred)
    counts: Counter[tuple[str, str]] = Counter(
        (t, p) for t, p in zip(y_true, y_pred, strict=True) if t != p
    )
    return [(t, p, n) for (t, p), n in counts.most_common(top_n)]


@dataclass(frozen=True, slots=True)
class CalibrationBin:
    """One bucket of the confidence-versus-accuracy curve."""

    low: float
    high: float
    count: int
    mean_confidence: float
    accuracy: float

    def as_dict(self) -> dict[str, float | int]:
        """Return a JSON-friendly mapping."""
        return {
            "low": self.low,
            "high": self.high,
            "count": self.count,
            "mean_confidence": self.mean_confidence,
            "accuracy": self.accuracy,
        }


@dataclass(frozen=True, slots=True)
class Calibration:
    """A binned calibration curve plus summary statistics."""

    bins: list[CalibrationBin]
    expected_calibration_error: float
    monotonic: bool
    spearman: float

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-friendly mapping."""
        return {
            "bins": [b.as_dict() for b in self.bins],
            "expected_calibration_error": self.expected_calibration_error,
            "monotonic": self.monotonic,
            "spearman": self.spearman,
        }


def calibration_curve(
    confidences: Sequence[float],
    correct: Sequence[bool],
    n_bins: int = 10,
) -> Calibration:
    """Bin predictions by confidence and measure accuracy within each bin.

    Bins are equal-width over ``[0, 1]``; the top bin is closed on the right so
    a confidence of exactly 1.0 lands in it. Empty bins are dropped.

    Returns:
        A :class:`Calibration` carrying the non-empty bins, the expected
        calibration error (support-weighted mean absolute gap between mean
        confidence and accuracy), whether bin accuracy is non-decreasing, and
        the Spearman rank correlation between per-item confidence and
        correctness (a direct answer to "does higher confidence mean higher
        accuracy?").
    """
    _check_lengths(confidences, correct)
    if n_bins < 1:
        msg = f"n_bins must be >= 1, got {n_bins}"
        raise ValueError(msg)

    buckets: list[list[bool]] = [[] for _ in range(n_bins)]
    bucket_conf: list[list[float]] = [[] for _ in range(n_bins)]
    for conf, ok in zip(confidences, correct, strict=True):
        clamped = min(max(conf, 0.0), 1.0)
        index = min(int(clamped * n_bins), n_bins - 1)
        buckets[index].append(bool(ok))
        bucket_conf[index].append(clamped)

    total = len(confidences)
    bins: list[CalibrationBin] = []
    ece = 0.0
    for i, flags in enumerate(buckets):
        if not flags:
            continue
        acc = sum(1 for f in flags if f) / len(flags)
        conf = sum(bucket_conf[i]) / len(bucket_conf[i])
        bins.append(
            CalibrationBin(
                low=i / n_bins,
                high=(i + 1) / n_bins,
                count=len(flags),
                mean_confidence=conf,
                accuracy=acc,
            )
        )
        ece += (len(flags) / total) * abs(conf - acc)

    accs = [b.accuracy for b in bins]
    monotonic = all(a <= b for a, b in zip(accs, accs[1:], strict=False))
    return Calibration(
        bins=bins,
        expected_calibration_error=ece,
        monotonic=monotonic,
        spearman=spearman(list(confidences), [1.0 if c else 0.0 for c in correct]),
    )


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Spearman rank correlation with average ranks for ties.

    Returns 0.0 when either variable is constant (correlation undefined).
    """
    _check_lengths(xs, ys)
    n = len(xs)
    if n < 2:
        return 0.0
    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num: float = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    dx: float = sum((a - mx) ** 2 for a in rx)
    dy: float = sum((b - my) ** 2 for b in ry)
    if dx <= 0.0 or dy <= 0.0:
        return 0.0
    return num / (math.sqrt(dx) * math.sqrt(dy))


def _ranks(values: Sequence[float]) -> list[float]:
    """Average ranks (1-based), ties sharing the mean of their positions."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        average = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = average
        i = j + 1
    return ranks


def _check_lengths(a: Sequence[object], b: Sequence[object]) -> None:
    """Raise if two aligned sequences differ in length."""
    if len(a) != len(b):
        msg = f"length mismatch: {len(a)} != {len(b)}"
        raise ValueError(msg)
