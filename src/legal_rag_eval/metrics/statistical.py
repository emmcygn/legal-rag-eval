"""Statistical significance testing for benchmark comparisons.

This module underwent an audit that found the previous implementation
indefensible: 21 uncorrected two-tailed paired t-tests at n=22, no confidence
intervals, no distribution-free test, a Cohen's d that silently returned 0.0
for a zero-variance non-zero-mean difference, an ``improvement_pct`` computed
as a fragile ratio of two small means, and no report of how much any result
depends on a single document.

The fix lives in two layers:

* Low-level, pure statistical primitives (``paired_bootstrap_ci``,
  ``wilcoxon_signed_rank``, ``rank_biserial_correlation``,
  ``cohens_d_paired``, ``holm_correction``, ``leave_one_document_out``) that
  can be tested and reasoned about independently.
* ``compute_all_comparisons``, which assembles those primitives into a
  ``ComparisonResult`` per (metric, treatment, baseline, embedding model)
  tuple, and applies the Holm correction across the *entire* family of tests
  run in one pass (not per baseline).

The legacy ``paired_t_test`` / ``compute_significance`` /
``compute_all_significance`` / ``SIGNIFICANCE_METRICS`` names are kept
importable with unchanged signatures and return types so existing callers
(the CLI entry point, the reporting layer, and their tests) keep working.
``ComparisonResult.to_significance_result()`` adapts a rich result back into
the legacy ``SignificanceResult`` shape for JSON export.
"""

from __future__ import annotations

import logging
import math
import re
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np
from scipy import stats  # type: ignore[import-untyped]

from legal_rag_eval.models import (
    SignificanceResult,
    StrategyName,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from legal_rag_eval.models import RetrievalMetrics

logger = logging.getLogger(__name__)

SIGNIFICANCE_METRICS = [
    "precision_at_1",
    "precision_at_5",
    "precision_at_10",
    "recall_at_5",
    "recall_at_10",
    "mrr",
    "ndcg_at_10",
]

# Best-effort fallback for grouping queries into documents when the caller of
# `compute_all_comparisons` does not supply an explicit query_id -> document_id
# mapping. Query ids in this codebase are conventionally
# f"{document_prefix}_q{n}" (e.g. "eu_gdpr_q1"), so stripping the trailing
# "_q<suffix>" token recovers a stable (if not always exact) grouping key.
# This is a heuristic, not a real document id: it is only used to partition
# queries for leave-one-document-out, not to look anything up elsewhere.
_QUERY_ID_SUFFIX_RE = re.compile(r"_q[0-9a-zA-Z]*$")


def _infer_document_id(query_id: str) -> str:
    """Best-effort document grouping key derived from a query id's prefix.

    Strips a trailing ``_q<suffix>`` token (the convention used by this
    project's query fixtures, e.g. ``"eu_gdpr_q1"`` -> ``"eu_gdpr"``). This is
    a heuristic fallback for when no explicit ``query_id -> document_id``
    mapping is available: it groups queries that plausibly came from the same
    document, but it does NOT tell you the true fixture document id (which
    may include additional suffix words, e.g. the real id is
    "eu_gdpr_excerpt", not "eu_gdpr"). Use the real mapping whenever you have
    one.
    """
    stripped = _QUERY_ID_SUFFIX_RE.sub("", query_id)
    return stripped if stripped else query_id


# -- Rich comparison result ---------------------------------------------------


@dataclass(frozen=True, slots=True)
class ComparisonResult:
    """Full statistical comparison of one metric between two strategies.

    Produced by `compute_all_comparisons`. Carries both a parametric (paired
    t-test, Cohen's d) and a distribution-free (Wilcoxon signed-rank,
    rank-biserial correlation) view of the same paired sample, a bootstrap
    confidence interval on the headline delta, a Holm-adjusted p-value
    computed across the whole family of tests run in one call, and a
    leave-one-document-out sensitivity range.
    """

    metric_name: str
    strategy_a: StrategyName
    strategy_b: StrategyName
    embedding_model: str
    n: int  # number of paired observations (queries)
    mean_a: float
    mean_b: float
    delta: float  # mean_a - mean_b, the headline figure
    ci_low: float  # paired bootstrap 95% CI on delta
    ci_high: float
    t_statistic: float
    p_value_t: float  # paired t-test
    p_value_wilcoxon: float  # Wilcoxon signed-rank
    p_value_holm: float  # Holm-adjusted p, over the whole family of tests
    significant_holm: bool  # p_value_holm < alpha
    cohens_d: float  # may be inf; never silently 0.0
    rank_biserial: float  # effect size matching Wilcoxon, in [-1, 1]
    n_documents: int
    lodo_min_delta: float  # smallest leave-one-document-out delta
    lodo_max_delta: float  # largest leave-one-document-out delta
    lodo_worst_document: str  # document whose removal moves delta furthest

    def to_significance_result(self) -> SignificanceResult:
        """Adapt to the legacy `SignificanceResult` shape.

        Maps ``p_value_holm`` -> ``p_value``, ``significant_holm`` ->
        ``significant``, ``cohens_d`` -> ``effect_size``, and recomputes the
        old ``improvement_pct`` ratio-of-means formula purely for backward
        compatibility with existing JSON export and reporting code. That
        formula is unstable when ``mean_b`` is small — prefer ``delta`` and
        ``ci_low``/``ci_high`` from this object for anything new.
        """
        improvement_pct = (
            (self.mean_a - self.mean_b) / self.mean_b * 100 if self.mean_b > 0 else 0.0
        )
        return SignificanceResult(
            metric_name=self.metric_name,
            strategy_a=self.strategy_a,
            strategy_b=self.strategy_b,
            mean_a=self.mean_a,
            mean_b=self.mean_b,
            improvement_pct=improvement_pct,
            t_statistic=self.t_statistic,
            p_value=self.p_value_holm,
            significant=self.significant_holm,
            effect_size=self.cohens_d,
            n_queries=self.n,
        )


# -- Statistical primitives ---------------------------------------------------


def paired_bootstrap_ci(
    values_a: Sequence[float],
    values_b: Sequence[float],
    *,
    n_resamples: int = 10000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval on the mean paired difference.

    Resamples the *pairs* (not `a` and `b` independently) with replacement
    ``n_resamples`` times using ``numpy.random.default_rng(seed)``, computes
    the mean of ``a - b`` for each resample, and returns the
    ``(1 - confidence) / 2`` and ``1 - (1 - confidence) / 2`` percentiles of
    that bootstrap distribution.

    Returns a tuple ``(ci_low, ci_high)`` in the same units as the metric
    (e.g. a difference of precision scores lies in [-1, 1]). Deterministic
    for a fixed ``seed`` and input; a different seed gives a slightly
    different interval since it is a Monte Carlo estimate. Requires at least
    2 paired observations; returns ``(nan, nan)`` otherwise. Does NOT tell
    you whether the underlying metric is normally distributed, and does NOT
    correct for multiple comparisons.
    """
    a = np.asarray(values_a, dtype=np.float64)
    b = np.asarray(values_b, dtype=np.float64)
    n = a.shape[0]
    if n < 2:
        return float("nan"), float("nan")

    diff = a - b
    rng = np.random.default_rng(seed)
    resample_idx = rng.integers(0, n, size=(n_resamples, n))
    resample_means = diff[resample_idx].mean(axis=1)

    alpha = 1.0 - confidence
    lo_pct = 100.0 * (alpha / 2.0)
    hi_pct = 100.0 * (1.0 - alpha / 2.0)
    ci_low, ci_high = np.percentile(resample_means, [lo_pct, hi_pct])
    return float(ci_low), float(ci_high)


def wilcoxon_signed_rank(
    values_a: Sequence[float],
    values_b: Sequence[float],
) -> tuple[float, float]:
    """Wilcoxon signed-rank test on the paired differences ``a - b``.

    Distribution-free alternative to the paired t-test: it does not assume
    the differences are normally distributed, only that they are
    symmetrically distributed around the median under the null hypothesis.
    Uses ``zero_method="wilcox"`` (zero differences are dropped before
    ranking) and a two-sided alternative.

    Returns ``(statistic, p_value)``. When every paired difference is zero,
    scipy has nothing left to rank and raises `ValueError`; that case is
    caught and reported as ``(0.0, 1.0)`` (no evidence of any difference).
    Does NOT tell you the size of the effect — pair with
    `rank_biserial_correlation` for that.
    """
    a = np.asarray(values_a, dtype=np.float64)
    b = np.asarray(values_b, dtype=np.float64)
    try:
        statistic, p_value = stats.wilcoxon(a, b, zero_method="wilcox")
    except ValueError:
        return 0.0, 1.0
    return float(statistic), float(p_value)


def rank_biserial_correlation(
    values_a: Sequence[float],
    values_b: Sequence[float],
) -> float:
    """Matched-pairs rank-biserial correlation, the effect size for Wilcoxon.

    Computed as ``(sum of ranks of positive differences - sum of ranks of
    negative differences) / (sum of all ranks)``, where ranks are assigned to
    ``|a - b|`` after dropping zero differences. Ranges over ``[-1, 1]``:
    ``+1`` means every non-zero difference favored `a`, ``-1`` means every
    one favored `b`, ``0`` means the positive and negative differences are
    balanced (including the case where every difference is zero). Does NOT
    tell you statistical significance — pair with `wilcoxon_signed_rank`.
    """
    a = np.asarray(values_a, dtype=np.float64)
    b = np.asarray(values_b, dtype=np.float64)
    diff = a - b
    nonzero = diff[diff != 0]
    if nonzero.size == 0:
        return 0.0

    ranks = stats.rankdata(np.abs(nonzero))
    positive_sum = float(ranks[nonzero > 0].sum())
    negative_sum = float(ranks[nonzero < 0].sum())
    total = positive_sum + negative_sum
    if total == 0.0:
        return 0.0
    return (positive_sum - negative_sum) / total


def cohens_d_paired(
    values_a: Sequence[float],
    values_b: Sequence[float],
) -> float:
    """Cohen's d for paired samples: mean(a - b) / sample_sd(a - b, ddof=1).

    A standardized effect size: how many standard deviations of the
    (paired) difference does the mean difference represent. Conventionally
    |d| ~ 0.2/0.5/0.8 are read as small/medium/large, but that convention
    was calibrated on other fields and does not automatically transfer here.

    When the differences have zero variance (every pair has exactly the same
    difference) the ratio is undefined in the usual sense: if the mean
    difference is also zero this returns ``0.0`` (no effect at all), but if
    every pair improved by the same non-zero amount this returns ``+inf`` or
    ``-inf`` (a perfectly consistent, arbitrarily precisely estimated
    effect) rather than silently reporting ``0.0``, which was the bug this
    replaces. Does NOT tell you whether the effect is practically
    meaningful, only its size relative to the observed spread.
    """
    a = np.asarray(values_a, dtype=np.float64)
    b = np.asarray(values_b, dtype=np.float64)
    diff = a - b
    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1)) if diff.size > 1 else 0.0

    if std_diff == 0.0:
        if mean_diff == 0.0:
            return 0.0
        return math.inf if mean_diff > 0.0 else -math.inf
    return mean_diff / std_diff


def holm_correction(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values for a family of tests.

    Given ``m`` p-values, sorts them ascending, multiplies the ``i``-th
    smallest (1-indexed) by ``m - i + 1``, then enforces monotonicity by
    taking the running maximum, and clamps every result to at most 1.0.
    Returns the adjusted p-values in the SAME ORDER as the input (not
    sorted), so ``result[i]`` corresponds to ``p_values[i]``.

    This controls the family-wise error rate across the whole family passed
    in — it must be called once over every test run together (every metric x
    baseline x embedding model), not once per baseline, or it under-corrects.
    An empty input returns an empty list. Does NOT tell you effect direction
    or size, only whether the p-value survives correction for having run
    many tests.
    """
    m = len(p_values)
    if m == 0:
        return []

    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted_in_rank_order: list[float] = []
    running_max = 0.0
    for rank, idx in enumerate(order):
        factor = m - rank
        candidate = min(float(p_values[idx]) * factor, 1.0)
        running_max = max(running_max, candidate)
        adjusted_in_rank_order.append(running_max)

    result = [0.0] * m
    for rank, idx in enumerate(order):
        result[idx] = adjusted_in_rank_order[rank]
    return result


def leave_one_document_out(
    values_a: Sequence[float],
    values_b: Sequence[float],
    document_ids: Sequence[str],
) -> dict[str, float]:
    """Recompute the mean paired delta with each document's queries removed.

    For each distinct id in ``document_ids``, drops every paired observation
    belonging to that document and recomputes ``mean(a - b)`` over what
    remains. The key is the removed document id; the value is the resulting
    delta. A document whose removal would leave fewer than 2 pairs is
    skipped (there is nothing meaningful left to average).

    This answers "how much does the headline result depend on any single
    document", not "is the result significant" — a result can be highly
    significant and still be driven almost entirely by one document, which
    is exactly what a large spread between the returned deltas reveals.
    Requires ``len(values_a) == len(values_b) == len(document_ids)``.
    """
    if not (len(values_a) == len(values_b) == len(document_ids)):
        msg = (
            "values_a, values_b, and document_ids must have equal length: "
            f"{len(values_a)}, {len(values_b)}, {len(document_ids)}"
        )
        raise ValueError(msg)

    a = np.asarray(values_a, dtype=np.float64)
    b = np.asarray(values_b, dtype=np.float64)
    diff = a - b
    docs = list(document_ids)

    result: dict[str, float] = {}
    for doc_id in sorted(set(docs)):
        keep_mask = np.array([d != doc_id for d in docs], dtype=bool)
        remaining = diff[keep_mask]
        if remaining.size < 2:
            continue
        result[doc_id] = float(np.mean(remaining))
    return result


# -- Legacy API (kept for existing callers) -----------------------------------


def paired_t_test(
    values_a: Sequence[float],
    values_b: Sequence[float],
    alpha: float = 0.05,
) -> tuple[float, float, bool, float]:
    """Perform a paired t-test comparing two sets of per-query metric values.

    Parametric test: assumes the paired differences are approximately
    normally distributed. Returns ``(t_statistic, p_value, is_significant,
    effect_size_cohens_d)`` where ``is_significant`` is a single uncorrected
    comparison against ``alpha`` (see `holm_correction` /
    `compute_all_comparisons` for the multiple-comparisons-safe version) and
    ``effect_size_cohens_d`` comes from `cohens_d_paired` (so it reports
    ``+inf``/``-inf`` rather than ``0.0`` for a zero-variance non-zero-mean
    difference). Does NOT tell you the result survives correction for
    running many such tests.
    """
    a = np.asarray(values_a, dtype=np.float64)
    b = np.asarray(values_b, dtype=np.float64)

    if len(a) != len(b):
        msg = f"Array lengths must match: {len(a)} vs {len(b)}"
        raise ValueError(msg)
    if len(a) < 2:
        msg = f"Need at least 2 paired observations, got {len(a)}"
        raise ValueError(msg)

    t_stat, p_value = stats.ttest_rel(a, b)

    if math.isnan(t_stat):
        t_stat = 0.0
    if math.isnan(p_value):
        p_value = 1.0

    cohens_d = cohens_d_paired(values_a, values_b)

    # bool(...) matters: p_value is a numpy scalar, so `p_value < alpha` is a
    # numpy.bool_, which json.dump cannot encode.
    return float(t_stat), float(p_value), bool(p_value < alpha), cohens_d


def compute_significance(
    lexichunk_metrics: Sequence[RetrievalMetrics],
    baseline_metrics: Sequence[RetrievalMetrics],
    baseline_strategy: StrategyName,
    alpha: float = 0.05,
) -> list[SignificanceResult]:
    """Compare LexiChunk vs one baseline across all significance metrics.

    Legacy entry point kept for existing callers. Runs one uncorrected
    paired t-test per metric in `SIGNIFICANCE_METRICS` — for a
    multiple-comparisons-corrected, distribution-free, bootstrap-CI, and
    leave-one-document-out-aware comparison across every strategy pair and
    embedding model at once, use `compute_all_comparisons` instead.
    """
    lc_by_query = {m.query_id: m for m in lexichunk_metrics}
    bl_by_query = {m.query_id: m for m in baseline_metrics}
    common_queries = sorted(set(lc_by_query.keys()) & set(bl_by_query.keys()))

    if len(common_queries) < 2:
        logger.warning(
            "Fewer than 2 common queries for %s vs %s, skipping significance.",
            StrategyName.LEXICHUNK.value,
            baseline_strategy.value,
        )
        return []

    results: list[SignificanceResult] = []

    for metric_name in SIGNIFICANCE_METRICS:
        values_a = [getattr(lc_by_query[q], metric_name) for q in common_queries]
        values_b = [getattr(bl_by_query[q], metric_name) for q in common_queries]

        mean_a = float(np.mean(values_a))
        mean_b = float(np.mean(values_b))

        improvement_pct = ((mean_a - mean_b) / mean_b * 100) if mean_b > 0 else 0.0

        t_stat, p_value, significant, effect_size = paired_t_test(values_a, values_b, alpha=alpha)

        results.append(
            SignificanceResult(
                metric_name=metric_name,
                strategy_a=StrategyName.LEXICHUNK,
                strategy_b=baseline_strategy,
                mean_a=mean_a,
                mean_b=mean_b,
                improvement_pct=improvement_pct,
                t_statistic=t_stat,
                p_value=p_value,
                significant=significant,
                effect_size=effect_size,
                n_queries=len(common_queries),
            )
        )

    return results


def compute_all_significance(
    all_metrics: Sequence[RetrievalMetrics],
    alpha: float = 0.05,
) -> list[SignificanceResult]:
    """Compare LexiChunk vs every other strategy, per embedding model.

    Legacy entry point kept for existing callers (unchanged behavior,
    including that it does not test `StrategyName.LEXICHUNK_CONTEXTUAL` and
    does not correct for running multiple tests). See
    `compute_all_comparisons` for the corrected replacement, which also
    covers the contextual strategy.
    """
    by_strategy_model: dict[tuple[str, str], list[RetrievalMetrics]] = defaultdict(list)
    for m in all_metrics:
        key = (m.strategy.value, m.embedding_model.value)
        by_strategy_model[key].append(m)

    models = sorted({m.embedding_model.value for m in all_metrics})
    baselines = sorted(
        {
            m.strategy
            for m in all_metrics
            if m.strategy != StrategyName.LEXICHUNK
            and m.strategy != StrategyName.LEXICHUNK_CONTEXTUAL
        }
    )

    all_results: list[SignificanceResult] = []

    for model_name in models:
        lc_key = (StrategyName.LEXICHUNK.value, model_name)
        lc_metrics = by_strategy_model.get(lc_key, [])

        if not lc_metrics:
            continue

        for baseline in baselines:
            bl_key = (baseline.value, model_name)
            bl_metrics = by_strategy_model.get(bl_key, [])

            if not bl_metrics:
                continue

            results = compute_significance(lc_metrics, bl_metrics, baseline, alpha=alpha)
            all_results.extend(results)

    return all_results


# -- New, corrected family-wise comparison ------------------------------------


def compute_all_comparisons(
    all_metrics: Sequence[RetrievalMetrics],
    *,
    alpha: float = 0.05,
    seed: int = 0,
    n_resamples: int = 10000,
    query_documents: Mapping[str, str] | None = None,
) -> list[ComparisonResult]:
    """Compare LexiChunk and LexiChunk-contextual against every baseline.

    For each embedding model, both `StrategyName.LEXICHUNK` and
    `StrategyName.LEXICHUNK_CONTEXTUAL` (when present) are compared against
    every strategy that is neither of those two (the true baselines: e.g.
    RCTS, sentence-split, fixed-size). The old implementation
    (`compute_all_significance`) silently dropped
    `StrategyName.LEXICHUNK_CONTEXTUAL` from testing entirely, so any claim
    about the contextual strategy's performance was never actually tested;
    this function fixes that.

    Pairing is by `RetrievalMetrics.query_id`, restricted to the queries
    common to both strategies and sorted, so which pairs feed each test is
    deterministic. For every (metric, treatment, baseline, embedding model)
    tuple this computes: the paired means and their delta, a paired
    bootstrap 95% CI on the delta (`paired_bootstrap_ci`, seeded with
    `seed`), a paired t-test and Cohen's d (`cohens_d_paired`), a Wilcoxon
    signed-rank test and matched-pairs rank-biserial correlation, and a
    leave-one-document-out delta range (`leave_one_document_out`).

    The Holm correction (`holm_correction`) is then applied ACROSS THE WHOLE
    RETURNED LIST in one pass — every metric, every baseline, and every
    embedding model together form one family — not per baseline, which is
    what made the original 21-tests-at-alpha-0.05 setup indefensible.

    `query_documents` maps `query_id -> document_id` for the
    leave-one-document-out grouping. When it is `None` (or missing an entry
    for a given query), the document id is derived heuristically from the
    query id's prefix (see `_infer_document_id`); this is a best-effort
    grouping, not necessarily the true fixture document id. `n_documents`
    counts the distinct grouping keys actually used; `lodo_*` fall back to
    `nan`/`""` only in the degenerate case where no document's removal
    leaves at least 2 pairs behind (so `leave_one_document_out` returns
    nothing to summarize).

    Comparisons with fewer than 2 common queries are skipped (nothing to
    pair). Does NOT tell you whether the *practical* effect size is large
    enough to matter for a real retrieval system — only whether, and how
    consistently, one strategy beats another on the sampled queries.
    """
    by_strategy_model: dict[tuple[str, str], list[RetrievalMetrics]] = defaultdict(list)
    for m in all_metrics:
        by_strategy_model[(m.strategy.value, m.embedding_model.value)].append(m)

    models = sorted({m.embedding_model.value for m in all_metrics})
    all_strategy_values = {m.strategy.value for m in all_metrics}
    treatments = [
        s
        for s in (StrategyName.LEXICHUNK, StrategyName.LEXICHUNK_CONTEXTUAL)
        if s.value in all_strategy_values
    ]
    baseline_values = sorted(
        all_strategy_values
        - {StrategyName.LEXICHUNK.value, StrategyName.LEXICHUNK_CONTEXTUAL.value}
    )

    provisional: list[ComparisonResult] = []

    for model_name in models:
        for treatment in treatments:
            treatment_metrics = by_strategy_model.get((treatment.value, model_name), [])
            if not treatment_metrics:
                continue
            treatment_by_query = {m.query_id: m for m in treatment_metrics}

            for baseline_value in baseline_values:
                baseline_metrics = by_strategy_model.get((baseline_value, model_name), [])
                if not baseline_metrics:
                    continue
                baseline_by_query = {m.query_id: m for m in baseline_metrics}

                common_queries = sorted(
                    set(treatment_by_query.keys()) & set(baseline_by_query.keys())
                )
                if len(common_queries) < 2:
                    continue

                document_ids = [
                    (query_documents.get(q) if query_documents is not None else None)
                    or _infer_document_id(q)
                    for q in common_queries
                ]
                n_documents = len(set(document_ids))

                for metric_name in SIGNIFICANCE_METRICS:
                    values_a = [getattr(treatment_by_query[q], metric_name) for q in common_queries]
                    values_b = [getattr(baseline_by_query[q], metric_name) for q in common_queries]

                    mean_a = float(np.mean(values_a))
                    mean_b = float(np.mean(values_b))
                    delta = mean_a - mean_b

                    ci_low, ci_high = paired_bootstrap_ci(
                        values_a, values_b, seed=seed, n_resamples=n_resamples
                    )
                    t_stat, p_value_t, _, cohens_d = paired_t_test(values_a, values_b, alpha=alpha)
                    _, p_value_wilcoxon = wilcoxon_signed_rank(values_a, values_b)
                    rank_biserial = rank_biserial_correlation(values_a, values_b)

                    lodo = leave_one_document_out(values_a, values_b, document_ids)
                    if lodo:
                        lodo_min_delta = min(lodo.values())
                        lodo_max_delta = max(lodo.values())
                        lodo_worst_document = max(
                            lodo, key=lambda doc_id: abs(lodo[doc_id] - delta)
                        )
                    else:
                        lodo_min_delta = float("nan")
                        lodo_max_delta = float("nan")
                        lodo_worst_document = ""

                    provisional.append(
                        ComparisonResult(
                            metric_name=metric_name,
                            strategy_a=treatment,
                            strategy_b=StrategyName(baseline_value),
                            embedding_model=model_name,
                            n=len(common_queries),
                            mean_a=mean_a,
                            mean_b=mean_b,
                            delta=delta,
                            ci_low=ci_low,
                            ci_high=ci_high,
                            t_statistic=t_stat,
                            p_value_t=p_value_t,
                            p_value_wilcoxon=p_value_wilcoxon,
                            p_value_holm=float("nan"),  # filled in below
                            significant_holm=False,  # filled in below
                            cohens_d=cohens_d,
                            rank_biserial=rank_biserial,
                            n_documents=n_documents,
                            lodo_min_delta=lodo_min_delta,
                            lodo_max_delta=lodo_max_delta,
                            lodo_worst_document=lodo_worst_document,
                        )
                    )

    holm_adjusted = holm_correction([r.p_value_t for r in provisional])
    return [
        replace(r, p_value_holm=p, significant_holm=bool(p < alpha))
        for r, p in zip(provisional, holm_adjusted, strict=True)
    ]
