"""Statistical significance testing for benchmark comparisons."""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np
from scipy import stats  # type: ignore[import-untyped]

from scaffolder.models import (
    SignificanceResult,
    StrategyName,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from scaffolder.models import RetrievalMetrics

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


def paired_t_test(
    values_a: Sequence[float],
    values_b: Sequence[float],
    alpha: float = 0.05,
) -> tuple[float, float, bool, float]:
    """Perform a paired t-test comparing two sets of per-query metric values.

    Returns:
        (t_statistic, p_value, is_significant, effect_size_cohens_d)
    """
    a = np.array(values_a, dtype=np.float64)
    b = np.array(values_b, dtype=np.float64)

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

    diff = a - b
    d_mean = float(np.mean(diff))
    d_std = float(np.std(diff, ddof=1))
    cohens_d = d_mean / d_std if d_std > 0 else 0.0

    # bool(...) matters: p_value is a numpy scalar, so `p_value < alpha` is a
    # numpy.bool_, which json.dump cannot encode.
    return float(t_stat), float(p_value), bool(p_value < alpha), cohens_d


def compute_significance(
    lexichunk_metrics: Sequence[RetrievalMetrics],
    baseline_metrics: Sequence[RetrievalMetrics],
    baseline_strategy: StrategyName,
    alpha: float = 0.05,
) -> list[SignificanceResult]:
    """Compare LexiChunk vs one baseline across all significance metrics."""
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
    """Compare LexiChunk vs every other strategy, per embedding model."""
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
