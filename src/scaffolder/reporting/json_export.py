"""JSON export and reconstruction for benchmark results.

``reconstruct_benchmark_result`` deliberately does not rebuild ``BenchmarkResult.comparisons``
(the rich ``ComparisonResult`` objects): that field is typed ``list[object]`` precisely because
reporting has no business depending on ``scaffolder.metrics.statistical`` for its shape, and a
generic dict-to-dataclass reconstruction of an arbitrary ``object`` field is not something this
module can do safely. Reconstructed results always keep ``legacy_structural_metrics`` and
``significance_results`` (via ``ComparisonResult.to_significance_result()`` at export time), so
the CLI/HTML significance section already has a correct fallback: see
``scaffolder.reporting.cli.render_significance_section``.
"""

from __future__ import annotations

import contextlib
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from scaffolder.models import (
    BenchmarkResult,
    EmbeddingModelName,
    GoldStructuralMetrics,
    LegacyStructuralMetrics,
    RetrievalMetrics,
    SignificanceResult,
    StrategyName,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

logger = logging.getLogger(__name__)

# Keys present only on the legacy (LexiChunk-self-graded) structural metric shape. Their
# presence on an entry under the "structural_metrics" key means it came from a pre-audit
# JSON export (where "structural_metrics" held what is now `LegacyStructuralMetrics`), and
# it must be routed to `BenchmarkResult.legacy_structural_metrics` instead.
_LEGACY_STRUCTURAL_MARKER_KEYS = frozenset(
    {"definition_preservation_rate", "cross_ref_resolution_rate", "hierarchy_depth_retained"}
)


def _serialize(obj: Any) -> Any:
    """Custom serializer for dataclass fields."""
    if isinstance(obj, Path):
        return str(obj)
    # Import here to avoid circular imports at module level
    from datetime import datetime as dt_cls

    if isinstance(obj, dt_cls):
        return obj.isoformat()
    if isinstance(obj, set):
        return sorted(obj)
    # scipy/numpy return numpy scalars (e.g. numpy.bool_ for `p_value < alpha`),
    # which json cannot encode. Unwrap them to their Python equivalents.
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def export_json(
    result: BenchmarkResult,
    output_path: str | Path,
    indent: int = 2,
) -> Path:
    """Export benchmark results to a JSON file.

    Args:
        result: The complete benchmark result to export.
        output_path: Path to write the JSON file. Parent dirs are created.
        indent: JSON indentation level.

    Returns:
        The Path where the file was written.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = asdict(result)

    with open(output_path, "w") as f:
        json.dump(data, f, indent=indent, default=_serialize)

    return output_path


def export_json_string(result: BenchmarkResult, indent: int = 2) -> str:
    """Export benchmark results as a JSON string.

    Useful for Streamlit download buttons and API responses.
    """
    data = asdict(result)
    return json.dumps(data, indent=indent, default=_serialize)


def load_json(path: str | Path) -> dict[str, Any]:
    """Load a previously exported benchmark result from JSON.

    Returns the raw dict — caller is responsible for reconstructing
    dataclass instances if needed.
    """
    with open(path) as f:
        return json.load(f)  # type: ignore[no-any-return]


def _is_legacy_structural_shape(entry: Mapping[str, Any]) -> bool:
    """True when ``entry``'s keys match `LegacyStructuralMetrics`, not `GoldStructuralMetrics`."""
    return any(key in entry for key in _LEGACY_STRUCTURAL_MARKER_KEYS)


def _parse_gold_structural(d: Mapping[str, Any]) -> GoldStructuralMetrics:
    return GoldStructuralMetrics(
        strategy=StrategyName(d["strategy"]),
        document_id=d["document_id"],
        chunk_count=d["chunk_count"],
        avg_chunk_chars=d["avg_chunk_chars"],
        chunk_size_cv=d["chunk_size_cv"],
        located_chunks=d["located_chunks"],
        localization_rate=d["localization_rate"],
        localization_coverage=d["localization_coverage"],
        n_leaf_clauses=d["n_leaf_clauses"],
        clause_fragmentation_rate=d["clause_fragmentation_rate"],
        n_top_level_clauses=d["n_top_level_clauses"],
        top_level_over_merge_rate=d["top_level_over_merge_rate"],
        sub_clause_grouping_rate=d["sub_clause_grouping_rate"],
        heading_recall=d.get("heading_recall"),
        heading_precision=d.get("heading_precision"),
        n_gold_headings=d["n_gold_headings"],
        n_definition_uses=d["n_definition_uses"],
        definition_attachment_recall=d["definition_attachment_recall"],
        n_gold_cross_refs=d["n_gold_cross_refs"],
        xref_target_recall=d.get("xref_target_recall"),
        xref_target_precision=d.get("xref_target_precision"),
    )


def _parse_legacy_structural(d: Mapping[str, Any]) -> LegacyStructuralMetrics:
    return LegacyStructuralMetrics(
        strategy=StrategyName(d["strategy"]),
        document_id=d["document_id"],
        clause_fragmentation_rate=d["clause_fragmentation_rate"],
        definition_preservation_rate=d["definition_preservation_rate"],
        cross_ref_resolution_rate=d["cross_ref_resolution_rate"],
        hierarchy_depth_retained=d["hierarchy_depth_retained"],
        chunk_size_cv=d["chunk_size_cv"],
        chunk_count=d["chunk_count"],
        avg_chunk_chars=d["avg_chunk_chars"],
    )


def _parse_retrieval(d: Mapping[str, Any]) -> RetrievalMetrics:
    return RetrievalMetrics(
        query_id=d["query_id"],
        strategy=StrategyName(d["strategy"]),
        embedding_model=EmbeddingModelName(d["embedding_model"]),
        precision_at_1=d["precision_at_1"],
        precision_at_3=d["precision_at_3"],
        precision_at_5=d["precision_at_5"],
        precision_at_10=d["precision_at_10"],
        recall_at_1=d["recall_at_1"],
        recall_at_3=d["recall_at_3"],
        recall_at_5=d["recall_at_5"],
        recall_at_10=d["recall_at_10"],
        mrr=d["mrr"],
        ndcg_at_10=d["ndcg_at_10"],
        drm_hit=d["drm_hit"],
        drm_rate=d.get("drm_rate", 0.0),
        n_relevant_sections=d.get("n_relevant_sections", 0),
    )


def _parse_significance(d: Mapping[str, Any]) -> SignificanceResult:
    return SignificanceResult(
        metric_name=d["metric_name"],
        strategy_a=StrategyName(d["strategy_a"]),
        strategy_b=StrategyName(d["strategy_b"]),
        mean_a=d["mean_a"],
        mean_b=d["mean_b"],
        improvement_pct=d["improvement_pct"],
        t_statistic=d["t_statistic"],
        p_value=d["p_value"],
        significant=d["significant"],
        effect_size=d["effect_size"],
        n_queries=d["n_queries"],
    )


def reconstruct_benchmark_result(data: dict[str, Any]) -> BenchmarkResult:
    """Reconstruct a `BenchmarkResult` from JSON data (the inverse of `export_json`).

    Malformed entries are skipped with a logged warning (not silently dropped) so data loss
    from a corrupt or hand-edited results file is visible. Also loads **older** result JSON
    whose ``"structural_metrics"`` key holds the legacy (LexiChunk-self-graded) shape: such
    entries are detected by their keys (see `_is_legacy_structural_shape`) and routed to
    ``legacy_structural_metrics``, leaving ``structural_metrics`` empty for that entry.

    ``comparisons`` (the rich `~scaffolder.metrics.statistical.ComparisonResult` objects) is
    intentionally left as reconstructed by `BenchmarkResult`'s default (empty) — see the
    module docstring. ``significance_results`` (the legacy, always-JSON-safe shape) is
    reconstructed normally and is what the CLI/HTML significance section falls back to.
    """
    result = BenchmarkResult(timestamp=data.get("timestamp", ""))

    for s in data.get("strategies", []):
        with contextlib.suppress(ValueError):
            result.strategies.append(StrategyName(s))

    result.documents = list(data.get("documents", []))

    for m in data.get("models", []):
        with contextlib.suppress(ValueError):
            result.models.append(EmbeddingModelName(m))

    result.seed = data.get("seed")
    result.lexichunk_version = data.get("lexichunk_version")
    result.lexichunk_commit = data.get("lexichunk_commit")
    config = data.get("config")
    result.config = dict(config) if isinstance(config, dict) else {}

    for i, sm_data in enumerate(data.get("structural_metrics", [])):
        try:
            if _is_legacy_structural_shape(sm_data):
                result.legacy_structural_metrics.append(_parse_legacy_structural(sm_data))
            else:
                result.structural_metrics.append(_parse_gold_structural(sm_data))
        except (KeyError, ValueError, TypeError) as exc:
            logger.warning("Skipping malformed structural_metrics[%d]: %s", i, exc)

    for i, lsm_data in enumerate(data.get("legacy_structural_metrics", [])):
        try:
            result.legacy_structural_metrics.append(_parse_legacy_structural(lsm_data))
        except (KeyError, ValueError, TypeError) as exc:
            logger.warning("Skipping malformed legacy_structural_metrics[%d]: %s", i, exc)

    for i, rm_data in enumerate(data.get("retrieval_metrics", [])):
        try:
            result.retrieval_metrics.append(_parse_retrieval(rm_data))
        except (KeyError, ValueError, TypeError) as exc:
            logger.warning("Skipping malformed retrieval_metrics[%d]: %s", i, exc)

    for i, sr_data in enumerate(data.get("significance_results", [])):
        try:
            result.significance_results.append(_parse_significance(sr_data))
        except (KeyError, ValueError, TypeError) as exc:
            logger.warning("Skipping malformed significance_results[%d]: %s", i, exc)

    return result
