"""Retrieval quality metrics for evaluating chunking strategies.

All functions are pure: they take retrieval results and return floats.
No side effects, no I/O, no model calls.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

from scipy.optimize import linear_sum_assignment

from scaffolder.models import (
    RelevanceGrade,
    RetrievalMetrics,
)
from scaffolder.relevance import (
    has_invalid_positive_gold,
    matching_sections,
    valid_relevant_sections,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from scaffolder.models import (
        Chunk,
        RelevantSection,
        RetrievalHit,
        RetrievalResult,
    )

logger = logging.getLogger(__name__)


def precision_at_k(
    hits: Sequence[RetrievalHit],
    relevant_sections: Sequence[RelevantSection],
    k: int,
) -> float:
    """Precision@k: fraction of top-k results that are relevant.

    P@k = |relevant in top-k| / k
    """
    if k <= 0:
        return 0.0

    sections = valid_relevant_sections(relevant_sections)
    top_k = _ranked_hits(hits, k)
    if not top_k:
        return 0.0

    relevant_count = sum(1 for h in top_k if _is_hit(h.chunk, sections))
    return relevant_count / k


def recall_at_k(
    hits: Sequence[RetrievalHit],
    relevant_sections: Sequence[RelevantSection],
    k: int,
) -> float:
    """Recall@k: fraction of all relevant items found in top-k.

    Returns 1.0 if there are no relevant sections (vacuously true).
    """
    sections = valid_relevant_sections(relevant_sections)
    if not sections:
        return 0.0 if has_invalid_positive_gold(relevant_sections) else 1.0
    if k <= 0:
        return 0.0

    matched_sections = {
        section
        for hit in _ranked_hits(hits, k)
        for section in matching_sections(hit.chunk, sections)
    }
    return len(matched_sections) / len(sections)


def mrr(
    hits: Sequence[RetrievalHit],
    relevant_sections: Sequence[RelevantSection],
) -> float:
    """Mean Reciprocal Rank: 1 / rank of first relevant result.

    Returns 0.0 if no relevant result is found.
    """
    sections = valid_relevant_sections(relevant_sections)

    for h in _ranked_hits(hits):
        if _is_hit(h.chunk, sections):
            return 1.0 / h.rank

    return 0.0


def ndcg_at_k(
    hits: Sequence[RetrievalHit],
    relevant_sections: Sequence[RelevantSection],
    k: int,
) -> float:
    """Compatibility alias for evidence-assignment NDCG v1.

    This is not classical chunk NDCG: each rank can claim one unseen gold
    section, selected by maximum discounted gain across all ranked evidence.
    """
    return evidence_assignment_ndcg_v1(hits, relevant_sections, k)


def evidence_assignment_ndcg_v1(
    hits: Sequence[RetrievalHit],
    relevant_sections: Sequence[RelevantSection],
    k: int,
) -> float:
    """Score ranked evidence using maximum-weight one-to-one gold assignment.

    Each rank can receive credit for at most one unseen gold section. The
    assignment maximizes discounted gain, while the ideal denominator remains
    the descending list of all valid gold grades.
    """
    sections = valid_relevant_sections(relevant_sections)
    if k <= 0 or not sections:
        return 0.0

    section_indices = {section: index for index, section in enumerate(sections)}
    weights = [[0.0] * (len(sections) + k) for _ in range(k)]
    for hit in _ranked_hits(hits, k):
        discount = math.log2(hit.rank + 1)
        for section in matching_sections(hit.chunk, sections):
            weights[hit.rank - 1][section_indices[section]] = (
                2**section.grade.value - 1
            ) / discount

    # Ideal grades: all relevant sections' grades sorted descending
    ideal_grades = sorted(
        [s.grade.value for s in sections],
        reverse=True,
    )[:k]
    ideal_grades.extend([0] * (k - len(ideal_grades)))

    costs = [[-weight for weight in row] for row in weights]
    row_indices, column_indices = linear_sum_assignment(costs)
    dcg = sum(weights[row][column] for row, column in zip(row_indices, column_indices, strict=True))
    idcg = _dcg(ideal_grades)

    if idcg == 0.0:
        return 0.0

    return float(dcg / idcg)


def _dcg(grades: list[int]) -> float:
    """Discounted Cumulative Gain."""
    return float(
        sum(
            (2**g - 1) / math.log2(i + 2)  # i+2: 0-indexed i, rank 1-indexed
            for i, g in enumerate(grades)
        )
    )


def drm_rate(
    hits: Sequence[RetrievalHit],
    query_document_ids: Sequence[str],
    k: int = 10,
) -> float:
    """Document Retrieval Mismatch rate.

    Measures how often retrieval pulls chunks from wrong documents.
    DRM = |top-k chunks from wrong documents| / k
    """
    if k <= 0 or not query_document_ids:
        return 0.0

    top_k = _ranked_hits(hits, k)
    if not top_k:
        return 0.0

    target_set = set(query_document_ids)
    mismatch_count = sum(1 for h in top_k if h.chunk.document_id not in target_set)
    return mismatch_count / k


# -- Internal helpers -------------------------------------------------------


def _is_hit(
    chunk: Chunk,
    relevant_sections: Sequence[RelevantSection],
) -> bool:
    """Check if a chunk matches any relevant section (binary relevance)."""
    return bool(matching_sections(chunk, relevant_sections))


def _get_grade(
    chunk: Chunk,
    relevant_sections: Sequence[RelevantSection],
) -> RelevanceGrade:
    """Get highest relevance grade for a chunk."""
    matched_sections = matching_sections(chunk, relevant_sections)
    return max(
        (section.grade for section in matched_sections),
        default=RelevanceGrade.IRRELEVANT,
        key=lambda grade: grade.value,
    )


def _ranked_hits(
    hits: Sequence[RetrievalHit],
    k: int | None = None,
) -> list[RetrievalHit]:
    """Return one valid occurrence of each ranked chunk in rank order."""
    ranked_hits: list[RetrievalHit] = []
    seen_chunk_ids: set[str] = set()
    seen_ranks: set[int] = set()

    for hit in sorted(hits, key=lambda candidate: candidate.rank):
        if (
            not isinstance(hit.rank, int)
            or isinstance(hit.rank, bool)
            or hit.rank <= 0
            or (k is not None and hit.rank > k)
        ):
            continue
        if hit.rank in seen_ranks or hit.chunk.id in seen_chunk_ids:
            continue
        ranked_hits.append(hit)
        seen_ranks.add(hit.rank)
        seen_chunk_ids.add(hit.chunk.id)

    return ranked_hits


# -- Convenience function ---------------------------------------------------


def compute_retrieval_metrics(result: RetrievalResult) -> RetrievalMetrics:
    """Compute all retrieval metrics for a single RetrievalResult."""
    hits = list(result.hits)
    sections = list(result.query.relevant_sections)

    return RetrievalMetrics(
        query_id=result.query.id,
        strategy=result.strategy,
        embedding_model=result.embedding_model,
        precision_at_1=precision_at_k(hits, sections, k=1),
        precision_at_3=precision_at_k(hits, sections, k=3),
        precision_at_5=precision_at_k(hits, sections, k=5),
        precision_at_10=precision_at_k(hits, sections, k=10),
        recall_at_1=recall_at_k(hits, sections, k=1),
        recall_at_3=recall_at_k(hits, sections, k=3),
        recall_at_5=recall_at_k(hits, sections, k=5),
        recall_at_10=recall_at_k(hits, sections, k=10),
        mrr=mrr(hits, sections),
        ndcg_at_10=ndcg_at_k(hits, sections, k=10),
        ndcg_metric="evidence_assignment_ndcg_v1",
        drm_hit=drm_rate(hits, result.query.document_ids, k=10) > 0.0,
    )
