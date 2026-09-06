"""Deterministic lexical ranking for the offline evidence benchmark."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

_TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)


@dataclass(frozen=True, slots=True)
class SpanCandidate:
    id: str
    document_id: str
    start: int
    end: int
    text: str


@dataclass(frozen=True, slots=True)
class RankedCandidate:
    candidate: SpanCandidate
    score: float
    rank: int


def tokenize(text: str) -> tuple[str, ...]:
    """Return deterministic case-folded Unicode word tokens."""
    return tuple(match.group(0).casefold() for match in _TOKEN_PATTERN.finditer(text))


def rank_lexical(
    query: str,
    candidates: Sequence[SpanCandidate],
    limit: int,
) -> tuple[RankedCandidate, ...]:
    """Rank positive-overlap candidates by cosine-normalized term frequency."""
    if not isinstance(query, str):
        raise ValueError("query must be a string")
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
        raise ValueError("limit must be at least 1")
    _validate_candidates(candidates)

    query_counts = Counter(tokenize(query))
    if not query_counts:
        return ()
    query_norm = math.sqrt(sum(count * count for count in query_counts.values()))

    scored: list[tuple[float, SpanCandidate]] = []
    for candidate in candidates:
        candidate_counts = Counter(tokenize(candidate.text))
        candidate_norm = math.sqrt(sum(count * count for count in candidate_counts.values()))
        dot_product = sum(
            count * candidate_counts.get(token, 0) for token, count in query_counts.items()
        )
        if dot_product == 0 or candidate_norm == 0:
            continue
        scored.append((dot_product / (query_norm * candidate_norm), candidate))

    scored.sort(
        key=lambda item: (
            -item[0],
            item[1].document_id,
            item[1].start,
            item[1].end,
            item[1].id,
        )
    )
    return tuple(
        RankedCandidate(candidate=candidate, score=score, rank=rank)
        for rank, (score, candidate) in enumerate(scored[:limit], start=1)
    )


def _validate_candidates(candidates: Sequence[SpanCandidate]) -> None:
    seen_ids: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate.id, str) or not candidate.id.strip():
            raise ValueError("candidate id must not be empty")
        if candidate.id in seen_ids:
            raise ValueError(f"duplicate candidate id: {candidate.id}")
        seen_ids.add(candidate.id)
        if not isinstance(candidate.document_id, str) or not candidate.document_id.strip():
            raise ValueError(f"candidate {candidate.id} document id must not be empty")
        if not isinstance(candidate.text, str) or not candidate.text:
            raise ValueError(f"candidate {candidate.id} text must not be empty")
        if (
            not isinstance(candidate.start, int)
            or isinstance(candidate.start, bool)
            or candidate.start < 0
        ):
            raise ValueError(f"candidate {candidate.id} start must be nonnegative")
        if (
            not isinstance(candidate.end, int)
            or isinstance(candidate.end, bool)
            or candidate.end <= candidate.start
        ):
            raise ValueError(f"candidate {candidate.id} end must be greater than start")
        if candidate.end - candidate.start != len(candidate.text):
            raise ValueError(f"candidate {candidate.id} source span length must match text")
