"""Retrieval quality metrics for evaluating chunking strategies.

All functions are pure: they take retrieval results and return floats. No side
effects, no I/O, no model calls.

Every metric in this module is built on top of a single relevance definition —
:func:`matched_sections` — which is also imported and reused by
:mod:`scaffolder.retrieval.simulator`. A prior version of this harness defined
"is this chunk relevant" independently in two places (once here, once in the
simulator) and a third, stricter way again inside NDCG's own ``elif`` chain, so
NDCG and P@k could (and did) disagree about which chunks counted. There is now
exactly one relevance test, shared by ``RetrievalResult.relevant_retrieved`` and
every metric below.

Relevance is decided purely by **span overlap** between a chunk's location in the
sanitised document and a gold-annotated relevant clause's subtree span — never by
matching text (substring or bag-of-words) against a human-written description.
The previous, text-based judge was monotone in chunk length: a longer chunk has
more vocabulary and is more likely to contain a paraphrase's words by chance,
which meant chunk *size* alone could move the numbers regardless of chunk
*quality*. Span overlap is not immune to this either — see the docstring of
:func:`matched_sections` — which is why ``min_overlap_chars`` exists and why
chunk sizes must always be reported next to these numbers, not just headline
metric values.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

from scaffolder.gold import overlap_chars
from scaffolder.models import (
    RelevanceGrade,
    RetrievalMetrics,
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

#: Default minimum character overlap for a chunk to count as matching a relevant
#: section. Mirrors ``BenchmarkConfig.relevance_min_overlap_chars``; callers running
#: inside a configured benchmark should pass that value explicitly rather than rely
#: on this default.
DEFAULT_MIN_OVERLAP_CHARS = 100


def matched_sections(
    chunk: Chunk,
    relevant_sections: Sequence[RelevantSection],
    min_overlap_chars: int = DEFAULT_MIN_OVERLAP_CHARS,
) -> tuple[RelevantSection, ...]:
    """Return every relevant section that ``chunk`` counts as retrieving.

    This is the single relevance definition every metric in this module (and
    :mod:`scaffolder.retrieval.simulator`) is built on. A chunk matches a section
    when all of the following hold:

    * the chunk is located (``chunk.char_start is not None``) — an unlocated
      chunk can never be relevant, no matter what its text says;
    * ``chunk.document_id == section.document_id``;
    * the two ``[char_start, char_end)`` spans overlap by at least
      ``min_overlap_chars`` characters, computed with
      :func:`scaffolder.gold.overlap_chars`.

    Exception: when the section's own span is shorter than ``min_overlap_chars``
    (a short clause, e.g. a 40-character definition), that absolute threshold
    would make it unmatchable by construction, so the requirement is instead that
    the overlap covers at least 50% of the section's own character count.

    Returns the matched sections in the order they were given, not sorted by
    grade or overlap size.

    What this does NOT measure: whether the chunk's *text* is actually about the
    section's subject matter — only whether their document spans overlap enough.
    A chunk that is large enough to straddle a relevant clause's boundary while
    being mostly about something else still matches. Span overlap replaces the
    old text-similarity judge (which was monotone in chunk length) but is itself
    still biased toward larger chunks, just less severely and in a way bounded by
    ``min_overlap_chars``. This is why chunk size must be reported alongside any
    of these metrics, not read in isolation.
    """
    if chunk.char_start is None or chunk.char_end is None:
        return ()

    matches: list[RelevantSection] = []
    for section in relevant_sections:
        if chunk.document_id != section.document_id:
            continue
        overlap = overlap_chars(
            chunk.char_start, chunk.char_end, section.char_start, section.char_end
        )
        section_len = section.char_end - section.char_start
        if section_len < min_overlap_chars:
            is_match = section_len > 0 and overlap >= 0.5 * section_len
        else:
            is_match = overlap >= min_overlap_chars
        if is_match:
            matches.append(section)

    return tuple(matches)


def is_relevant(
    chunk: Chunk,
    relevant_sections: Sequence[RelevantSection],
    min_overlap_chars: int = DEFAULT_MIN_OVERLAP_CHARS,
) -> bool:
    """Whether ``chunk`` matches at least one section in ``relevant_sections``.

    Binary view of :func:`matched_sections`; see that function's docstring for
    the exact relevance rule and its limits. Returns ``False`` for every
    unlocated chunk, regardless of its text.
    """
    return bool(matched_sections(chunk, relevant_sections, min_overlap_chars))


def relevance_grade(
    chunk: Chunk,
    relevant_sections: Sequence[RelevantSection],
    min_overlap_chars: int = DEFAULT_MIN_OVERLAP_CHARS,
) -> RelevanceGrade:
    """Highest :class:`RelevanceGrade` among the sections ``chunk`` matches.

    Graded view of :func:`matched_sections`: ``RelevanceGrade.IRRELEVANT`` when
    the chunk matches nothing, otherwise the maximum ``grade`` among the matched
    sections. Uses the exact same match test as :func:`is_relevant`, so a chunk
    that is relevant under ``is_relevant`` always has a grade above
    ``IRRELEVANT`` and vice versa — this is the fix for the bug where NDCG used a
    stricter, independent relevance definition than P@k.
    """
    matches = matched_sections(chunk, relevant_sections, min_overlap_chars)
    if not matches:
        return RelevanceGrade.IRRELEVANT
    return max((section.grade for section in matches), key=lambda grade: grade.value)


def precision_at_k(
    hits: Sequence[RetrievalHit],
    relevant_sections: Sequence[RelevantSection],
    k: int,
    min_overlap_chars: int = DEFAULT_MIN_OVERLAP_CHARS,
) -> float:
    """Precision@k: fraction of the actually-retrieved top-k results that are relevant.

    P@k = |relevant chunks in top-k| / min(k, |top-k|)

    Range: [0.0, 1.0]. The denominator is the number of chunks actually present
    in the top-k, not the nominal ``k`` — a strategy that only returns 3 chunks
    when asked for 10 is scored out of 3, not silently penalised by a fixed
    denominator of 10 (that would conflate "retrieved nothing relevant" with
    "returned fewer chunks than requested"). Returns 0.0 when there are no hits
    at rank <= k, and 0.0 for ``k <= 0``.

    Relevance is decided by :func:`is_relevant` (span overlap only). This does
    NOT weight by relevance grade — a RELATED match counts identically to an
    EXACT match; use :func:`ndcg_at_k` when grade matters.
    """
    if k <= 0:
        return 0.0

    top_k = [h for h in hits if h.rank <= k]
    if not top_k:
        return 0.0

    relevant_count = sum(
        1 for h in top_k if is_relevant(h.chunk, relevant_sections, min_overlap_chars)
    )
    return relevant_count / min(k, len(top_k))


def recall_at_k(
    hits: Sequence[RetrievalHit],
    relevant_sections: Sequence[RelevantSection],
    k: int,
    min_overlap_chars: int = DEFAULT_MIN_OVERLAP_CHARS,
) -> float:
    """Recall@k: fraction of distinct annotated relevant sections found in the top-k.

    R@k = |distinct relevant sections matched by >=1 chunk in top-k| / |relevant_sections|

    Range: [0.0, 1.0], always — the numerator is a subset of the denominator's
    section set by construction (it counts distinct *sections*, never chunks),
    so R@k can never exceed 1 even when several retrieved chunks all match the
    same section. Returns 1.0 vacuously when the query has no annotated relevant
    sections, and 0.0 for ``k <= 0``.

    This does NOT credit finding a section more than once, and does NOT
    distinguish a section found by a marginal (just-over-threshold) overlap from
    one found by a chunk that fully contains it.
    """
    if not relevant_sections:
        return 1.0
    if k <= 0:
        return 0.0

    top_k = [h for h in hits if h.rank <= k]
    found: set[RelevantSection] = set()
    for h in top_k:
        found.update(matched_sections(h.chunk, relevant_sections, min_overlap_chars))

    return len(found) / len(relevant_sections)


def mrr(
    hits: Sequence[RetrievalHit],
    relevant_sections: Sequence[RelevantSection],
    min_overlap_chars: int = DEFAULT_MIN_OVERLAP_CHARS,
) -> float:
    """Mean Reciprocal Rank: 1 / rank of the first relevant hit.

    Range: [0.0, 1.0]. Returns 0.0 if no hit in ``hits`` is relevant. Considers
    only the single earliest relevant hit; it says nothing about how many
    relevant results follow it or how they are graded.
    """
    sorted_hits = sorted(hits, key=lambda h: h.rank)

    for h in sorted_hits:
        if is_relevant(h.chunk, relevant_sections, min_overlap_chars):
            return 1.0 / h.rank

    return 0.0


def ndcg_at_k(
    hits: Sequence[RetrievalHit],
    relevant_sections: Sequence[RelevantSection],
    k: int,
    min_overlap_chars: int = DEFAULT_MIN_OVERLAP_CHARS,
) -> float:
    """Normalized Discounted Cumulative Gain at k, using graded relevance.

    Grades: EXACT=3, SAME_SECTION=2, RELATED=1, IRRELEVANT=0 (:class:`RelevanceGrade`).
    NDCG@k = DCG@k / IDCG@k, where DCG@k = sum_i (2**grade_i - 1) / log2(i + 2)
    over the top-k hits in rank order (i is 0-indexed), and IDCG@k is the same
    formula applied to the relevant sections' own grades sorted descending and
    truncated to k.

    Every grade here comes from :func:`relevance_grade`, the same relevance
    definition ``precision_at_k`` uses via :func:`is_relevant` — a prior version
    of this function graded chunks with its own, stricter ``elif`` chain, so a
    chunk P@k counted as relevant could be graded IRRELEVANT by NDCG. That is
    fixed by construction now: nothing in this module defines relevance for
    itself.

    Range: nominally [0.0, 1.0], guarded at both ends. Returns 0.0 when there
    are no relevant sections, when ``k <= 0``, or when IDCG@k is 0 (e.g. every
    annotated grade is IRRELEVANT, which should not occur but is guarded rather
    than dividing by zero). The result is additionally clamped to 1.0: when
    several retrieved chunks in the top-k each independently match the *same*
    single annotation (for example two overlapping chunks that both straddle
    one clause), each contributes that annotation's full gain to DCG, so DCG can
    legitimately exceed IDCG, which assumes one hit per annotation. That excess
    is clamped away here rather than left unbounded or hidden inside the ratio,
    so document it: an NDCG of exactly 1.0 can mean either a perfect ranking or
    a clamped one, and cross-checking against P@k / R@k tells you which.
    """
    if k <= 0 or not relevant_sections:
        return 0.0

    sorted_hits = sorted(hits, key=lambda h: h.rank)[:k]
    actual_grades = [
        relevance_grade(h.chunk, relevant_sections, min_overlap_chars).value for h in sorted_hits
    ]

    # Pad with zeros if fewer than k hits
    while len(actual_grades) < k:
        actual_grades.append(0)

    # Ideal grades: all relevant sections' grades sorted descending
    ideal_grades = sorted(
        (s.grade.value for s in relevant_sections),
        reverse=True,
    )[:k]
    while len(ideal_grades) < k:
        ideal_grades.append(0)

    dcg = _dcg(actual_grades)
    idcg = _dcg(ideal_grades)

    if idcg == 0.0:
        return 0.0

    return min(1.0, dcg / idcg)


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
    """Document Retrieval Mismatch rate: fraction of top-k hits from the wrong document.

    DRM = |top-k chunks whose document_id is not one of query_document_ids| / k

    Range: [0.0, 1.0]. Returns 0.0 for ``k <= 0``, an empty ``query_document_ids``,
    or no hits at rank <= k. This measures nothing about relevance within the
    correct document — a chunk from the right document can still be irrelevant,
    and this metric would not catch it. All five fixture documents currently
    share a single FAISS index per (strategy, model), which is exactly the
    condition under which this rate is informative; ``RetrievalMetrics.drm_hit``
    (a boolean derived from this rate) is not, because with a shared index
    almost every query has at least one wrong-document hit somewhere in a top-10.
    """
    if k <= 0 or not query_document_ids:
        return 0.0

    top_k = [h for h in hits if h.rank <= k]
    if not top_k:
        return 0.0

    target_set = set(query_document_ids)
    mismatch_count = sum(1 for h in top_k if h.chunk.document_id not in target_set)
    return mismatch_count / k


def compute_retrieval_metrics(
    result: RetrievalResult,
    min_overlap_chars: int = DEFAULT_MIN_OVERLAP_CHARS,
) -> RetrievalMetrics:
    """Compute every retrieval metric for a single :class:`RetrievalResult`.

    All of precision/recall/MRR/NDCG are computed with the same
    ``min_overlap_chars`` threshold and the same underlying relevance test
    (:func:`is_relevant` / :func:`relevance_grade`); see this module's docstring
    for why that single-definition property matters.
    """
    hits = list(result.hits)
    sections = list(result.query.relevant_sections)
    rate = drm_rate(hits, result.query.document_ids, k=10)

    return RetrievalMetrics(
        query_id=result.query.id,
        strategy=result.strategy,
        embedding_model=result.embedding_model,
        precision_at_1=precision_at_k(hits, sections, k=1, min_overlap_chars=min_overlap_chars),
        precision_at_3=precision_at_k(hits, sections, k=3, min_overlap_chars=min_overlap_chars),
        precision_at_5=precision_at_k(hits, sections, k=5, min_overlap_chars=min_overlap_chars),
        precision_at_10=precision_at_k(hits, sections, k=10, min_overlap_chars=min_overlap_chars),
        recall_at_1=recall_at_k(hits, sections, k=1, min_overlap_chars=min_overlap_chars),
        recall_at_3=recall_at_k(hits, sections, k=3, min_overlap_chars=min_overlap_chars),
        recall_at_5=recall_at_k(hits, sections, k=5, min_overlap_chars=min_overlap_chars),
        recall_at_10=recall_at_k(hits, sections, k=10, min_overlap_chars=min_overlap_chars),
        mrr=mrr(hits, sections, min_overlap_chars=min_overlap_chars),
        ndcg_at_10=ndcg_at_k(hits, sections, k=10, min_overlap_chars=min_overlap_chars),
        drm_hit=rate > 0.0,
        drm_rate=rate,
        n_relevant_sections=len(sections),
    )
