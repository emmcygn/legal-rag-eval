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

Ranked hits are additionally normalised before scoring (:func:`_ranked_hits`):
non-integer, boolean, non-positive and out-of-range ranks are dropped, and each
rank and each chunk id is counted at most once. Without that, a strategy that
emitted the same chunk twice inflated P@k and NDCG for free. Gold annotations are
normalised too (:func:`valid_relevant_sections`): IRRELEVANT grades and
degenerate spans are discarded, and duplicate annotations of the same span
collapse to their highest grade so recall's denominator cannot be padded.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

from scipy.optimize import linear_sum_assignment

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

#: Identifier for the NDCG variant computed here, recorded on every
#: :class:`~scaffolder.models.RetrievalMetrics` row so old exports that used the
#: unversioned per-rank formula are never silently compared against new ones.
NDCG_METRIC_VERSION = "evidence_assignment_ndcg_v1"


def valid_relevant_sections(
    relevant_sections: Sequence[RelevantSection],
) -> tuple[RelevantSection, ...]:
    """Return the uniquely identifiable, positively graded, well-formed gold sections.

    A gold annotation is usable only if it is graded above ``IRRELEVANT`` and has
    a non-empty span (``char_end > char_start``); a zero-width or inverted span
    can never be overlapped, so leaving it in would silently depress recall.
    Annotations that name the same ``(document_id, section_id, span)`` collapse to
    the single highest-graded one, so a query whose YAML lists the same clause
    twice does not get a padded recall denominator.
    """
    sections_by_identity: dict[tuple[str, str, int, int], RelevantSection] = {}

    for section in relevant_sections:
        if section.grade == RelevanceGrade.IRRELEVANT:
            continue
        if section.char_end <= section.char_start:
            continue

        identity = (
            section.document_id,
            section.section_id,
            section.char_start,
            section.char_end,
        )
        existing = sections_by_identity.get(identity)
        if existing is None or section.grade.value > existing.grade.value:
            sections_by_identity[identity] = section

    return tuple(sections_by_identity.values())


def has_invalid_positive_gold(relevant_sections: Sequence[RelevantSection]) -> bool:
    """Whether a positively graded annotation carries an unusable span.

    Used to tell "this query genuinely has no relevant sections" (recall is
    vacuously 1.0) apart from "this query's annotations are malformed" (recall is
    0.0, so a broken annotation is visible as a failure rather than a free win).
    """
    return any(
        section.grade != RelevanceGrade.IRRELEVANT and section.char_end <= section.char_start
        for section in relevant_sections
    )


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

    Only sections surviving :func:`valid_relevant_sections` are considered, so an
    IRRELEVANT-graded or zero-width annotation never credits a chunk.

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
    for section in valid_relevant_sections(relevant_sections):
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

    P@k = |distinct relevant chunks in top-k| / min(k, deepest rank returned)

    Range: [0.0, 1.0]. Two separate decisions make up that denominator, and both
    are load-bearing:

    * It is *not* a fixed ``k``. A strategy that only returns 3 chunks when asked
      for 10 is scored out of 3, not silently penalised by a denominator of 10 —
      that would conflate "retrieved nothing relevant" with "returned fewer
      chunks than requested".
    * It is *not* the count of surviving hits either. It is the deepest rank the
      strategy actually reached, so ranks it consumed but wasted still count: a
      duplicate chunk at ranks 1 and 2 is scored 1/2, not 1/1, and a gap (hits at
      ranks 1 and 3 with nothing at 2) is scored out of 3. Otherwise emitting the
      same chunk twice would be free, which is exactly the bug this denominator
      was changed to close.

    Relevance is decided by :func:`is_relevant` (span overlap only). This does
    NOT weight by relevance grade — a RELATED match counts identically to an
    EXACT match; use :func:`ndcg_at_k` when grade matters.

    Returns 0.0 when there are no valid hits at rank <= k, and 0.0 for ``k <= 0``.
    """
    if k <= 0:
        return 0.0

    top_k = _ranked_hits(hits, k)
    if not top_k:
        return 0.0

    # Deepest rank across every *validly ranked* hit, before chunk-id
    # de-duplication -- a rank a duplicate consumed is still a rank consumed.
    slots = min(k, max(h.rank for h in _valid_ranks(hits, k)))
    relevant_count = sum(
        1 for h in top_k if is_relevant(h.chunk, relevant_sections, min_overlap_chars)
    )
    return relevant_count / slots


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
    sections = valid_relevant_sections(relevant_sections)
    if not sections:
        return 0.0 if has_invalid_positive_gold(relevant_sections) else 1.0
    if k <= 0:
        return 0.0

    found: set[RelevantSection] = set()
    for h in _ranked_hits(hits, k):
        found.update(matched_sections(h.chunk, sections, min_overlap_chars))

    return len(found) / len(sections)


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
    for h in _ranked_hits(hits):
        if is_relevant(h.chunk, relevant_sections, min_overlap_chars):
            return 1.0 / h.rank

    return 0.0


def ndcg_at_k(
    hits: Sequence[RetrievalHit],
    relevant_sections: Sequence[RelevantSection],
    k: int,
    min_overlap_chars: int = DEFAULT_MIN_OVERLAP_CHARS,
) -> float:
    """Compatibility alias for :func:`evidence_assignment_ndcg_v1`.

    This is deliberately not classical per-rank chunk NDCG; see that function for
    the exact rule and for why the per-rank formula was replaced.
    """
    return evidence_assignment_ndcg_v1(hits, relevant_sections, k, min_overlap_chars)


def evidence_assignment_ndcg_v1(
    hits: Sequence[RetrievalHit],
    relevant_sections: Sequence[RelevantSection],
    k: int,
    min_overlap_chars: int = DEFAULT_MIN_OVERLAP_CHARS,
) -> float:
    """Graded ranking score using maximum-weight one-to-one gold assignment.

    Grades: EXACT=3, SAME_SECTION=2, RELATED=1, IRRELEVANT=0 (:class:`RelevanceGrade`).
    Each rank may claim at most one *distinct* gold section and each gold section
    may be claimed by at most one rank; the assignment chosen is the one
    maximising total discounted gain (Hungarian algorithm via
    :func:`scipy.optimize.linear_sum_assignment`), with a rank/section pair's
    weight being ``(2**grade - 1) / log2(rank + 1)`` when that chunk matches that
    section and 0 otherwise. IDCG is the annotated grades sorted descending and
    truncated to k, so the ratio is bounded by [0.0, 1.0] *by construction*.

    Why not the per-rank formula: when several retrieved chunks in the top-k each
    independently match the *same* annotation (two overlapping chunks straddling
    one clause), the per-rank version gave each of them that annotation's full
    gain, so DCG could exceed IDCG and had to be clamped to 1.0 — which silently
    conflated a perfect ranking with a double-counted one, and made the score
    depend on how the gold list happened to be ordered. One-to-one assignment
    removes both problems: redundant chunks earn nothing extra, and the result is
    invariant to gold-list permutation and to the Python hash seed.

    Every grade comes from the same span-overlap relevance test
    ``precision_at_k`` uses, so a chunk P@k counts as relevant is never graded
    IRRELEVANT here.

    Range: [0.0, 1.0]. Returns 0.0 when there are no usable relevant sections,
    when ``k <= 0``, or when IDCG is 0.
    """
    sections = valid_relevant_sections(relevant_sections)
    if k <= 0 or not sections:
        return 0.0

    section_indices = {section: index for index, section in enumerate(sections)}
    # Columns beyond len(sections) are zero-weight slack, so ranks that match
    # nothing (or whose section is already claimed) can be assigned for free.
    weights = [[0.0] * (len(sections) + k) for _ in range(k)]
    for hit in _ranked_hits(hits, k):
        discount = math.log2(hit.rank + 1)
        for section in matched_sections(hit.chunk, sections, min_overlap_chars):
            weights[hit.rank - 1][section_indices[section]] = (
                2**section.grade.value - 1
            ) / discount

    ideal_grades = sorted((s.grade.value for s in sections), reverse=True)[:k]
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


def _valid_ranks(
    hits: Sequence[RetrievalHit],
    k: int | None = None,
) -> list[RetrievalHit]:
    """Hits whose rank is a positive integer within ``k``, duplicates included.

    Separate from :func:`_ranked_hits` because P@k's denominator has to count a
    rank a duplicate chunk consumed, while its numerator must not score that
    chunk twice.
    """
    return [
        hit
        for hit in hits
        if isinstance(hit.rank, int)
        and not isinstance(hit.rank, bool)
        and hit.rank > 0
        and (k is None or hit.rank <= k)
    ]


def _ranked_hits(
    hits: Sequence[RetrievalHit],
    k: int | None = None,
) -> list[RetrievalHit]:
    """Return one valid occurrence of each ranked chunk, in rank order.

    Normalises a hit list before any metric reads it:

    * ranks that are not positive integers are dropped (``bool`` is rejected
      explicitly — it is an ``int`` subclass, so ``True`` would otherwise pass as
      rank 1);
    * ranks above ``k`` are dropped when ``k`` is given;
    * each rank is taken once, and each ``chunk.id`` is taken once — the earliest
      rank wins.

    The chunk-id de-duplication is the load-bearing part: without it a strategy
    that emitted the same chunk at ranks 1 and 2 scored it twice in P@k and NDCG,
    which is free score for a bug.
    """
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

    top_k = _ranked_hits(hits, k)
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
        ndcg_metric=NDCG_METRIC_VERSION,
        drm_hit=rate > 0.0,
        drm_rate=rate,
        n_relevant_sections=len(valid_relevant_sections(sections)),
    )
