"""Independent controls for retrieval evaluator edge cases.

These started life against the text-snippet relevance judge. The judge is now
span overlap against gold clause spans, so the scenarios are expressed as spans —
but every guarantee being pinned here is the same one, and each is a bug that was
actually shipped at some point: duplicate chunks scoring twice, rank gaps being
free, NDCG exceeding 1.0, NDCG depending on gold-list order or on PYTHONHASHSEED,
malformed gold silently crediting retrieval, and ``True`` passing as rank 1.
"""

from __future__ import annotations

import os
import subprocess
import sys

from scaffolder.metrics.retrieval import mrr, ndcg_at_k, precision_at_k, recall_at_k
from scaffolder.models import Chunk, RelevanceGrade, RelevantSection, RetrievalHit, StrategyName

# Small enough that the fixtures below stay readable, large enough that the
# short-section branch of `matched_sections` (overlap >= 50% of a sub-threshold
# section) is not what is under test here.
OVERLAP = 50


def _hit(
    rank: int | float | bool,
    span: tuple[int, int] | None,
    chunk_id: str | None = None,
    document_id: str = "doc1",
) -> RetrievalHit:
    start, end = span if span is not None else (None, None)
    chunk = Chunk(
        id=chunk_id or f"chunk-{rank}",
        text="x" * ((end - start) if span is not None else 0),
        document_id=document_id,
        strategy=StrategyName.LEXICHUNK,
        index=max(int(rank) - 1, 0) if isinstance(rank, (int, float)) else 0,
        char_start=start,
        char_end=end,
    )
    return RetrievalHit(chunk=chunk, score=1.0, rank=rank)


def _section(
    section_id: str,
    span: tuple[int, int],
    grade: RelevanceGrade,
    document_id: str = "doc1",
) -> RelevantSection:
    return RelevantSection(
        document_id=document_id,
        section_id=section_id,
        char_start=span[0],
        char_end=span[1],
        grade=grade,
    )


def test_missing_and_nonpositive_gold_do_not_credit_retrieval() -> None:
    """An IRRELEVANT grade and a zero-width span are both unusable annotations.

    Recall must be 0.0 rather than the vacuous 1.0 given to a query that
    genuinely has no relevant sections, so a broken annotation reads as a failure
    instead of a free win.
    """
    sections = (
        _section("clause_1", (0, 200), RelevanceGrade.IRRELEVANT),
        _section("clause_2", (300, 300), RelevanceGrade.EXACT),
    )
    hits = [_hit(1, (0, 200))]

    assert precision_at_k(hits, sections, k=1, min_overlap_chars=OVERLAP) == 0.0
    assert recall_at_k(hits, sections, k=1, min_overlap_chars=OVERLAP) == 0.0
    assert mrr(hits, sections, min_overlap_chars=OVERLAP) == 0.0
    assert ndcg_at_k(hits, sections, k=1, min_overlap_chars=OVERLAP) == 0.0


def test_duplicate_chunk_does_not_inflate_precision_or_ndcg() -> None:
    """The same chunk at two ranks consumed two slots but delivered one result."""
    sections = (_section("clause_1", (0, 200), RelevanceGrade.EXACT),)
    hits = [
        _hit(1, (0, 200), chunk_id="same-chunk"),
        _hit(2, (0, 200), chunk_id="same-chunk"),
    ]

    assert precision_at_k(hits, sections, k=2, min_overlap_chars=OVERLAP) == 0.5
    assert recall_at_k(hits, sections, k=2, min_overlap_chars=OVERLAP) == 1.0
    assert ndcg_at_k(hits, sections, k=2, min_overlap_chars=OVERLAP) == 1.0


def test_rank_gaps_keep_precision_denominator_and_ndcg_discount() -> None:
    """A hit at rank 3 means ranks 1-3 were used, whether or not rank 2 arrived."""
    sections = (_section("clause_1", (0, 200), RelevanceGrade.EXACT),)
    hits = [
        _hit(1, (400, 600)),
        _hit(3, (0, 200)),
    ]

    assert precision_at_k(hits, sections, k=3, min_overlap_chars=OVERLAP) == 1.0 / 3.0
    assert mrr(hits, sections, min_overlap_chars=OVERLAP) == 1.0 / 3.0
    assert ndcg_at_k(hits, sections, k=3, min_overlap_chars=OVERLAP) == 0.5


def test_one_chunk_covering_two_sections_recovers_both_for_recall() -> None:
    sections = (
        _section("clause_1", (0, 200), RelevanceGrade.EXACT),
        _section("clause_2", (200, 400), RelevanceGrade.EXACT),
    )
    hits = [_hit(1, (0, 400))]

    assert precision_at_k(hits, sections, k=1, min_overlap_chars=OVERLAP) == 1.0
    assert recall_at_k(hits, sections, k=1, min_overlap_chars=OVERLAP) == 1.0


def test_ndcg_uses_maximum_weight_assignment_for_overlapping_evidence() -> None:
    """Rank 1 straddles both clauses; rank 2 reaches only the second.

    The greedy reading (rank 1 claims whichever section comes first in the gold
    list) can strand rank 2 with nothing. One-to-one maximum-weight assignment
    gives rank 1 the section only rank 1 can reach, so both are credited.
    """
    sections = (
        _section("clause_alpha", (0, 200), RelevanceGrade.EXACT),
        _section("clause_beta", (150, 400), RelevanceGrade.EXACT),
    )
    hits = [
        _hit(1, (100, 220)),
        _hit(2, (250, 400)),
    ]

    assert ndcg_at_k(hits, sections, k=2, min_overlap_chars=OVERLAP) == 1.0


def test_ndcg_is_invariant_to_gold_permutation() -> None:
    sections = (
        _section("clause_alpha", (0, 200), RelevanceGrade.EXACT),
        _section("clause_beta", (150, 400), RelevanceGrade.EXACT),
    )
    hits = [
        _hit(1, (100, 220)),
        _hit(2, (250, 400)),
    ]

    assert ndcg_at_k(hits, sections, k=2, min_overlap_chars=OVERLAP) == 1.0
    assert ndcg_at_k(hits, tuple(reversed(sections)), k=2, min_overlap_chars=OVERLAP) == 1.0


def test_ndcg_is_invariant_to_python_hash_seed() -> None:
    program = """
from scaffolder.metrics.retrieval import ndcg_at_k
from scaffolder.models import Chunk, RelevanceGrade, RelevantSection, RetrievalHit, StrategyName

sections = (
    RelevantSection('doc1', 'clause_alpha', 0, 200, RelevanceGrade.EXACT),
    RelevantSection('doc1', 'clause_beta', 150, 400, RelevanceGrade.EXACT),
)
hits = (
    RetrievalHit(
        Chunk('one', 'x' * 120, 'doc1', StrategyName.LEXICHUNK, 0, char_start=100, char_end=220),
        1.0,
        1,
    ),
    RetrievalHit(
        Chunk('two', 'x' * 150, 'doc1', StrategyName.LEXICHUNK, 1, char_start=250, char_end=400),
        0.5,
        2,
    ),
)
print(ndcg_at_k(hits, sections, 2, 50))
"""
    scores = []
    for seed in ("0", "1", "2", "3"):
        environment = {**os.environ, "PYTHONHASHSEED": seed}
        completed = subprocess.run(
            [sys.executable, "-c", program],
            check=True,
            capture_output=True,
            env=environment,
            text=True,
        )
        scores.append(float(completed.stdout))

    assert scores == [1.0, 1.0, 1.0, 1.0]


def test_document_id_case_does_not_collapse_distinct_gold_sections() -> None:
    """Two documents whose ids differ only in case are two documents."""
    sections = (
        _section("clause_1", (0, 200), RelevanceGrade.EXACT, "doc1"),
        _section("clause_1", (0, 200), RelevanceGrade.EXACT, "DOC1"),
    )
    hits = [_hit(1, (0, 200), document_id="doc1")]

    assert recall_at_k(hits, sections, k=1, min_overlap_chars=OVERLAP) == 0.5


def test_non_integer_or_boolean_ranks_are_ignored() -> None:
    """``True`` is an ``int`` subclass and must not slip through as rank 1."""
    sections = (_section("clause_1", (0, 200), RelevanceGrade.EXACT),)
    hits = [
        _hit(True, (0, 200), chunk_id="bool-rank"),
        _hit(1.0, (0, 200), chunk_id="float-rank"),
    ]

    assert precision_at_k(hits, sections, k=1, min_overlap_chars=OVERLAP) == 0.0
    assert recall_at_k(hits, sections, k=1, min_overlap_chars=OVERLAP) == 0.0
    assert mrr(hits, sections, min_overlap_chars=OVERLAP) == 0.0
    assert ndcg_at_k(hits, sections, k=1, min_overlap_chars=OVERLAP) == 0.0


def test_ndcg_stays_bounded_when_multiple_hits_share_one_gold_section() -> None:
    """Two distinct chunks over one clause cannot be paid for it twice."""
    sections = (_section("clause_1", (0, 200), RelevanceGrade.EXACT),)
    hits = [
        _hit(1, (0, 200), chunk_id="first"),
        _hit(2, (0, 200), chunk_id="second"),
    ]

    score = ndcg_at_k(hits, sections, k=2, min_overlap_chars=OVERLAP)
    assert 0.0 <= score <= 1.0
    assert score == 1.0
