"""Independent controls for retrieval evaluator edge cases."""

from __future__ import annotations

import os
import subprocess
import sys

from scaffolder.metrics.retrieval import mrr, ndcg_at_k, precision_at_k, recall_at_k
from scaffolder.models import Chunk, RelevanceGrade, RelevantSection, RetrievalHit, StrategyName


def _hit(
    rank: int | float | bool,
    text: str,
    chunk_id: str | None = None,
    document_id: str = "doc1",
) -> RetrievalHit:
    chunk = Chunk(
        id=chunk_id or f"chunk-{rank}",
        text=text,
        document_id=document_id,
        strategy=StrategyName.LEXICHUNK,
        index=max(rank - 1, 0),
    )
    return RetrievalHit(chunk=chunk, score=1.0, rank=rank)


def _section(
    section_id: str,
    text: str,
    grade: RelevanceGrade,
    document_id: str = "doc1",
) -> RelevantSection:
    return RelevantSection(
        document_id=document_id,
        section_id=section_id,
        text_snippet=text,
        grade=grade,
    )


def test_missing_and_nonpositive_gold_do_not_credit_retrieval() -> None:
    sections = (
        _section("clause_1", "answer passage", RelevanceGrade.IRRELEVANT),
        _section("", "", RelevanceGrade.EXACT),
    )
    hits = [_hit(1, "This contains the answer passage.")]

    assert precision_at_k(hits, sections, k=1) == 0.0
    assert recall_at_k(hits, sections, k=1) == 0.0
    assert mrr(hits, sections) == 0.0
    assert ndcg_at_k(hits, sections, k=1) == 0.0


def test_duplicate_chunk_does_not_inflate_precision_or_ndcg() -> None:
    sections = (_section("clause_1", "answer passage", RelevanceGrade.EXACT),)
    hits = [
        _hit(1, "This contains the answer passage.", chunk_id="same-chunk"),
        _hit(2, "This contains the answer passage.", chunk_id="same-chunk"),
    ]

    assert precision_at_k(hits, sections, k=2) == 0.5
    assert recall_at_k(hits, sections, k=2) == 1.0
    assert ndcg_at_k(hits, sections, k=2) == 1.0


def test_rank_gaps_keep_precision_denominator_and_ndcg_discount() -> None:
    sections = (_section("clause_1", "answer passage", RelevanceGrade.EXACT),)
    hits = [
        _hit(1, "unrelated"),
        _hit(3, "This contains the answer passage."),
    ]

    assert precision_at_k(hits, sections, k=3) == 1.0 / 3.0
    assert mrr(hits, sections) == 1.0 / 3.0
    assert ndcg_at_k(hits, sections, k=3) == 0.5


def test_one_chunk_covering_two_sections_recovers_both_for_recall() -> None:
    sections = (
        _section("clause_1", "first answer", RelevanceGrade.EXACT),
        _section("clause_2", "second answer", RelevanceGrade.EXACT),
    )
    hits = [_hit(1, "The first answer and second answer are both in this chunk.")]

    assert precision_at_k(hits, sections, k=1) == 1.0
    assert recall_at_k(hits, sections, k=1) == 1.0


def test_ndcg_uses_maximum_weight_assignment_for_overlapping_evidence() -> None:
    sections = (
        _section("clause_alpha", "shared passage", RelevanceGrade.EXACT),
        _section("clause_beta", "shared passage", RelevanceGrade.EXACT),
    )
    hits = [
        _hit(1, "shared passage"),
        _hit(2, "Clause beta: the remaining answer"),
    ]

    assert ndcg_at_k(hits, sections, k=2) == 1.0


def test_ndcg_is_invariant_to_gold_permutation() -> None:
    sections = (
        _section("clause_alpha", "shared passage", RelevanceGrade.EXACT),
        _section("clause_beta", "shared passage", RelevanceGrade.EXACT),
    )
    hits = [
        _hit(1, "shared passage"),
        _hit(2, "Clause beta: the remaining answer"),
    ]

    assert ndcg_at_k(hits, sections, k=2) == 1.0
    assert ndcg_at_k(hits, tuple(reversed(sections)), k=2) == 1.0


def test_ndcg_is_invariant_to_python_hash_seed() -> None:
    program = """
from scaffolder.metrics.retrieval import ndcg_at_k
from scaffolder.models import Chunk, RelevanceGrade, RelevantSection, RetrievalHit, StrategyName

sections = (
    RelevantSection('doc1', 'clause_alpha', 'shared passage', RelevanceGrade.EXACT),
    RelevantSection('doc1', 'clause_beta', 'shared passage', RelevanceGrade.EXACT),
)
hits = (
    RetrievalHit(Chunk('one', 'shared passage', 'doc1', StrategyName.LEXICHUNK, 0), 1.0, 1),
    RetrievalHit(
        Chunk('two', 'Clause beta: the remaining answer', 'doc1', StrategyName.LEXICHUNK, 1),
        0.5,
        2,
    ),
)
print(ndcg_at_k(hits, sections, 2))
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
    sections = (
        _section("clause_1", "lowercase document answer", RelevanceGrade.EXACT, "doc1"),
        _section("clause_1", "uppercase document answer", RelevanceGrade.EXACT, "DOC1"),
    )
    hits = [_hit(1, "lowercase document answer", document_id="doc1")]

    assert recall_at_k(hits, sections, k=1) == 0.5


def test_non_integer_or_boolean_ranks_are_ignored() -> None:
    sections = (_section("clause_1", "answer passage", RelevanceGrade.EXACT),)
    hits = [
        _hit(True, "answer passage", chunk_id="bool-rank"),
        _hit(1.0, "answer passage", chunk_id="float-rank"),
    ]

    assert precision_at_k(hits, sections, k=1) == 0.0
    assert recall_at_k(hits, sections, k=1) == 0.0
    assert mrr(hits, sections) == 0.0
    assert ndcg_at_k(hits, sections, k=1) == 0.0


def test_ndcg_stays_bounded_when_multiple_hits_share_one_gold_section() -> None:
    sections = (_section("clause_1", "answer passage", RelevanceGrade.EXACT),)
    hits = [
        _hit(1, "answer passage", chunk_id="first"),
        _hit(2, "answer passage", chunk_id="second"),
    ]

    score = ndcg_at_k(hits, sections, k=2)
    assert 0.0 <= score <= 1.0
