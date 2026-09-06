"""Tests for retrieval quality metrics.

These build synthetic chunks and relevant sections directly (no dependency on
the real ``gold/*.json`` or ``queries/*.yaml``, both of which are being
rewritten concurrently) so relevance is entirely determined by the
``char_start``/``char_end`` spans given here.
"""

from __future__ import annotations

from scaffolder.metrics.retrieval import (
    DEFAULT_MIN_OVERLAP_CHARS,
    compute_retrieval_metrics,
    drm_rate,
    is_relevant,
    matched_sections,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    relevance_grade,
)
from scaffolder.models import (
    AnnotatedQuery,
    Chunk,
    EmbeddingModelName,
    Jurisdiction,
    RelevanceGrade,
    RelevantSection,
    RetrievalHit,
    RetrievalResult,
    StrategyName,
)


def _chunk(
    char_start: int | None,
    char_end: int | None,
    document_id: str = "doc1",
    text: str = "chunk text",
    index: int = 0,
) -> Chunk:
    return Chunk(
        id=f"chunk_{index}",
        text=text,
        document_id=document_id,
        strategy=StrategyName.LEXICHUNK,
        index=index,
        char_start=char_start,
        char_end=char_end,
    )


def _hit(rank: int, chunk: Chunk, score: float | None = None) -> RetrievalHit:
    return RetrievalHit(chunk=chunk, score=score if score is not None else 1.0 / rank, rank=rank)


def _section(
    char_start: int,
    char_end: int,
    grade: RelevanceGrade = RelevanceGrade.EXACT,
    document_id: str = "doc1",
    section_id: str = "1",
) -> RelevantSection:
    return RelevantSection(
        document_id=document_id,
        section_id=section_id,
        char_start=char_start,
        char_end=char_end,
        grade=grade,
        description="",
    )


# A single 300-char relevant clause.
CLAUSE = _section(1000, 1300, grade=RelevanceGrade.EXACT)


class TestMatchedSectionsLengthBias:
    """The core regression: relevance must be span overlap, not chunk length."""

    def test_small_overlapping_chunk_is_relevant(self) -> None:
        small = _chunk(1000, 1200)  # 200 chars, fully inside the clause
        assert is_relevant(small, [CLAUSE])
        assert matched_sections(small, [CLAUSE]) == (CLAUSE,)

    def test_large_non_overlapping_chunk_is_irrelevant(self) -> None:
        # 3000 chars, but located far from the clause: zero overlap.
        big = _chunk(5000, 8000)
        assert not is_relevant(big, [CLAUSE])
        assert matched_sections(big, [CLAUSE]) == ()

    def test_large_chunk_wins_by_size_alone_is_rejected(self) -> None:
        """A chunk merely being large must not make it relevant on its own.

        Under the old text-similarity judge, a long chunk had more vocabulary
        and was more likely to match a paraphrase by chance. Span overlap must
        not reproduce that: only a chunk whose span actually meets the overlap
        threshold counts, regardless of how much larger it is than the clause.
        """
        small = _chunk(1000, 1200, index=0)  # overlaps: relevant
        big = _chunk(2000, 5000, index=1)  # 3000 chars, no overlap: irrelevant
        assert is_relevant(small, [CLAUSE])
        assert not is_relevant(big, [CLAUSE])
        assert relevance_grade(big, [CLAUSE]) == RelevanceGrade.IRRELEVANT


class TestMatchedSectionsShortClause:
    def test_short_clause_matchable_via_fifty_percent_rule(self) -> None:
        short_clause = _section(100, 140, grade=RelevanceGrade.EXACT)  # 40 chars
        chunk = _chunk(90, 130)  # overlap = [100,130) = 30 chars = 75% of 40
        assert is_relevant(chunk, [short_clause], min_overlap_chars=100)

    def test_short_clause_below_fifty_percent_is_not_matched(self) -> None:
        short_clause = _section(100, 140, grade=RelevanceGrade.EXACT)  # 40 chars
        chunk = _chunk(100, 118)  # overlap = 18 chars = 45% of 40
        assert not is_relevant(chunk, [short_clause], min_overlap_chars=100)

    def test_long_clause_uses_absolute_threshold(self) -> None:
        # 300-char clause: 50% would be 150 chars, but the absolute floor of
        # min_overlap_chars applies once the clause is >= min_overlap_chars long.
        chunk_below = _chunk(1000, 1090)  # 90 chars overlap < 100
        chunk_at = _chunk(1000, 1100)  # exactly 100 chars overlap
        assert not is_relevant(chunk_below, [CLAUSE])
        assert is_relevant(chunk_at, [CLAUSE])


class TestUnlocatedChunk:
    def test_unlocated_chunk_never_relevant(self) -> None:
        chunk = _chunk(None, None)
        assert not is_relevant(chunk, [CLAUSE])
        assert relevance_grade(chunk, [CLAUSE]) == RelevanceGrade.IRRELEVANT
        assert matched_sections(chunk, [CLAUSE]) == ()

    def test_partially_unlocated_chunk_never_relevant(self) -> None:
        chunk = Chunk(
            id="c",
            text="x",
            document_id="doc1",
            strategy=StrategyName.LEXICHUNK,
            index=0,
            char_start=1000,
            char_end=None,
        )
        assert not is_relevant(chunk, [CLAUSE])


class TestWrongDocument:
    def test_matching_span_wrong_document_is_irrelevant(self) -> None:
        chunk = _chunk(1000, 1200, document_id="doc2")
        assert not is_relevant(chunk, [CLAUSE])


class TestRelevanceGradeAgreesWithIsRelevant:
    """The B2 regression: NDCG and precision must share one relevance test.

    Previously ndcg_at_k graded relevance with its own, stricter ``elif`` chain
    that could disagree with the boolean test precision/recall/MRR used. Now
    both are derived from the same ``matched_sections``, so a chunk is relevant
    under one iff it is relevant (non-IRRELEVANT) under the other, for any
    combination of sections.
    """

    def test_agreement_across_multiple_overlapping_sections(self) -> None:
        sections = [
            _section(0, 500, grade=RelevanceGrade.RELATED, section_id="a"),
            _section(400, 900, grade=RelevanceGrade.SAME_SECTION, section_id="b"),
            _section(2000, 2050, grade=RelevanceGrade.EXACT, section_id="c"),  # 50 chars, short
        ]
        chunks = [
            _chunk(0, 100, index=0),  # matches "a" only
            _chunk(450, 550, index=1),  # matches "a" and "b"
            _chunk(2010, 2040, index=2),  # 30/50 = 60% of short clause "c": matches
            _chunk(10_000, 10_100, index=3),  # matches nothing
        ]
        for chunk in chunks:
            relevant = is_relevant(chunk, sections)
            grade = relevance_grade(chunk, sections)
            assert relevant == (grade != RelevanceGrade.IRRELEVANT), (
                f"chunk {chunk.id}: is_relevant={relevant} but relevance_grade={grade!r}"
            )

        # And the specific grades are the max among matches, not just "matched or not".
        assert relevance_grade(chunks[1], sections) == RelevanceGrade.SAME_SECTION


class TestPrecisionAtK:
    def test_full_top_k_precision(self) -> None:
        hits = [
            _hit(1, _chunk(1000, 1200, index=0)),  # relevant
            _hit(2, _chunk(9000, 9200, index=1)),  # irrelevant
            _hit(3, _chunk(1050, 1250, index=2)),  # relevant
            _hit(4, _chunk(9000, 9200, index=3)),  # irrelevant
            _hit(5, _chunk(9000, 9200, index=4)),  # irrelevant
        ]
        assert precision_at_k(hits, [CLAUSE], k=5) == 0.4

    def test_fewer_than_k_hits_divides_by_hit_count(self) -> None:
        """P@k with fewer than k results must divide by the actual hit count."""
        hits = [
            _hit(1, _chunk(1000, 1200, index=0)),  # relevant
            _hit(2, _chunk(9000, 9200, index=1)),  # irrelevant
        ]
        # 1 relevant of 2 actual hits = 0.5, NOT 1/5 = 0.2.
        assert precision_at_k(hits, [CLAUSE], k=5) == 0.5

    def test_k_le_zero_returns_zero(self) -> None:
        hits = [_hit(1, _chunk(1000, 1200))]
        assert precision_at_k(hits, [CLAUSE], k=0) == 0.0

    def test_no_hits_returns_zero(self) -> None:
        assert precision_at_k([], [CLAUSE], k=5) == 0.0

    def test_no_hits_at_or_below_k_returns_zero(self) -> None:
        hits = [_hit(6, _chunk(1000, 1200))]
        assert precision_at_k(hits, [CLAUSE], k=5) == 0.0


class TestRecallAtK:
    def test_recall_never_exceeds_one_with_duplicate_matches(self) -> None:
        """Several chunks matching the same annotation must not inflate recall past 1."""
        hits = [
            _hit(1, _chunk(1000, 1200, index=0)),
            _hit(2, _chunk(1050, 1250, index=1)),
            _hit(3, _chunk(1100, 1300, index=2)),
        ]
        r = recall_at_k(hits, [CLAUSE], k=10)
        assert r == 1.0
        assert r <= 1.0

    def test_recall_counts_distinct_sections_only(self) -> None:
        sections = [
            _section(0, 200, section_id="a"),
            _section(1000, 1300, section_id="b"),
        ]
        hits = [
            _hit(1, _chunk(0, 200, index=0)),  # matches "a"
            _hit(2, _chunk(1000, 1300, index=1)),  # matches "b"
        ]
        assert recall_at_k(hits, sections, k=10) == 1.0

        # If only "a" is retrieved, recall is 1/2.
        assert recall_at_k(hits[:1], sections, k=10) == 0.5

    def test_no_relevant_sections_is_vacuously_one(self) -> None:
        hits = [_hit(1, _chunk(1000, 1200))]
        assert recall_at_k(hits, [], k=10) == 1.0

    def test_k_le_zero_returns_zero(self) -> None:
        hits = [_hit(1, _chunk(1000, 1200))]
        assert recall_at_k(hits, [CLAUSE], k=0) == 0.0

    def test_bulk_random_never_exceeds_one(self) -> None:
        """Pin R@k <= 1 as an invariant over many overlapping retrieved chunks."""
        sections = [_section(i * 1000, i * 1000 + 300, section_id=str(i)) for i in range(5)]
        hits = [
            _hit(rank, _chunk((rank % 5) * 1000, (rank % 5) * 1000 + 300, index=rank))
            for rank in range(1, 21)
        ]
        for k in (1, 3, 5, 10, 20):
            assert 0.0 <= recall_at_k(hits, sections, k=k) <= 1.0


class TestMRR:
    def test_first_hit_relevant(self) -> None:
        hits = [_hit(1, _chunk(1000, 1200))]
        assert mrr(hits, [CLAUSE]) == 1.0

    def test_third_hit_relevant(self) -> None:
        hits = [
            _hit(1, _chunk(9000, 9200, index=0)),
            _hit(2, _chunk(9000, 9200, index=1)),
            _hit(3, _chunk(1000, 1200, index=2)),
        ]
        assert mrr(hits, [CLAUSE]) == 1.0 / 3.0

    def test_no_relevant_hits(self) -> None:
        hits = [_hit(1, _chunk(9000, 9200))]
        assert mrr(hits, [CLAUSE]) == 0.0


class TestNDCG:
    def test_no_relevant_sections_returns_zero(self) -> None:
        hits = [_hit(1, _chunk(1000, 1200))]
        assert ndcg_at_k(hits, [], k=10) == 0.0

    def test_k_le_zero_returns_zero(self) -> None:
        hits = [_hit(1, _chunk(1000, 1200))]
        assert ndcg_at_k(hits, [CLAUSE], k=0) == 0.0

    def test_perfect_ranking_is_one(self) -> None:
        sections = [
            _section(0, 200, grade=RelevanceGrade.EXACT, section_id="a"),
            _section(1000, 1300, grade=RelevanceGrade.SAME_SECTION, section_id="b"),
            _section(2000, 2300, grade=RelevanceGrade.RELATED, section_id="c"),
        ]
        hits = [
            _hit(1, _chunk(0, 200, index=0)),
            _hit(2, _chunk(1000, 1300, index=1)),
            _hit(3, _chunk(2000, 2300, index=2)),
        ]
        assert ndcg_at_k(hits, sections, k=3) > 0.999

    def test_imperfect_ranking_below_one(self) -> None:
        sections = [
            _section(0, 200, grade=RelevanceGrade.EXACT, section_id="a"),
            _section(1000, 1300, grade=RelevanceGrade.RELATED, section_id="b"),
        ]
        hits = [
            _hit(1, _chunk(1000, 1300, index=0)),  # RELATED first
            _hit(2, _chunk(0, 200, index=1)),  # EXACT second
        ]
        ndcg = ndcg_at_k(hits, sections, k=2)
        assert 0.0 < ndcg < 1.0

    def test_no_relevant_hits_is_zero(self) -> None:
        hits = [_hit(1, _chunk(9000, 9200))]
        assert ndcg_at_k(hits, [CLAUSE], k=10) == 0.0

    def test_clamped_to_one_when_duplicate_matches_exceed_idcg(self) -> None:
        """Several chunks matching one annotation can push DCG above IDCG; must clamp."""
        hits = [
            _hit(1, _chunk(1000, 1200, index=0)),
            _hit(2, _chunk(1050, 1250, index=1)),
            _hit(3, _chunk(1100, 1300, index=2)),
        ]
        ndcg = ndcg_at_k(hits, [CLAUSE], k=3)
        assert ndcg <= 1.0

    def test_agrees_with_precision_on_relevant_chunks(self) -> None:
        """The B2 regression, phrased at the metric level: whenever P@k counts a
        chunk as relevant, NDCG's grading must not treat it as IRRELEVANT.
        """
        sections = [
            _section(0, 500, grade=RelevanceGrade.RELATED, section_id="a"),
            _section(400, 900, grade=RelevanceGrade.SAME_SECTION, section_id="b"),
        ]
        hits = [
            _hit(1, _chunk(450, 550, index=0)),  # overlaps both
            _hit(2, _chunk(10_000, 10_100, index=1)),  # irrelevant
        ]
        p_at_2 = precision_at_k(hits, sections, k=2)
        assert p_at_2 == 0.5  # one of two hits relevant
        assert relevance_grade(hits[0].chunk, sections) != RelevanceGrade.IRRELEVANT
        assert relevance_grade(hits[1].chunk, sections) == RelevanceGrade.IRRELEVANT


class TestDRMRate:
    def test_all_from_target_doc(self) -> None:
        hits = [_hit(i, _chunk(1000, 1200, index=i)) for i in range(1, 4)]
        assert drm_rate(hits, ["doc1"], k=10) == 0.0

    def test_some_from_wrong_doc(self) -> None:
        hits = [
            _hit(1, _chunk(1000, 1200, document_id="doc1", index=0)),
            _hit(2, _chunk(1000, 1200, document_id="doc2", index=1)),
            _hit(3, _chunk(1000, 1200, document_id="doc2", index=2)),
        ]
        assert abs(drm_rate(hits, ["doc1"], k=3) - 2.0 / 3.0) < 1e-9

    def test_empty_document_ids_returns_zero(self) -> None:
        hits = [_hit(1, _chunk(1000, 1200))]
        assert drm_rate(hits, [], k=10) == 0.0


class TestComputeRetrievalMetrics:
    def test_fills_all_fields_and_ranges(self) -> None:
        sections = (CLAUSE,)
        query = AnnotatedQuery(
            id="q1",
            text="test query",
            document_ids=["doc1"],
            jurisdiction=Jurisdiction.UK,
            relevant_sections=sections,
            category="clause_lookup",
        )
        hits = [
            _hit(1, _chunk(1000, 1200, index=0)),  # relevant
            _hit(2, _chunk(9000, 9200, index=1)),  # irrelevant
        ]
        result = RetrievalResult(
            query=query,
            strategy=StrategyName.LEXICHUNK,
            embedding_model=EmbeddingModelName.MINILM,
            hits=tuple(hits),
            relevant_retrieved=1,
            total_relevant=1,
        )
        metrics = compute_retrieval_metrics(result)

        assert metrics.query_id == "q1"
        assert metrics.strategy == StrategyName.LEXICHUNK
        assert metrics.n_relevant_sections == 1
        assert metrics.precision_at_1 == 1.0
        assert metrics.recall_at_10 == 1.0
        assert metrics.mrr == 1.0
        assert 0.0 <= metrics.ndcg_at_10 <= 1.0
        assert metrics.drm_rate == 0.0
        assert metrics.drm_hit is False

    def test_min_overlap_chars_is_threaded_through(self) -> None:
        # A chunk overlapping the clause by only 50 chars is relevant at a lax
        # threshold but not at the strict default.
        query = AnnotatedQuery(
            id="q2",
            text="test",
            document_ids=["doc1"],
            jurisdiction=Jurisdiction.UK,
            relevant_sections=(CLAUSE,),
            category="clause_lookup",
        )
        hits = (_hit(1, _chunk(1250, 1400, index=0)),)  # overlap = 50 chars
        result = RetrievalResult(
            query=query,
            strategy=StrategyName.LEXICHUNK,
            embedding_model=EmbeddingModelName.MINILM,
            hits=hits,
            relevant_retrieved=0,
            total_relevant=1,
        )
        strict = compute_retrieval_metrics(result, min_overlap_chars=100)
        lax = compute_retrieval_metrics(result, min_overlap_chars=50)
        assert strict.precision_at_1 == 0.0
        assert lax.precision_at_1 == 1.0


def test_default_min_overlap_chars_matches_config_default() -> None:
    assert DEFAULT_MIN_OVERLAP_CHARS == 100
