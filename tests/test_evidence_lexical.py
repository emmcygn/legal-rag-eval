from __future__ import annotations

import math

import pytest

from legal_rag_eval.evidence.lexical import SpanCandidate, rank_lexical, tokenize


def _candidate(
    candidate_id: str,
    text: str,
    *,
    document_id: str = "doc",
    start: int = 0,
) -> SpanCandidate:
    return SpanCandidate(candidate_id, document_id, start, start + len(text), text)


def test_tokenize_casefolds_unicode_words() -> None:
    assert tokenize("CAFÉ fees—Due_2") == ("café", "fees", "due_2")


def test_rank_lexical_uses_cosine_term_frequency() -> None:
    ranked = rank_lexical(
        "payment payment notice",
        (
            _candidate("exact", "payment payment notice"),
            _candidate("partial", "payment notice term"),
        ),
        limit=2,
    )

    assert [item.candidate.id for item in ranked] == ["exact", "partial"]
    assert ranked[0].score == pytest.approx(1.0)
    assert ranked[1].score == pytest.approx(3 / (math.sqrt(5) * math.sqrt(3)))
    assert [item.rank for item in ranked] == [1, 2]


def test_rank_lexical_ties_ignore_input_order() -> None:
    first = _candidate("b", "notice", document_id="b-doc", start=10)
    second = _candidate("a", "notice", document_id="a-doc", start=20)

    forward = rank_lexical("notice", (first, second), limit=2)
    reverse = rank_lexical("notice", (second, first), limit=2)

    assert [item.candidate.id for item in forward] == ["a", "b"]
    assert [item.candidate.id for item in reverse] == ["a", "b"]


def test_rank_lexical_excludes_zero_overlap_and_empty_queries() -> None:
    candidates = (_candidate("one", "payment notice"),)

    assert rank_lexical("termination", candidates, limit=1) == ()
    assert rank_lexical("---", candidates, limit=1) == ()


def test_rank_lexical_caps_results() -> None:
    candidates = (_candidate("a", "notice"), _candidate("b", "notice", start=20))

    assert len(rank_lexical("notice", candidates, limit=1)) == 1


@pytest.mark.parametrize(
    "candidates, message",
    [
        ((_candidate("", "text"),), "id"),
        ((_candidate("a", ""),), "text"),
        ((SpanCandidate("a", "doc", -1, 2, "abc"),), "start"),
        ((SpanCandidate("a", "doc", 2, 2, "abc"),), "end"),
        ((_candidate("same", "one"), _candidate("same", "two", start=10)), "duplicate"),
    ],
)
def test_rank_lexical_rejects_invalid_candidates(
    candidates: tuple[SpanCandidate, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        rank_lexical("text", candidates, limit=2)


def test_rank_lexical_rejects_nonpositive_limit() -> None:
    with pytest.raises(ValueError, match="limit"):
        rank_lexical("text", (), limit=0)


def test_rank_lexical_rejects_boolean_limit_and_offsets() -> None:
    with pytest.raises(ValueError, match="limit"):
        rank_lexical("text", (), limit=True)
    with pytest.raises(ValueError, match="start"):
        rank_lexical("text", (SpanCandidate("a", "doc", True, 4, "text"),), limit=1)
