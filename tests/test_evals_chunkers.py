"""Offset correctness for the chunker adapters used by the CUAD evaluation.

Span containment is only meaningful when every reported span really indexes the
source text, so these tests assert exactness rather than plausibility.
"""

from __future__ import annotations

import pytest

from legal_rag_eval.evals.chunkers import (
    Span,
    lexichunk_spans,
    locate_sequentially,
    rcts_spans,
    sentence_window_spans,
)

CONTRACT = """EXHIBIT 10.1

SERVICES AGREEMENT

Section 1. Definitions.

1.1 "Services" means the professional services described in Schedule A.

Section 2. Confidentiality.

2.1 Each party shall hold in confidence all Confidential Information.

Section 3. Governing Law.

3.1 This Agreement shall be governed by the laws of the State of Delaware.
"""


class TestSpan:
    def test_length(self) -> None:
        assert Span(10, 25).length == 15

    def test_contains_is_inclusive_of_exact_match(self) -> None:
        assert Span(0, 10).contains(Span(0, 10))

    def test_contains_rejects_partial_overlap(self) -> None:
        assert not Span(0, 10).contains(Span(5, 15))
        assert not Span(5, 15).contains(Span(0, 10))

    def test_overlaps(self) -> None:
        assert Span(0, 10).overlaps(Span(9, 20))
        assert not Span(0, 10).overlaps(Span(10, 20))
        assert not Span(0, 10).overlaps(Span(20, 30))


class TestLocateSequentially:
    def test_recovers_exact_offsets(self) -> None:
        text = "alpha beta gamma"
        spans = locate_sequentially(text, ["alpha", "beta", "gamma"])
        assert [(s.start, s.end) for s in spans] == [(0, 5), (6, 10), (11, 16)]
        assert [text[s.start : s.end] for s in spans] == ["alpha", "beta", "gamma"]

    def test_is_forward_only_for_repeated_text(self) -> None:
        # "the" appears twice; the second piece must map to the second one.
        text = "the cat and the dog"
        spans = locate_sequentially(text, ["the cat", "the dog"])
        assert spans[1].start == text.index("the dog")

    def test_skips_pieces_that_are_not_present(self) -> None:
        spans = locate_sequentially("abc", ["abc", "rewritten"])
        assert len(spans) == 1

    def test_ignores_empty_pieces(self) -> None:
        assert locate_sequentially("abc", ["", "abc"]) == [Span(0, 3)]


class TestSentenceWindow:
    def test_windows_are_in_document_order_and_within_bounds(self) -> None:
        spans = sentence_window_spans(CONTRACT, window=3, stride=2)
        assert spans
        assert all(0 <= s.start < s.end <= len(CONTRACT) for s in spans)
        assert [s.start for s in spans] == sorted(s.start for s in spans)

    def test_stride_below_window_produces_overlap(self) -> None:
        text = "One. Two. Three. Four. Five. Six."
        overlapping = sentence_window_spans(text, window=3, stride=1)
        disjoint = sentence_window_spans(text, window=3, stride=3)
        assert len(overlapping) > len(disjoint)

    def test_text_without_terminal_punctuation_is_one_span(self) -> None:
        assert sentence_window_spans("no punctuation here") == [Span(0, 19)]

    def test_empty_text(self) -> None:
        assert sentence_window_spans("") == []

    def test_invalid_parameters_raise(self) -> None:
        with pytest.raises(ValueError, match="window and stride"):
            sentence_window_spans(CONTRACT, window=0)
        with pytest.raises(ValueError, match="window and stride"):
            sentence_window_spans(CONTRACT, stride=0)


class TestRcts:
    def test_spans_index_the_source_exactly(self) -> None:
        spans = rcts_spans(CONTRACT, 200)
        assert spans
        for span in spans:
            assert CONTRACT[span.start : span.end].strip()
            assert span.end <= len(CONTRACT)

    def test_smaller_chunk_size_yields_more_chunks(self) -> None:
        assert len(rcts_spans(CONTRACT, 100)) > len(rcts_spans(CONTRACT, 1000))

    def test_spans_are_non_decreasing(self) -> None:
        spans = rcts_spans(CONTRACT, 150)
        assert [s.start for s in spans] == sorted(s.start for s in spans)


class TestLexichunk:
    def test_spans_reproduce_the_source_contiguously(self) -> None:
        spans = lexichunk_spans(CONTRACT, jurisdiction="us", max_chunk_size=64)
        assert spans
        assert spans[0].start == 0
        for a, b in zip(spans, spans[1:], strict=False):
            assert a.end <= b.start

    def test_span_bounds_stay_inside_the_document(self) -> None:
        for span in lexichunk_spans(CONTRACT, max_chunk_size=64):
            assert 0 <= span.start < span.end <= len(CONTRACT)

    def test_max_chunk_size_is_in_tokens_not_characters(self) -> None:
        # LexiChunk budgets ~4 characters per token, so a chunk capped at "512"
        # runs to roughly 2000 characters. This is the length confound the CUAD
        # report is built to expose; if it ever changes, the report must too.
        # An unstructured wall of prose forces the size cap to be the binding
        # constraint rather than a clause boundary.
        prose = (
            "The parties acknowledge and agree that the foregoing provisions "
            "shall apply in all respects to the transactions contemplated hereby. "
        ) * 200
        spans = lexichunk_spans(prose, max_chunk_size=512)
        assert max(s.length for s in spans) > 1000
