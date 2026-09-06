"""Offset-preserving chunker adapters used by the CUAD evaluation.

Every adapter returns ``(start, end)`` character spans into the *original*
document text. Span containment is only meaningful if those offsets are exact,
so the adapters never rely on a splitter's own bookkeeping unless it is known
to be exact (LexiChunk's is; LangChain's splitters do not expose offsets at
all, so their chunks are re-located in the source with a forward-only cursor).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

#: Sentence boundary: terminal punctuation followed by whitespace. Deliberately
#: naive — this is the *baseline* everyone reaches for, not a good legal parser.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True, slots=True)
class Span:
    """A half-open character span ``[start, end)`` in the source document."""

    start: int
    end: int

    @property
    def length(self) -> int:
        """Number of characters covered."""
        return self.end - self.start

    def contains(self, other: Span) -> bool:
        """True when ``other`` lies wholly inside this span."""
        return self.start <= other.start and other.end <= self.end

    def overlaps(self, other: Span) -> bool:
        """True when the two spans share at least one character."""
        return self.start < other.end and other.start < self.end


def locate_sequentially(text: str, pieces: Sequence[str]) -> list[Span]:
    """Re-locate splitter output in the source with a forward-only cursor.

    LangChain's splitters strip surrounding whitespace and can drop separator
    characters, so the concatenation of their output is not the input. Scanning
    forward from the end of the previous match keeps the mapping monotonic and
    avoids matching a repeated boilerplate line back at the top of the file.

    A piece that cannot be found from the cursor (which would mean the splitter
    rewrote the text) is skipped rather than silently mis-located; the caller
    can compare ``len(result)`` to ``len(pieces)`` to detect that.
    """
    spans: list[Span] = []
    cursor = 0
    for piece in pieces:
        if not piece:
            continue
        index = text.find(piece, cursor)
        if index < 0:
            index = text.find(piece)
            if index < 0:
                continue
        spans.append(Span(index, index + len(piece)))
        cursor = index + len(piece)
    return spans


def sentence_window_spans(text: str, window: int = 3, stride: int = 2) -> list[Span]:
    """Sentence-window splitting: overlapping windows of ``window`` sentences.

    Args:
        text: Document text.
        window: Sentences per chunk.
        stride: Sentences advanced between consecutive chunks. A stride below
            ``window`` produces overlap, which mechanically helps containment —
            that is exactly the confound the CUAD report is designed to expose.

    Returns:
        Character spans covering each window, in document order.
    """
    if window < 1 or stride < 1:
        msg = f"window and stride must be >= 1, got window={window} stride={stride}"
        raise ValueError(msg)

    bounds = list(_sentence_bounds(text))
    if not bounds:
        return [Span(0, len(text))] if text else []

    spans: list[Span] = []
    for i in range(0, len(bounds), stride):
        group = bounds[i : i + window]
        if not group:
            break
        spans.append(Span(group[0][0], group[-1][1]))
        if i + window >= len(bounds):
            break
    return spans


def _sentence_bounds(text: str) -> Iterator[tuple[int, int]]:
    """Yield ``(start, end)`` for each naive sentence, skipping blank runs."""
    cursor = 0
    for match in _SENTENCE_RE.finditer(text):
        end = match.start()
        if text[cursor:end].strip():
            yield (cursor, end)
        cursor = match.end()
    if text[cursor:].strip():
        yield (cursor, len(text))


def rcts_spans(text: str, chunk_size: int, chunk_overlap: int = 0) -> list[Span]:
    """Character spans for LangChain's ``RecursiveCharacterTextSplitter``."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return locate_sequentially(text, splitter.split_text(text))


def lexichunk_spans(
    text: str,
    *,
    jurisdiction: str = "us",
    max_chunk_size: int = 512,
) -> list[Span]:
    """Character spans for LexiChunk.

    ``max_chunk_size`` is measured in **tokens** (LexiChunk assumes 4 characters
    per token by default), not characters — the single most important thing to
    hold in mind when comparing against character-based splitters.
    """
    from lexichunk import LegalChunker

    chunker = LegalChunker(jurisdiction=jurisdiction, max_chunk_size=max_chunk_size)
    return [Span(c.char_start, c.char_end) for c in chunker.chunk(text)]
