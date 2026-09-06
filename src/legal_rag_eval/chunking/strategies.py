"""Chunking strategy wrappers for LexiChunk and baselines."""

from __future__ import annotations

import logging
import re
import time
from typing import Any

from legal_rag_eval.models import (
    Chunk,
    ChunkSet,
    Document,
    StrategyName,
)

logger = logging.getLogger(__name__)


def _build_context_header(legal_chunk: Any) -> str:
    """Build context header for contextual retrieval when build_embedded_text unavailable.

    Format:
    [Section: 2.1 | Type: obligation | Terms: Service Provider, Client]
    <original chunk text>
    """
    parts: list[str] = []

    if hasattr(legal_chunk, "hierarchy_path") and legal_chunk.hierarchy_path:
        parts.append(f"Section: {legal_chunk.hierarchy_path}")

    clause_type = getattr(legal_chunk, "clause_type", None)
    if clause_type is not None:
        ct_str = str(clause_type.value) if hasattr(clause_type, "value") else str(clause_type)
        parts.append(f"Type: {ct_str}")

    if hasattr(legal_chunk, "defined_terms_used") and legal_chunk.defined_terms_used:
        terms = ", ".join(sorted(legal_chunk.defined_terms_used)[:5])
        parts.append(f"Terms: {terms}")

    if hasattr(legal_chunk, "cross_references") and legal_chunk.cross_references:
        refs = [str(r.raw_text) for r in legal_chunk.cross_references[:3]]
        parts.append(f"Refs: {', '.join(refs)}")

    if parts:
        return f"[{' | '.join(parts)}]\n"
    return ""


def _xref_target_string(cross_reference: Any) -> str:
    """Compose the cross-reference target string recorded in chunk metadata.

    LexiChunk reports a reference's numbering (``target_identifier``, e.g. ``"VIII"``)
    and its label word (``target_kind``, e.g. ``"article"``) as two separate fields.
    The gold-scored ``xref_target_recall``/``xref_target_precision`` metrics
    (:mod:`legal_rag_eval.metrics.gold`) compare against a single string via
    ``normalize_identifier``, which folds a leading label word into the identifier
    (``"article VIII"`` -> ``"article_8"``) or strips it entirely for a plain clause
    number (``"clause 12.3"`` -> ``"12.3"``). Emitting only the bare identifier (as
    before) discards that label, so every article/section-scoped target silently
    failed to match its gold counterpart.
    """
    identifier = str(cross_reference.target_identifier)
    kind = str(getattr(cross_reference, "target_kind", "") or "").strip()
    return f"{kind} {identifier}" if kind else identifier


class LexiChunkStrategy:
    """Wrapper around LexiChunk's LegalChunker.

    Extracts rich metadata into Chunk.metadata:
    - clause_type: str (e.g. "definitions", "obligation", "limitation")
    - confidence: float (0.0-1.0)
    - defined_terms: list[str]
    - cross_references: list[dict]
    - section_hierarchy: str (e.g. "1 > Definitions.")

    A single strategy instance is reused across every fixture document in a benchmark
    run (see ``_STRATEGY_REGISTRY``), so the ``LegalChunker`` cannot be built once in
    ``__init__`` with a fixed jurisdiction -- the fixtures mix UK, US and EU documents,
    and ``LegalChunker`` defaults to ``jurisdiction="uk"``. Parsing a US document (Roman
    numeral "Article VIII" headers) under the UK profile fails to recognise the
    article-level numbering at all, which collapses ``hierarchy_path`` to just the
    deepest leaf and silently drops every heading/cross-reference claim above it.
    ``_chunker_for`` builds (and caches) one chunker per jurisdiction, defaulting to
    each document's own ``document.jurisdiction`` -- unless the caller pinned a
    jurisdiction explicitly via constructor kwargs, in which case that pin wins for
    every document, preserving the previous override behaviour.
    """

    name = StrategyName.LEXICHUNK

    def __init__(self, **kwargs: Any) -> None:
        self._kwargs = kwargs
        self._chunkers: dict[str, Any] = {}

    def _chunker_for(self, document: Document) -> Any:
        from lexichunk import LegalChunker

        jurisdiction = self._kwargs.get("jurisdiction", document.jurisdiction.value)
        chunker = self._chunkers.get(jurisdiction)
        if chunker is None:
            chunker = LegalChunker(**{**self._kwargs, "jurisdiction": jurisdiction})
            self._chunkers[jurisdiction] = chunker
        return chunker

    def chunk(self, document: Document) -> ChunkSet:
        chunker = self._chunker_for(document)
        start = time.perf_counter()
        legal_chunks = chunker.chunk(document.text, document_id=document.id)
        elapsed = time.perf_counter() - start

        chunks: list[Chunk] = []
        for i, lc in enumerate(legal_chunks):
            metadata: dict[str, Any] = {}
            if lc.clause_type is not None:
                ct = lc.clause_type
                metadata["clause_type"] = str(ct.value) if hasattr(ct, "value") else str(ct)
            if lc.classification_confidence is not None:
                metadata["confidence"] = lc.classification_confidence
            if lc.defined_terms_used:
                metadata["defined_terms"] = list(lc.defined_terms_used)
            if lc.cross_references:
                metadata["cross_references"] = [
                    {"raw_text": str(cr.raw_text), "target": _xref_target_string(cr)}
                    for cr in lc.cross_references
                ]
            if lc.hierarchy_path:
                metadata["section_hierarchy"] = str(lc.hierarchy_path)

            chunks.append(
                Chunk(
                    id=f"lexichunk_{document.id}_{i}",
                    text=lc.content,
                    document_id=document.id,
                    strategy=self.name,
                    index=i,
                    metadata=metadata,
                )
            )

        return ChunkSet(
            strategy=self.name,
            document_id=document.id,
            chunks=tuple(chunks),
            elapsed_seconds=elapsed,
        )


class LexiChunkContextualStrategy:
    """LexiChunk with contextual retrieval headers.

    Uses context headers prepended to each chunk before embedding. The headers
    include section hierarchy, clause type, defined terms, and cross-references.
    This implements Anthropic's contextual retrieval approach for legal documents.
    """

    name = StrategyName.LEXICHUNK_CONTEXTUAL

    def __init__(self, **kwargs: Any) -> None:
        self._kwargs = kwargs
        self._chunkers: dict[str, Any] = {}

    def _chunker_for(self, document: Document) -> Any:
        from lexichunk import LegalChunker

        # See LexiChunkStrategy._chunker_for: one strategy instance is reused across
        # every fixture document, so the chunker's jurisdiction has to be resolved per
        # document rather than fixed at construction time.
        jurisdiction = self._kwargs.get("jurisdiction", document.jurisdiction.value)
        chunker = self._chunkers.get(jurisdiction)
        if chunker is None:
            chunker = LegalChunker(**{**self._kwargs, "jurisdiction": jurisdiction})
            self._chunkers[jurisdiction] = chunker
        return chunker

    def chunk(self, document: Document) -> ChunkSet:
        chunker = self._chunker_for(document)
        start = time.perf_counter()
        legal_chunks = chunker.chunk(document.text, document_id=document.id)
        elapsed = time.perf_counter() - start

        chunks: list[Chunk] = []
        for i, lc in enumerate(legal_chunks):
            # Build contextual embedded text with header
            embedded_text = _build_context_header(lc) + lc.content

            metadata: dict[str, Any] = {"contextual": True}
            if lc.clause_type is not None:
                ct = lc.clause_type
                metadata["clause_type"] = str(ct.value) if hasattr(ct, "value") else str(ct)
            if lc.classification_confidence is not None:
                metadata["confidence"] = lc.classification_confidence
            if lc.defined_terms_used:
                metadata["defined_terms"] = list(lc.defined_terms_used)
            if lc.cross_references:
                metadata["cross_references"] = [
                    {"raw_text": str(cr.raw_text), "target": _xref_target_string(cr)}
                    for cr in lc.cross_references
                ]
            if lc.hierarchy_path:
                metadata["section_hierarchy"] = str(lc.hierarchy_path)
            # Store original text for structural metrics
            metadata["original_text"] = lc.content

            chunks.append(
                Chunk(
                    id=f"lexichunk_ctx_{document.id}_{i}",
                    text=embedded_text,
                    document_id=document.id,
                    strategy=self.name,
                    index=i,
                    metadata=metadata,
                )
            )

        return ChunkSet(
            strategy=self.name,
            document_id=document.id,
            chunks=tuple(chunks),
            elapsed_seconds=elapsed,
        )


class RCTSStrategy:
    """Wrapper around LangChain's RecursiveCharacterTextSplitter.

    The chunk size is part of the strategy's identity, not a hidden default: the same
    splitter at 512 and at 1024 characters is two different baselines, and the second one
    is the size-matched control for LexiChunk. Pass ``name`` to label the variant.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        name: StrategyName = StrategyName.RCTS_512,
    ) -> None:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self.name = name

    def chunk(self, document: Document) -> ChunkSet:
        start = time.perf_counter()
        texts = self._splitter.split_text(document.text)
        elapsed = time.perf_counter() - start

        chunks = tuple(
            Chunk(
                id=f"{self.name.value}_{document.id}_{i}",
                text=t,
                document_id=document.id,
                strategy=self.name,
                index=i,
                metadata={
                    "chunk_size_param": self._chunk_size,
                    "chunk_overlap_param": self._chunk_overlap,
                },
            )
            for i, t in enumerate(texts)
        )

        return ChunkSet(
            strategy=self.name,
            document_id=document.id,
            chunks=chunks,
            elapsed_seconds=elapsed,
        )


class SentenceSplitStrategy:
    """Split text on sentence boundaries using regex.

    Splits on: period/question/exclamation followed by whitespace + uppercase,
    OR double newline. Short fragments are merged into the previous chunk
    to avoid tiny chunks.
    """

    name = StrategyName.SENTENCE_SPLIT

    _SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+(?=[A-Z])|(?:\n\s*\n)")

    def __init__(self, min_chunk_chars: int = 100) -> None:
        self._min_chunk_chars = min_chunk_chars

    def chunk(self, document: Document) -> ChunkSet:
        start = time.perf_counter()
        raw_sentences = self._SENTENCE_PATTERN.split(document.text)

        # Merge short sentences to avoid tiny chunks
        merged: list[str] = []
        buffer = ""
        for sentence in raw_sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if buffer and (
                len(buffer) < self._min_chunk_chars
                or len(buffer) + len(sentence) < self._min_chunk_chars
            ):
                buffer = buffer + " " + sentence
            else:
                if buffer:
                    merged.append(buffer)
                buffer = sentence
        if buffer:
            merged.append(buffer)

        elapsed = time.perf_counter() - start

        chunks = tuple(
            Chunk(
                id=f"sentence_split_{document.id}_{i}",
                text=t,
                document_id=document.id,
                strategy=self.name,
                index=i,
                metadata={},
            )
            for i, t in enumerate(merged)
        )

        return ChunkSet(
            strategy=self.name,
            document_id=document.id,
            chunks=chunks,
            elapsed_seconds=elapsed,
        )


class FixedSizeStrategy:
    """Fixed-size chunking at exactly N characters with no overlap.

    This is the simplest possible baseline. Chunks are cut at exact
    character boundaries with no regard for words or sentences.
    """

    name = StrategyName.FIXED_SIZE

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 0) -> None:
        if chunk_overlap:
            msg = "FixedSizeStrategy does not implement overlap; pass chunk_overlap=0."
            raise ValueError(msg)
        self._chunk_size = chunk_size

    def chunk(self, document: Document) -> ChunkSet:
        start = time.perf_counter()
        text = document.text
        texts: list[str] = []
        for i in range(0, len(text), self._chunk_size):
            segment = text[i : i + self._chunk_size]
            if segment.strip():
                texts.append(segment)
        elapsed = time.perf_counter() - start

        chunks = tuple(
            Chunk(
                id=f"fixed_size_{document.id}_{i}",
                text=t,
                document_id=document.id,
                strategy=self.name,
                index=i,
                metadata={"chunk_size_param": self._chunk_size},
            )
            for i, t in enumerate(texts)
        )

        return ChunkSet(
            strategy=self.name,
            document_id=document.id,
            chunks=chunks,
            elapsed_seconds=elapsed,
        )
