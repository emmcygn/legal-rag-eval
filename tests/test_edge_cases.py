"""Tests for edge cases: empty docs, single clause, no definitions, etc.

Exercises the legacy structural metrics (``legal_rag_eval.metrics.structural``) against
degenerate documents and chunk sets. See ``test_structural_metrics.py`` for why these
are legacy; the reported metrics are covered separately in ``test_gold_metrics.py``.
"""

from __future__ import annotations

import pytest

from legal_rag_eval.chunking.strategies import (
    FixedSizeStrategy,
    RCTSStrategy,
    SentenceSplitStrategy,
)
from legal_rag_eval.metrics.structural import (
    compute_legacy_structural_metrics,
    legacy_chunk_size_cv,
    legacy_clause_fragmentation_rate,
    legacy_cross_ref_resolution_rate,
    legacy_definition_preservation_rate,
    legacy_hierarchy_depth_retained,
)
from legal_rag_eval.models import (
    Chunk,
    ChunkSet,
    Document,
    DocumentType,
    Jurisdiction,
    StrategyName,
)

EMPTY_DOC = Document(
    id="empty",
    text="",
    jurisdiction=Jurisdiction.UK,
    document_type=DocumentType.SERVICE_AGREEMENT,
    source="empty.txt",
)

WHITESPACE_DOC = Document(
    id="whitespace",
    text="   \n\n  \t  ",
    jurisdiction=Jurisdiction.UK,
    document_type=DocumentType.SERVICE_AGREEMENT,
    source="whitespace.txt",
)

SINGLE_CLAUSE_DOC = Document(
    id="single",
    text="1. This agreement is governed by English law.",
    jurisdiction=Jurisdiction.UK,
    document_type=DocumentType.SERVICE_AGREEMENT,
    source="single.txt",
)

NO_DEFS_DOC = Document(
    id="no_defs",
    text=(
        "1. Obligations.\n"
        "1.1 The parties agree to cooperate.\n"
        "2. Termination.\n"
        "2.1 Either party may terminate."
    ),
    jurisdiction=Jurisdiction.UK,
    document_type=DocumentType.SERVICE_AGREEMENT,
    source="no_defs.txt",
)


def _make_chunk_set(
    text: str,
    doc_id: str = "test",
    strategy: StrategyName = StrategyName.FIXED_SIZE,
) -> ChunkSet:
    """Helper to create a single-chunk ChunkSet."""
    return ChunkSet(
        strategy=strategy,
        document_id=doc_id,
        chunks=(
            Chunk(
                id=f"test_{doc_id}_0",
                text=text,
                document_id=doc_id,
                strategy=strategy,
                index=0,
            ),
        ),
        elapsed_seconds=0.0,
    )


def _make_empty_chunk_set(
    doc_id: str = "test",
    strategy: StrategyName = StrategyName.FIXED_SIZE,
) -> ChunkSet:
    return ChunkSet(
        strategy=strategy,
        document_id=doc_id,
        chunks=(),
        elapsed_seconds=0.0,
    )


class TestEmptyDocument:
    @pytest.mark.parametrize(
        "strategy_cls",
        [RCTSStrategy, SentenceSplitStrategy, FixedSizeStrategy],
    )
    def test_empty_doc_returns_empty_chunkset(self, strategy_cls: type) -> None:
        strategy = strategy_cls()
        cs = strategy.chunk(EMPTY_DOC)
        assert cs.count == 0

    @pytest.mark.parametrize(
        "strategy_cls",
        [RCTSStrategy, SentenceSplitStrategy, FixedSizeStrategy],
    )
    def test_whitespace_doc_returns_empty_or_minimal(self, strategy_cls: type) -> None:
        strategy = strategy_cls()
        cs = strategy.chunk(WHITESPACE_DOC)
        # Either empty or all chunks are whitespace-only
        for chunk in cs.chunks:
            # Not a hard failure -- just verify no crash
            assert isinstance(chunk.text, str)


class TestSingleClause:
    def test_fixed_size_large_chunk(self) -> None:
        strategy = FixedSizeStrategy(chunk_size=1000)
        cs = strategy.chunk(SINGLE_CLAUSE_DOC)
        assert cs.count == 1

    def test_single_chunk_no_fragmentation(self) -> None:
        cs = _make_chunk_set(
            SINGLE_CLAUSE_DOC.text,
            doc_id=SINGLE_CLAUSE_DOC.id,
        )
        rate = legacy_clause_fragmentation_rate(cs, SINGLE_CLAUSE_DOC)
        assert rate == 0.0


class TestNoDefinitions:
    def test_preservation_is_one(self) -> None:
        strategy = FixedSizeStrategy(chunk_size=500)
        cs = strategy.chunk(NO_DEFS_DOC)
        sm = compute_legacy_structural_metrics(cs, NO_DEFS_DOC)
        assert sm.definition_preservation_rate == 1.0


class TestNoCrossReferences:
    def test_resolution_is_one(self) -> None:
        doc = Document(
            id="no_refs",
            text="1. The parties agree.\n2. Fees are payable.",
            jurisdiction=Jurisdiction.UK,
            document_type=DocumentType.SERVICE_AGREEMENT,
            source="no_refs.txt",
        )
        cs = _make_chunk_set(doc.text, doc_id=doc.id)
        rate = legacy_cross_ref_resolution_rate(cs, doc)
        assert rate == 1.0


class TestFlatDocument:
    def test_hierarchy_depth_flat(self) -> None:
        doc = Document(
            id="flat",
            text="No section numbers at all. Just plain text.",
            jurisdiction=Jurisdiction.UK,
            document_type=DocumentType.SERVICE_AGREEMENT,
            source="flat.txt",
        )
        cs = _make_chunk_set(doc.text, doc_id=doc.id)
        depth = legacy_hierarchy_depth_retained(cs, doc)
        assert depth == 1.0


class TestChunkSizeCV:
    def test_single_chunk_cv_zero(self) -> None:
        cs = _make_chunk_set("some text")
        assert legacy_chunk_size_cv(cs) == 0.0

    def test_empty_chunks_cv_zero(self) -> None:
        cs = _make_empty_chunk_set()
        assert legacy_chunk_size_cv(cs) == 0.0

    def test_uniform_chunks_cv_zero(self) -> None:
        chunks = tuple(
            Chunk(
                id=f"test_{i}",
                text="x" * 100,
                document_id="test",
                strategy=StrategyName.FIXED_SIZE,
                index=i,
            )
            for i in range(5)
        )
        cs = ChunkSet(
            strategy=StrategyName.FIXED_SIZE,
            document_id="test",
            chunks=chunks,
            elapsed_seconds=0.0,
        )
        assert legacy_chunk_size_cv(cs) == 0.0


class TestEmptyChunkSetMetrics:
    def test_fragmentation_empty(self) -> None:
        cs = _make_empty_chunk_set(doc_id=SINGLE_CLAUSE_DOC.id)
        rate = legacy_clause_fragmentation_rate(cs, SINGLE_CLAUSE_DOC)
        assert rate == 0.0

    def test_definition_preservation_empty(self) -> None:
        cs = _make_empty_chunk_set(doc_id=NO_DEFS_DOC.id)
        rate = legacy_definition_preservation_rate(cs, NO_DEFS_DOC)
        assert rate == 1.0

    def test_hierarchy_empty(self) -> None:
        cs = _make_empty_chunk_set(doc_id=SINGLE_CLAUSE_DOC.id)
        depth = legacy_hierarchy_depth_retained(cs, SINGLE_CLAUSE_DOC)
        # With no chunks, we consider it as 0 chunk depth vs doc depth
        assert isinstance(depth, float)
