"""Tests for LexiChunkContextualStrategy and contextual retrieval headers."""

from __future__ import annotations

from scaffolder.chunking import get_all_strategies, get_strategy
from scaffolder.chunking.strategies import (
    LexiChunkContextualStrategy,
    LexiChunkStrategy,
    _build_context_header,
)
from scaffolder.models import (
    Document,
    DocumentType,
    Jurisdiction,
    StrategyName,
)

TINY_DOC = Document(
    id="ctx_test",
    text=(
        "1. DEFINITIONS\n"
        '"Service Provider" means the party providing services under this Agreement.\n'
        '"Client" means the party receiving services.\n\n'
        "2. OBLIGATIONS\n"
        "2.1 The Service Provider shall deliver services as defined in Section 1.\n"
        "2.2 Subject to Clause 3, the Client shall pay the fees.\n\n"
        "3. PAYMENT TERMS\n"
        "3.1 Fees are due within 30 days of invoice date.\n"
    ),
    jurisdiction=Jurisdiction.UK,
    document_type=DocumentType.SERVICE_AGREEMENT,
    source="ctx_test.txt",
)


class TestLexiChunkContextualStrategy:
    def test_produces_chunks_with_contextual_metadata(self) -> None:
        strategy = LexiChunkContextualStrategy()
        cs = strategy.chunk(TINY_DOC)
        assert cs.count >= 1
        for chunk in cs.chunks:
            assert chunk.metadata.get("contextual") is True

    def test_contextual_chunk_text_longer_than_raw(self) -> None:
        """Contextual chunks should be at least as long as raw (header added)."""
        raw = LexiChunkStrategy()
        ctx = LexiChunkContextualStrategy()
        raw_cs = raw.chunk(TINY_DOC)
        ctx_cs = ctx.chunk(TINY_DOC)
        assert raw_cs.count == ctx_cs.count
        for raw_chunk, ctx_chunk in zip(raw_cs.chunks, ctx_cs.chunks, strict=True):
            assert ctx_chunk.char_count >= raw_chunk.char_count

    def test_contextual_chunk_has_original_text(self) -> None:
        strategy = LexiChunkContextualStrategy()
        cs = strategy.chunk(TINY_DOC)
        for chunk in cs.chunks:
            assert "original_text" in chunk.metadata
            assert len(chunk.metadata["original_text"]) > 0  # type: ignore[arg-type]

    def test_same_chunk_count_as_raw(self) -> None:
        raw = LexiChunkStrategy()
        ctx = LexiChunkContextualStrategy()
        assert raw.chunk(TINY_DOC).count == ctx.chunk(TINY_DOC).count

    def test_chunk_ids_follow_convention(self) -> None:
        strategy = LexiChunkContextualStrategy()
        cs = strategy.chunk(TINY_DOC)
        for i, chunk in enumerate(cs.chunks):
            assert chunk.id == f"lexichunk_ctx_ctx_test_{i}"

    def test_strategy_name(self) -> None:
        strategy = LexiChunkContextualStrategy()
        assert strategy.name == StrategyName.LEXICHUNK_CONTEXTUAL

    def test_elapsed_seconds_positive(self) -> None:
        strategy = LexiChunkContextualStrategy()
        cs = strategy.chunk(TINY_DOC)
        assert cs.elapsed_seconds > 0


class TestBuildContextHeader:
    def test_empty_header_for_bare_chunk(self) -> None:
        """A minimal object with no attributes produces empty header."""

        class BareChunk:
            content = "some text"

        assert _build_context_header(BareChunk()) == ""

    def test_header_contains_section(self) -> None:
        class WithHierarchy:
            hierarchy_path = "1 > 1.2"

        header = _build_context_header(WithHierarchy())
        assert "Section:" in header
        assert "1 > 1.2" in header

    def test_header_contains_clause_type(self) -> None:
        class WithType:
            clause_type = type("CT", (), {"value": "obligation"})()

        header = _build_context_header(WithType())
        assert "Type: obligation" in header

    def test_header_contains_terms(self) -> None:
        class WithTerms:
            defined_terms_used = {"Service Provider", "Client"}

        header = _build_context_header(WithTerms())
        assert "Terms:" in header

    def test_header_format(self) -> None:
        class FullChunk:
            hierarchy_path = "2.1"
            clause_type = type("CT", (), {"value": "obligation"})()
            defined_terms_used = {"Client"}
            cross_references = []

        header = _build_context_header(FullChunk())
        assert header.startswith("[")
        assert header.strip().endswith("]")


class TestContextualRegistry:
    def test_registry_includes_contextual(self) -> None:
        strategies = get_all_strategies()
        names = {s.name for s in strategies}
        assert StrategyName.LEXICHUNK_CONTEXTUAL in names

    def test_get_strategy_contextual(self) -> None:
        strategy = get_strategy(StrategyName.LEXICHUNK_CONTEXTUAL)
        assert strategy.name == StrategyName.LEXICHUNK_CONTEXTUAL

    def test_registry_now_has_six(self) -> None:
        strategies = get_all_strategies()
        assert len(strategies) == 6
