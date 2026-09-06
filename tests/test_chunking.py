"""Tests for ChunkingPipeline and strategy wrappers."""

from __future__ import annotations

import pytest

from scaffolder.chunking import (
    DEFAULT_STRATEGIES,
    ChunkingPipeline,
    FixedSizeStrategy,
    LexiChunkStrategy,
    RCTSStrategy,
    SentenceSplitStrategy,
    get_all_strategies,
    get_strategy,
    strategy_params,
)
from scaffolder.fixtures import FixtureManager
from scaffolder.models import (
    Document,
    DocumentType,
    Jurisdiction,
    StrategyName,
)

TINY_DOC = Document(
    id="test_doc",
    text=(
        "1. Definitions.\n"
        '1.1 "Service Provider" means the party providing services under this Agreement.\n'
        '1.2 "Client" means the party receiving services.\n'
        "2. Obligations.\n"
        "2.1 The Service Provider shall deliver services as defined in Section 1.1.\n"
        "2.2 Subject to Clause 3, the Client shall pay the fees set out in Schedule 1.\n"
        "3. Payment Terms.\n"
        "3.1 Fees are due within 30 days of invoice date.\n"
        "3.2 Late payments shall accrue interest at 4% above the base rate.\n"
    ),
    jurisdiction=Jurisdiction.UK,
    document_type=DocumentType.SERVICE_AGREEMENT,
    source="test.txt",
)


class TestLexiChunkStrategy:
    def test_returns_chunkset(self) -> None:
        strategy = LexiChunkStrategy()
        cs = strategy.chunk(TINY_DOC)
        assert cs.count >= 1
        assert cs.document_id == "test_doc"
        assert cs.strategy == StrategyName.LEXICHUNK

    def test_chunks_have_correct_strategy(self) -> None:
        strategy = LexiChunkStrategy()
        cs = strategy.chunk(TINY_DOC)
        for chunk in cs.chunks:
            assert chunk.strategy == StrategyName.LEXICHUNK

    def test_chunk_ids_follow_convention(self) -> None:
        strategy = LexiChunkStrategy()
        cs = strategy.chunk(TINY_DOC)
        for i, chunk in enumerate(cs.chunks):
            assert chunk.id == f"lexichunk_test_doc_{i}"

    def test_has_metadata(self) -> None:
        strategy = LexiChunkStrategy()
        cs = strategy.chunk(TINY_DOC)
        # At least some chunks should have metadata
        has_metadata = any(len(c.metadata) > 0 for c in cs.chunks)
        assert has_metadata

    def test_elapsed_seconds_positive(self) -> None:
        strategy = LexiChunkStrategy()
        cs = strategy.chunk(TINY_DOC)
        assert cs.elapsed_seconds > 0


# A US-style document using roman-numeral articles, the numbering convention
# LegalChunker's "uk" default (jurisdiction is never pinned by the strategy's
# constructor kwargs in these tests) does not recognise as a heading at all.
US_DOC = Document(
    id="test_us_doc",
    text=(
        "ARTICLE I - DEFINITIONS\n\n"
        'Section 1.01 Definitions. As used in this Agreement, "Affiliate" means '
        "any entity controlling, controlled by, or under common control with a Party.\n\n"
        "ARTICLE II - TERM\n\n"
        "Section 2.01 Term. This Agreement is effective as of the date set forth "
        "above and shall continue as described in Article I.\n"
    ),
    jurisdiction=Jurisdiction.US,
    document_type=DocumentType.MSA,
    source="test_us.txt",
)


class TestLexiChunkStrategyJurisdiction:
    """A single strategy instance is reused across every fixture document in a real
    benchmark run (see ``chunking._STRATEGY_REGISTRY``), so ``LegalChunker`` cannot be
    built once with a fixed jurisdiction in ``__init__`` -- it must pick up each
    document's own ``jurisdiction`` per call. Before this was fixed, every US fixture
    was parsed under LegalChunker's "uk" default, which does not recognise roman-numeral
    "ARTICLE I" headings, collapsing the hierarchy breadcrumb to just the leaf section
    and making every heading/cross-reference gold metric score 0.0 on US documents.
    """

    def test_uses_document_jurisdiction_for_us_roman_numeral_headings(self) -> None:
        strategy = LexiChunkStrategy()
        cs = strategy.chunk(US_DOC)
        hierarchies = [
            c.metadata.get("section_hierarchy")
            for c in cs.chunks
            if isinstance(c.metadata.get("section_hierarchy"), str)
        ]
        # The Section 2.01 chunk's breadcrumb must carry "Article II" as an ancestor.
        # Under the "uk" default, "ARTICLE II" is not recognised as a heading at all,
        # so the breadcrumb would be flat (just "Section 2.01", no ">" ancestor).
        assert any("Article II" in h and ">" in h for h in hierarchies)

    def test_cross_reference_target_carries_kind_prefix(self) -> None:
        strategy = LexiChunkStrategy()
        cs = strategy.chunk(US_DOC)
        targets = [
            ref["target"] for c in cs.chunks for ref in c.metadata.get("cross_references", [])
        ]
        # The "Article I" cross-reference is article-kind; its recorded target must
        # carry the "article" label so normalize_identifier can fold it into
        # "article_1" -- the bare roman numeral "I" alone normalizes to "i", which
        # matches no gold identifier.
        assert any(t.lower().startswith("article") for t in targets)

    def test_reused_instance_does_not_leak_jurisdiction_across_documents(self) -> None:
        # A real benchmark run chunks a mix of UK, US and EU fixtures with one shared
        # strategy instance; chunking a UK document first must not pin the chunker's
        # jurisdiction for the US document chunked next.
        strategy = LexiChunkStrategy()
        strategy.chunk(TINY_DOC)  # UK document first.
        cs = strategy.chunk(US_DOC)
        hierarchies = [
            c.metadata.get("section_hierarchy")
            for c in cs.chunks
            if isinstance(c.metadata.get("section_hierarchy"), str)
        ]
        assert any("Article II" in h and ">" in h for h in hierarchies)

    def test_explicit_jurisdiction_kwarg_overrides_document_jurisdiction(self) -> None:
        # Preserves the pre-fix override behaviour: a caller that pins a jurisdiction
        # explicitly gets that jurisdiction for every document, regardless of what
        # document.jurisdiction says.
        strategy = LexiChunkStrategy(jurisdiction="uk")
        cs = strategy.chunk(US_DOC)
        hierarchies = [
            c.metadata.get("section_hierarchy")
            for c in cs.chunks
            if isinstance(c.metadata.get("section_hierarchy"), str)
        ]
        assert not any("Article II" in h and ">" in h for h in hierarchies)


class TestRCTSStrategy:
    def test_returns_chunkset(self) -> None:
        strategy = RCTSStrategy()
        cs = strategy.chunk(TINY_DOC)
        assert cs.count >= 1
        assert cs.document_id == "test_doc"
        # Default constructor labels the instance RCTS_512.
        assert cs.strategy == StrategyName.RCTS_512

    def test_chunks_roughly_configured_size(self) -> None:
        strategy = RCTSStrategy(chunk_size=200, name=StrategyName.RCTS_512)
        cs = strategy.chunk(TINY_DOC)
        for chunk in cs.chunks:
            # Allow some tolerance due to splitting logic
            assert chunk.char_count <= 250

    def test_chunk_ids_follow_convention(self) -> None:
        strategy = RCTSStrategy(name=StrategyName.RCTS_512)
        cs = strategy.chunk(TINY_DOC)
        for i, chunk in enumerate(cs.chunks):
            assert chunk.id == f"rcts_512_test_doc_{i}"

    def test_rcts_512_and_1024_are_distinct_strategies(self) -> None:
        """rcts_512 and rcts_1024 must be separately labelled, independent strategies."""
        s512 = get_strategy(StrategyName.RCTS_512)
        s1024 = get_strategy(StrategyName.RCTS_1024)
        assert s512.name == StrategyName.RCTS_512
        assert s1024.name == StrategyName.RCTS_1024
        assert s512.name != s1024.name

    def test_rcts_512_and_1024_produce_different_chunk_counts(self) -> None:
        """On a document large enough to matter, the two chunk sizes must diverge."""
        fm = FixtureManager()
        doc = fm.get_by_id("uk_service_agreement")
        s512 = get_strategy(StrategyName.RCTS_512)
        s1024 = get_strategy(StrategyName.RCTS_1024)
        cs512 = s512.chunk(doc)
        cs1024 = s1024.chunk(doc)
        assert cs512.count != cs1024.count
        # The smaller chunk size must produce more, smaller chunks.
        assert cs512.count > cs1024.count
        assert cs512.avg_chunk_size < cs1024.avg_chunk_size

    def test_rcts_512_records_its_own_size_and_overlap_in_metadata(self) -> None:
        strategy = get_strategy(StrategyName.RCTS_512)
        cs = strategy.chunk(TINY_DOC)
        for chunk in cs.chunks:
            assert chunk.metadata["chunk_size_param"] == 512
            assert chunk.metadata["chunk_overlap_param"] == 50

    def test_rcts_1024_records_its_own_size_and_overlap_in_metadata(self) -> None:
        strategy = get_strategy(StrategyName.RCTS_1024)
        cs = strategy.chunk(TINY_DOC)
        for chunk in cs.chunks:
            assert chunk.metadata["chunk_size_param"] == 1024
            assert chunk.metadata["chunk_overlap_param"] == 100


class TestSentenceSplitStrategy:
    def test_returns_chunkset(self) -> None:
        strategy = SentenceSplitStrategy()
        cs = strategy.chunk(TINY_DOC)
        assert cs.count >= 1
        assert cs.document_id == "test_doc"
        assert cs.strategy == StrategyName.SENTENCE_SPLIT

    def test_min_chunk_chars_respected(self) -> None:
        strategy = SentenceSplitStrategy(min_chunk_chars=50)
        cs = strategy.chunk(TINY_DOC)
        # All chunks except possibly the last should be >= min_chunk_chars
        for chunk in cs.chunks[:-1]:
            assert chunk.char_count >= 50


class TestFixedSizeStrategy:
    def test_returns_chunkset(self) -> None:
        strategy = FixedSizeStrategy()
        cs = strategy.chunk(TINY_DOC)
        assert cs.count >= 1
        assert cs.document_id == "test_doc"
        assert cs.strategy == StrategyName.FIXED_SIZE

    def test_chunks_at_most_configured_size(self) -> None:
        strategy = FixedSizeStrategy(chunk_size=512)
        cs = strategy.chunk(TINY_DOC)
        for chunk in cs.chunks:
            assert chunk.char_count <= 512

    def test_chunk_ids_follow_convention(self) -> None:
        strategy = FixedSizeStrategy()
        cs = strategy.chunk(TINY_DOC)
        for i, chunk in enumerate(cs.chunks):
            assert chunk.id == f"fixed_size_test_doc_{i}"

    def test_nonzero_overlap_raises(self) -> None:
        """FixedSizeStrategy implements no overlap; a non-zero value must be rejected,
        not silently ignored."""
        with pytest.raises(ValueError, match="overlap"):
            FixedSizeStrategy(chunk_overlap=1)


class TestStrategyRegistry:
    def test_get_all_strategies_returns_six(self) -> None:
        strategies = get_all_strategies()
        assert len(strategies) == 6

    def test_get_all_strategy_names_match_default_strategies(self) -> None:
        strategies = get_all_strategies()
        names = {s.name for s in strategies}
        assert names == set(DEFAULT_STRATEGIES)
        assert names == {
            StrategyName.LEXICHUNK,
            StrategyName.LEXICHUNK_CONTEXTUAL,
            StrategyName.RCTS_512,
            StrategyName.RCTS_1024,
            StrategyName.SENTENCE_SPLIT,
            StrategyName.FIXED_SIZE,
        }

    def test_get_strategy_by_name(self) -> None:
        strategy = get_strategy(StrategyName.LEXICHUNK)
        assert strategy.name == StrategyName.LEXICHUNK

    def test_get_contextual_strategy_by_name(self) -> None:
        strategy = get_strategy(StrategyName.LEXICHUNK_CONTEXTUAL)
        assert strategy.name == StrategyName.LEXICHUNK_CONTEXTUAL

    def test_strategy_params_reports_parameters_that_ran(self) -> None:
        assert strategy_params(StrategyName.RCTS_512) == {
            "chunk_size": 512,
            "chunk_overlap": 50,
        }
        assert strategy_params(StrategyName.RCTS_1024) == {
            "chunk_size": 1024,
            "chunk_overlap": 100,
        }
        assert strategy_params(StrategyName.FIXED_SIZE) == {"chunk_size": 512}
        assert strategy_params(StrategyName.SENTENCE_SPLIT) == {"min_chunk_chars": 100}

    def test_strategy_params_unknown_strategy_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown strategy"):
            strategy_params(StrategyName.RCTS)


class TestChunkingPipeline:
    def test_run_returns_one_result_per_strategy(self) -> None:
        strategies = get_all_strategies()
        pipeline = ChunkingPipeline(strategies)
        results = pipeline.run([TINY_DOC])
        assert len(results) == 6

    def test_each_result_has_correct_strategy(self) -> None:
        strategies = get_all_strategies()
        pipeline = ChunkingPipeline(strategies)
        results = pipeline.run([TINY_DOC])
        result_names = {r.strategy for r in results}
        assert StrategyName.LEXICHUNK in result_names
        assert StrategyName.RCTS_512 in result_names
        assert StrategyName.RCTS_1024 in result_names

    def test_run_single(self) -> None:
        strategies = get_all_strategies()
        pipeline = ChunkingPipeline(strategies)
        result = pipeline.run_single(StrategyName.RCTS_512, [TINY_DOC])
        assert result.strategy == StrategyName.RCTS_512
        assert len(result.chunk_sets) == 1

    def test_run_single_contextual(self) -> None:
        strategies = get_all_strategies()
        pipeline = ChunkingPipeline(strategies)
        result = pipeline.run_single(StrategyName.LEXICHUNK_CONTEXTUAL, [TINY_DOC])
        assert result.strategy == StrategyName.LEXICHUNK_CONTEXTUAL
        assert len(result.chunk_sets) == 1

    def test_empty_strategies_raises(self) -> None:
        with pytest.raises(ValueError, match="At least one"):
            ChunkingPipeline([])

    def test_integration_with_fixtures(self) -> None:
        """Run all strategies on all real fixture documents."""
        fm = FixtureManager()
        docs = fm.load_all()
        strategies = get_all_strategies()
        pipeline = ChunkingPipeline(strategies)
        results = pipeline.run(docs)

        assert len(results) == 6
        for result in results:
            assert len(result.chunk_sets) == 5
            for cs in result.chunk_sets:
                assert cs.count > 0
                assert cs.elapsed_seconds > 0
