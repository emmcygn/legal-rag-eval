"""End-to-end integration tests for the full benchmark pipeline."""

from __future__ import annotations

from scaffolder.chunking import ChunkingPipeline, get_all_strategies
from scaffolder.fixtures import FixtureManager
from scaffolder.metrics.structural import compute_legacy_structural_metrics
from scaffolder.models import StrategyName

# Strategies that emit verbatim substrings of the document: every chunk should be
# exactly locatable by scaffolder.gold.locate_chunks.
_EXACT_LOCALIZATION_STRATEGIES = (
    StrategyName.RCTS_512,
    StrategyName.RCTS_1024,
    StrategyName.FIXED_SIZE,
)


class TestFullStructuralPipeline:
    def test_fixtures_to_metrics(self) -> None:
        """Full pipeline: fixtures -> chunking -> structural metrics."""
        fm = FixtureManager()
        docs = fm.load_all()
        assert len(docs) == 5

        strategies = get_all_strategies()
        pipeline = ChunkingPipeline(strategies)
        results = pipeline.run(docs)

        assert len(results) == 6  # 6 strategies

        for sr in results:
            assert len(sr.chunk_sets) == 5
            for cs in sr.chunk_sets:
                assert cs.count > 0
                doc = fm.get_by_id(cs.document_id)
                sm = compute_legacy_structural_metrics(cs, doc)
                assert 0.0 <= sm.clause_fragmentation_rate <= 1.0
                assert 0.0 <= sm.definition_preservation_rate <= 1.0
                assert 0.0 <= sm.cross_ref_resolution_rate <= 1.0
                assert 0.0 <= sm.hierarchy_depth_retained <= 1.0

    def test_all_strategies_produce_unique_chunk_ids(self) -> None:
        """No duplicate chunk IDs within a strategy run on a document."""
        fm = FixtureManager()
        docs = fm.load_all()
        strategies = get_all_strategies()

        for strategy in strategies:
            for doc in docs:
                cs = strategy.chunk(doc)
                ids = [c.id for c in cs.chunks]
                assert len(ids) == len(set(ids)), (
                    f"Duplicate IDs in {strategy.name.value} for {doc.id}"
                )

    def test_no_empty_chunks(self) -> None:
        """No strategy produces empty-text chunks on real documents."""
        fm = FixtureManager()
        docs = fm.load_all()
        strategies = get_all_strategies()

        for strategy in strategies:
            for doc in docs:
                cs = strategy.chunk(doc)
                for chunk in cs.chunks:
                    assert len(chunk.text.strip()) > 0, (
                        f"Empty chunk in {strategy.name.value} for {doc.id}"
                    )

    def test_baseline_chunks_are_fully_located(self) -> None:
        """Baselines emit verbatim substrings of the sanitised document, so
        ChunkingPipeline.run's attach_spans (scaffolder.chunking.pipeline) must locate
        every one of their chunks: char_start/char_end are never None for rcts_512,
        rcts_1024, or fixed_size on the real fixtures.
        """
        fm = FixtureManager()
        docs = fm.load_all()
        strategies = get_all_strategies()
        pipeline = ChunkingPipeline(strategies)
        results = pipeline.run(docs)

        for sr in results:
            if sr.strategy not in _EXACT_LOCALIZATION_STRATEGIES:
                continue
            for cs in sr.chunk_sets:
                unlocated = [c for c in cs.chunks if not c.located]
                assert not unlocated, (
                    f"{sr.strategy.value}/{cs.document_id}: "
                    f"{len(unlocated)} of {cs.count} chunks not located"
                )

    def test_lexichunk_chunks_are_mostly_located(self) -> None:
        """LexiChunk may prepend a synthesised heading breadcrumb that is not verbatim
        in the source, so it is held only to "some chunks located", not exact -- unlike
        the baselines above.
        """
        fm = FixtureManager()
        docs = fm.load_all()
        strategies = get_all_strategies()
        pipeline = ChunkingPipeline(strategies)
        results = pipeline.run(docs)

        for sr in results:
            if sr.strategy is not StrategyName.LEXICHUNK:
                continue
            total = sum(cs.count for cs in sr.chunk_sets)
            located = sum(sum(1 for c in cs.chunks if c.located) for cs in sr.chunk_sets)
            assert total > 0
            assert located / total > 0.0
