"""End-to-end integration tests for the full benchmark pipeline."""

from __future__ import annotations

from scaffolder.chunking import ChunkingPipeline, get_all_strategies, get_strategy
from scaffolder.chunking.pipeline import attach_spans
from scaffolder.fixtures import FixtureManager
from scaffolder.gold import load_gold
from scaffolder.metrics.gold import compute_gold_structural_metrics
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


class TestUSDocumentGoldStructuralMetrics:
    """Regression test for the jurisdiction/normalisation bug that made
    ``heading_recall`` and ``xref_target_recall`` score exactly 0.0 for LexiChunk on
    both US fixtures (``us_msa``, ``us_terms_of_service``).

    Root causes (both fixed together, see ``scaffolder.chunking.strategies`` and
    ``scaffolder.metrics.gold.normalize_identifier``):

    1. ``LexiChunkStrategy``/``LexiChunkContextualStrategy`` built one ``LegalChunker``
       per strategy instance, shared across every fixture document, so it was always
       parsed under LegalChunker's ``jurisdiction="uk"`` default. A US contract's
       roman-numeral "Article VIII" headings are not recognised at all under the UK
       profile, so every chunk's ``section_hierarchy`` breadcrumb collapsed to just its
       own leaf line with no article/section ancestors -- ``heading_metrics`` could
       never compose a correct claim.
    2. Even once parsed under ``jurisdiction="us"``, a cross-reference's emitted
       ``target`` carried only the bare roman numeral (``"VIII"``), dropping
       ``target_kind`` (``"article"``) entirely, and ``normalize_identifier`` had no
       roman-to-arabic conversion -- so ``"VIII"`` normalised to ``"viii"``, which
       matches no gold identifier such as ``"article_8"``.

    This test runs the real LexiChunk build against the real ``us_msa`` fixture and
    its hand-checked gold annotations end to end, so a regression in either fix (or in
    a future LexiChunk parser change) fails here even if the synthetic unit tests in
    ``test_gold_metrics.py`` and ``test_chunking.py`` do not exercise it.
    """

    def test_us_msa_heading_and_xref_recall(self) -> None:
        fm = FixtureManager()
        doc = fm.get_by_id("us_msa")
        gold = load_gold("us_msa")

        strategy = get_strategy(StrategyName.LEXICHUNK)
        chunk_set = attach_spans(strategy.chunk(doc), doc)

        metrics = compute_gold_structural_metrics(chunk_set, doc, gold)

        assert metrics.heading_recall is not None
        assert metrics.heading_recall > 0.8
        assert metrics.xref_target_recall is not None
        assert metrics.xref_target_recall > 0.5
