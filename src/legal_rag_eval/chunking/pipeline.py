"""ChunkingPipeline: orchestrates running all strategies on all documents."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

from legal_rag_eval.gold import locate_chunks, sanitize
from legal_rag_eval.models import (
    ChunkingStrategy,
    ChunkSet,
    Document,
    StrategyName,
    StrategyResult,
)

logger = logging.getLogger(__name__)


def attach_spans(chunk_set: ChunkSet, document: Document) -> ChunkSet:
    """Return a copy of ``chunk_set`` whose chunks carry their span in the document.

    Localisation is identical for every strategy — the chunk text is matched against the
    sanitised document — so no strategy is scored on offsets it reported about itself.
    Chunks that are not a contiguous span of the source (for instance because the chunker
    prepended a synthesised heading) keep as much of their verbatim body as can be located;
    chunks that cannot be located at all get ``char_start is None`` and are counted in the
    localisation rate rather than silently scoring zero.
    """
    text = sanitize(document.text)
    spans = locate_chunks([c.text for c in chunk_set.chunks], text)
    located = tuple(
        chunk.with_span(None, None)
        if span is None
        else chunk.with_span(span.char_start, span.char_end)
        for chunk, span in zip(chunk_set.chunks, spans, strict=True)
    )
    return ChunkSet(
        strategy=chunk_set.strategy,
        document_id=chunk_set.document_id,
        chunks=located,
        elapsed_seconds=chunk_set.elapsed_seconds,
    )


class ChunkingPipeline:
    """Runs multiple chunking strategies across multiple documents.

    Usage:
        strategies = get_all_strategies()
        pipeline = ChunkingPipeline(strategies)
        results = pipeline.run(documents)
        # results: list[StrategyResult], one per strategy
    """

    def __init__(self, strategies: Sequence[ChunkingStrategy]) -> None:
        self._strategies = list(strategies)
        if not self._strategies:
            msg = "At least one strategy is required."
            raise ValueError(msg)

    def run(self, documents: Sequence[Document]) -> list[StrategyResult]:
        """Run all strategies on all documents.

        Returns:
            One StrategyResult per strategy, each containing ChunkSets
            for every document.
        """
        results: list[StrategyResult] = []

        for strategy in self._strategies:
            logger.info("Running strategy: %s", strategy.name.value)
            total_start = time.perf_counter()
            chunk_sets = []

            for doc in documents:
                logger.info(
                    "  Chunking %s (%d chars)...",
                    doc.id,
                    doc.char_count,
                )
                cs = attach_spans(strategy.chunk(doc), doc)
                chunk_sets.append(cs)
                logger.info(
                    "    -> %d chunks in %.3fs (avg %.0f chars/chunk)",
                    cs.count,
                    cs.elapsed_seconds,
                    cs.avg_chunk_size,
                )

            total_elapsed = time.perf_counter() - total_start
            results.append(
                StrategyResult(
                    strategy=strategy.name,
                    chunk_sets=tuple(chunk_sets),
                    total_elapsed_seconds=total_elapsed,
                )
            )

        return results

    def run_single(
        self, strategy_name: StrategyName, documents: Sequence[Document]
    ) -> StrategyResult:
        """Run a single named strategy on all documents."""
        for strategy in self._strategies:
            if strategy.name == strategy_name:
                total_start = time.perf_counter()
                chunk_sets = []
                for doc in documents:
                    chunk_sets.append(attach_spans(strategy.chunk(doc), doc))
                total_elapsed = time.perf_counter() - total_start
                return StrategyResult(
                    strategy=strategy.name,
                    chunk_sets=tuple(chunk_sets),
                    total_elapsed_seconds=total_elapsed,
                )
        msg = (
            f"Strategy '{strategy_name}' not found in pipeline. "
            f"Available: {[s.name for s in self._strategies]}"
        )
        raise ValueError(msg)
