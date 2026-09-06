"""Retrieval simulation: query execution and relevance matching.

Relevance itself is not defined here. :func:`RetrievalSimulator.run` uses
:func:`scaffolder.metrics.retrieval.is_relevant` — the same span-overlap
definition every metric in :mod:`scaffolder.metrics.retrieval` uses — to fill in
``RetrievalResult.relevant_retrieved``. A prior version of this module carried
its own near-duplicate copy of the relevance test, so the count printed on a
``RetrievalResult`` and the metrics computed from it could quietly diverge; see
``tests/test_simulator.py`` for a regression test pinning the two together.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from scaffolder.metrics.retrieval import (
    DEFAULT_MIN_OVERLAP_CHARS,
    is_relevant,
    valid_relevant_sections,
)
from scaffolder.models import (
    RetrievalResult,
    StrategyName,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from scaffolder.embedding.pipeline import EmbeddingPipeline
    from scaffolder.models import AnnotatedQuery, EmbeddingModelName
    from scaffolder.retrieval.index import IndexRegistry

logger = logging.getLogger(__name__)


def load_queries_from_yaml(
    queries_dir: str = "queries",
    gold_dir: str = "gold",
) -> list[AnnotatedQuery]:
    """Load query YAML files and resolve them to gold clause spans.

    Thin wrapper: parses the YAML with :func:`scaffolder.queries.load_queries`,
    loads every gold annotation with :func:`scaffolder.gold.load_all_gold`, and
    resolves the two together with :func:`scaffolder.queries.resolve_queries`.
    Raises ``scaffolder.gold.GoldError`` if a gold file is missing or invalid, or
    ``scaffolder.queries.QueryError`` if a query names an identifier that does
    not exist in its document's gold annotation.
    """
    from scaffolder.gold import load_all_gold
    from scaffolder.queries import load_queries, resolve_queries

    raw_queries = load_queries(queries_dir)
    gold_by_document = load_all_gold(gold_dir)
    return resolve_queries(raw_queries, gold_by_document)


class RetrievalSimulator:
    """Runs queries against FAISS indices and produces RetrievalResults.

    For each query, the simulator:
    1. Embeds the query text using the specified model.
    2. Searches the (strategy, model) index for top-k results.
    3. Matches retrieved chunks against annotated relevant sections.
    4. Produces a RetrievalResult with matched relevance info.
    """

    def __init__(
        self,
        index_registry: IndexRegistry,
        embedding_pipeline: EmbeddingPipeline,
    ) -> None:
        self._registry = index_registry
        self._embedding = embedding_pipeline

    def run(
        self,
        queries: Sequence[AnnotatedQuery],
        strategies: Sequence[StrategyName],
        models: Sequence[EmbeddingModelName],
        k: int = 10,
        min_overlap_chars: int = DEFAULT_MIN_OVERLAP_CHARS,
    ) -> list[RetrievalResult]:
        """Run all queries against all (strategy, model) indices.

        Returns one RetrievalResult per (query, strategy, model) combination.
        ``min_overlap_chars`` is forwarded to
        :func:`scaffolder.metrics.retrieval.is_relevant` and must match whatever
        value :func:`scaffolder.metrics.retrieval.compute_retrieval_metrics` is
        later called with, or ``relevant_retrieved`` and the computed metrics
        will disagree.
        """
        results: list[RetrievalResult] = []

        for model in models:
            query_texts = [q.text for q in queries]
            query_embeddings = self._embedding.embed_texts(query_texts, model)

            for strategy in strategies:
                try:
                    index = self._registry.get(strategy, model)
                except KeyError:
                    logger.warning(
                        "No index for %s/%s, skipping.",
                        strategy.value,
                        model.value,
                    )
                    continue

                for i, query in enumerate(queries):
                    query_vec = query_embeddings[i]
                    hits = index.search(query_vec, k=k)
                    relevant_sections = valid_relevant_sections(query.relevant_sections)

                    relevant_retrieved = sum(
                        1
                        for h in hits
                        if is_relevant(h.chunk, query.relevant_sections, min_overlap_chars)
                    )

                    results.append(
                        RetrievalResult(
                            query=query,
                            strategy=strategy,
                            embedding_model=model,
                            hits=tuple(hits),
                            relevant_retrieved=relevant_retrieved,
                            total_relevant=len(relevant_sections),
                        )
                    )

        logger.info(
            "Simulation complete: %d results (%d queries x %d strategies x %d models)",
            len(results),
            len(queries),
            len(strategies),
            len(models),
        )
        return results
