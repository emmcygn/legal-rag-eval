"""Retrieval simulation: query execution and relevance matching."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from scaffolder.models import (
    EmbeddingModelName,
    RelevanceGrade,
    RetrievalResult,
    StrategyName,
)
from scaffolder.relevance import matching_sections, valid_relevant_sections

if TYPE_CHECKING:
    from collections.abc import Sequence

    from scaffolder.embedding.pipeline import EmbeddingPipeline
    from scaffolder.models import AnnotatedQuery, Chunk, RelevantSection
    from scaffolder.retrieval.index import IndexRegistry

logger = logging.getLogger(__name__)


def load_queries_from_yaml(queries_dir: str = "queries") -> list[AnnotatedQuery]:
    """Load query annotations from YAML files using Agent B's loader.

    Bridges Agent B's AnnotatedQuery format (queries.py) to models.py format.
    """
    from pathlib import Path

    from scaffolder.models import (
        AnnotatedQuery as ModelQuery,
    )
    from scaffolder.models import (
        Jurisdiction,
        RelevanceGrade,
    )
    from scaffolder.models import (
        RelevantSection as ModelSection,
    )
    from scaffolder.queries import load_queries

    raw_queries = load_queries(Path(queries_dir))

    model_queries: list[ModelQuery] = []
    for rq in raw_queries:
        # Infer jurisdiction from document_id prefix
        doc_id = rq.document_id
        if doc_id.startswith("uk_"):
            jurisdiction = Jurisdiction.UK
        elif doc_id.startswith("us_"):
            jurisdiction = Jurisdiction.US
        elif doc_id.startswith("eu_"):
            jurisdiction = Jurisdiction.EU
        else:
            jurisdiction = Jurisdiction.UK

        sections = tuple(
            ModelSection(
                document_id=rq.document_id,
                section_id=s.section_id,
                text_snippet=s.description[:200] if s.description else "",
                grade=RelevanceGrade(s.relevance),
            )
            for s in rq.relevant_sections
        )

        model_queries.append(
            ModelQuery(
                id=rq.id,
                text=rq.text,
                document_ids=[rq.document_id],
                jurisdiction=jurisdiction,
                relevant_sections=sections,
                category=rq.failure_mode,
            )
        )

    return model_queries


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
    ) -> list[RetrievalResult]:
        """Run all queries against all (strategy, model) indices.

        Returns one RetrievalResult per (query, strategy, model) combination.
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
                        1 for h in hits if _is_relevant(h.chunk, relevant_sections)
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


def _is_relevant(
    chunk: Chunk,
    relevant_sections: Sequence[RelevantSection],
) -> bool:
    """Check if a retrieved chunk matches any relevant section."""
    return bool(matching_sections(chunk, relevant_sections))


def get_relevance_grade(
    chunk: Chunk,
    relevant_sections: Sequence[RelevantSection],
) -> RelevanceGrade:
    """Get the relevance grade for a chunk (used in NDCG).

    Returns the highest matching grade, or IRRELEVANT if no match.
    """
    matched_sections = matching_sections(chunk, relevant_sections)
    return max(
        (section.grade for section in matched_sections),
        default=RelevanceGrade.IRRELEVANT,
        key=lambda grade: grade.value,
    )
