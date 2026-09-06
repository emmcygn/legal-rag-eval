"""FAISS-based vector index for similarity search."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import faiss
import numpy as np
import numpy.typing as npt

from legal_rag_eval.models import (
    EmbeddingModelName,
    RetrievalHit,
    StrategyName,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from legal_rag_eval.embedding.pipeline import EmbeddingPipeline
    from legal_rag_eval.models import Chunk, StrategyResult

logger = logging.getLogger(__name__)


class VectorIndex:
    """FAISS flat inner-product index wrapping chunk embeddings.

    Uses IndexFlatIP since embeddings are L2-normalized,
    so inner product equals cosine similarity.
    """

    def __init__(self, dimension: int) -> None:
        self._dimension = dimension
        self._index = faiss.IndexFlatIP(dimension)
        self._chunks: list[Chunk] = []

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def size(self) -> int:
        """Number of vectors in the index."""
        return int(self._index.ntotal)

    def add(
        self,
        chunks: Sequence[Chunk],
        embeddings: npt.NDArray[np.float32],
    ) -> None:
        """Add chunks and their embeddings to the index."""
        if len(chunks) != embeddings.shape[0]:
            msg = f"Chunks ({len(chunks)}) and embeddings ({embeddings.shape[0]}) length mismatch."
            raise ValueError(msg)
        if embeddings.shape[1] != self._dimension:
            msg = (
                f"Embedding dimension ({embeddings.shape[1]}) != "
                f"index dimension ({self._dimension})."
            )
            raise ValueError(msg)

        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        self._chunks.extend(chunks)
        self._index.add(embeddings)
        logger.debug("Added %d vectors to index (total: %d)", len(chunks), self.size)

    def search(
        self,
        query_vector: npt.NDArray[np.float32],
        k: int = 10,
    ) -> list[RetrievalHit]:
        """Search for the k most similar chunks to the query vector."""
        if self.size == 0:
            return []

        qv = query_vector.reshape(1, -1).astype(np.float32)
        qv = np.ascontiguousarray(qv)
        actual_k = min(k, self.size)

        scores, indices = self._index.search(qv, actual_k)

        hits: list[RetrievalHit] = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0], strict=True)):
            if idx < 0:
                continue
            hits.append(
                RetrievalHit(
                    chunk=self._chunks[int(idx)],
                    score=float(score),
                    rank=rank + 1,
                )
            )

        return hits

    def reset(self) -> None:
        """Clear the index and all stored chunks."""
        self._index.reset()
        self._chunks.clear()


@dataclass(frozen=True, slots=True)
class IndexKey:
    """Key for looking up a VectorIndex."""

    strategy: StrategyName
    model: EmbeddingModelName


class IndexRegistry:
    """Registry of VectorIndex instances keyed by (strategy, model)."""

    def __init__(self) -> None:
        self._indices: dict[IndexKey, VectorIndex] = {}

    def build(
        self,
        strategy: StrategyName,
        model: EmbeddingModelName,
        chunks: Sequence[Chunk],
        embeddings: npt.NDArray[np.float32],
        dimension: int | None = None,
    ) -> VectorIndex:
        """Build and register a new index."""
        dim = dimension or embeddings.shape[1]
        key = IndexKey(strategy=strategy, model=model)

        index = VectorIndex(dimension=dim)
        index.add(chunks, embeddings)
        self._indices[key] = index

        logger.info(
            "Built index for %s/%s: %d vectors, %d dims",
            strategy.value,
            model.value,
            index.size,
            dim,
        )
        return index

    def get(self, strategy: StrategyName, model: EmbeddingModelName) -> VectorIndex:
        """Get a previously built index.

        Raises:
            KeyError: If no index exists for this (strategy, model) pair.
        """
        key = IndexKey(strategy=strategy, model=model)
        if key not in self._indices:
            available = [(k.strategy.value, k.model.value) for k in self._indices]
            msg = f"No index for ({strategy.value}, {model.value}). Available: {available}"
            raise KeyError(msg)
        return self._indices[key]

    def keys(self) -> list[IndexKey]:
        """Return all registered (strategy, model) keys."""
        return list(self._indices.keys())

    @property
    def count(self) -> int:
        """Number of registered indices."""
        return len(self._indices)


def build_all_indices(
    strategy_results: list[StrategyResult],
    embedding_pipeline: EmbeddingPipeline,
    models: Sequence[EmbeddingModelName],
) -> IndexRegistry:
    """Build FAISS indices for every (strategy, model) pair.

    Concatenates all chunks from all documents into a single index
    per (strategy, model) for cross-document retrieval / DRM testing.
    """
    registry = IndexRegistry()

    for sr in strategy_results:
        all_chunks: list[Chunk] = []
        for cs in sr.chunk_sets:
            all_chunks.extend(cs.chunks)

        if not all_chunks:
            logger.warning("No chunks for strategy %s, skipping.", sr.strategy.value)
            continue

        for model in models:
            embeddings = embedding_pipeline.embed_chunks(all_chunks, model)
            registry.build(sr.strategy, model, all_chunks, embeddings)

    return registry
