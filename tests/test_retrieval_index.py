"""Tests for VectorIndex and IndexRegistry."""

from __future__ import annotations

import numpy as np
import pytest

from legal_rag_eval.models import (
    Chunk,
    EmbeddingModelName,
    StrategyName,
)
from legal_rag_eval.retrieval import IndexRegistry, VectorIndex


def _random_embeddings(n: int, dim: int) -> np.ndarray:
    """Create normalized random vectors."""
    rng = np.random.default_rng(42)
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


def _make_chunks(n: int, strategy: StrategyName = StrategyName.LEXICHUNK) -> list[Chunk]:
    return [
        Chunk(
            id=f"{strategy.value}_doc_{i}",
            text=f"Chunk text number {i}",
            document_id="doc",
            strategy=strategy,
            index=i,
        )
        for i in range(n)
    ]


DIM = 64


class TestVectorIndex:
    def test_add_and_size(self) -> None:
        index = VectorIndex(dimension=DIM)
        chunks = _make_chunks(5)
        embs = _random_embeddings(5, DIM)
        index.add(chunks, embs)
        assert index.size == 5

    def test_search_returns_k_results(self) -> None:
        index = VectorIndex(dimension=DIM)
        chunks = _make_chunks(10)
        embs = _random_embeddings(10, DIM)
        index.add(chunks, embs)

        query = _random_embeddings(1, DIM)[0]
        hits = index.search(query, k=5)
        assert len(hits) == 5

    def test_search_sorted_by_descending_score(self) -> None:
        index = VectorIndex(dimension=DIM)
        chunks = _make_chunks(10)
        embs = _random_embeddings(10, DIM)
        index.add(chunks, embs)

        query = _random_embeddings(1, DIM)[0]
        hits = index.search(query, k=5)
        scores = [h.score for h in hits]
        assert scores == sorted(scores, reverse=True)

    def test_search_k_larger_than_index(self) -> None:
        index = VectorIndex(dimension=DIM)
        chunks = _make_chunks(3)
        embs = _random_embeddings(3, DIM)
        index.add(chunks, embs)

        query = _random_embeddings(1, DIM)[0]
        hits = index.search(query, k=10)
        assert len(hits) == 3

    def test_search_returns_itself_as_top1(self) -> None:
        index = VectorIndex(dimension=DIM)
        chunks = _make_chunks(5)
        embs = _random_embeddings(5, DIM)
        index.add(chunks, embs)

        # Search for the first vector — should be top result
        hits = index.search(embs[0], k=1)
        assert len(hits) == 1
        assert hits[0].chunk.id == chunks[0].id
        assert hits[0].score > 0.99  # cosine sim ~= 1.0

    def test_empty_index_returns_empty(self) -> None:
        index = VectorIndex(dimension=DIM)
        query = _random_embeddings(1, DIM)[0]
        assert index.search(query, k=5) == []

    def test_reset_clears_index(self) -> None:
        index = VectorIndex(dimension=DIM)
        chunks = _make_chunks(5)
        embs = _random_embeddings(5, DIM)
        index.add(chunks, embs)
        index.reset()
        assert index.size == 0

    def test_mismatched_lengths_raises(self) -> None:
        index = VectorIndex(dimension=DIM)
        chunks = _make_chunks(3)
        embs = _random_embeddings(5, DIM)
        with pytest.raises(ValueError, match="length mismatch"):
            index.add(chunks, embs)

    def test_mismatched_dimensions_raises(self) -> None:
        index = VectorIndex(dimension=DIM)
        chunks = _make_chunks(5)
        embs = _random_embeddings(5, DIM * 2)
        with pytest.raises(ValueError, match="dimension"):
            index.add(chunks, embs)

    def test_ranks_are_1_indexed(self) -> None:
        index = VectorIndex(dimension=DIM)
        chunks = _make_chunks(5)
        embs = _random_embeddings(5, DIM)
        index.add(chunks, embs)
        hits = index.search(embs[0], k=3)
        assert [h.rank for h in hits] == [1, 2, 3]


class TestIndexRegistry:
    def test_build_and_get(self) -> None:
        registry = IndexRegistry()
        chunks = _make_chunks(5)
        embs = _random_embeddings(5, DIM)
        registry.build(StrategyName.LEXICHUNK, EmbeddingModelName.MINILM, chunks, embs)

        index = registry.get(StrategyName.LEXICHUNK, EmbeddingModelName.MINILM)
        assert index.size == 5

    def test_get_unknown_raises(self) -> None:
        registry = IndexRegistry()
        with pytest.raises(KeyError, match="No index"):
            registry.get(StrategyName.RCTS, EmbeddingModelName.MINILM)

    def test_count(self) -> None:
        registry = IndexRegistry()
        chunks = _make_chunks(5)
        embs = _random_embeddings(5, DIM)
        registry.build(StrategyName.LEXICHUNK, EmbeddingModelName.MINILM, chunks, embs)
        registry.build(StrategyName.RCTS, EmbeddingModelName.MINILM, chunks, embs)
        assert registry.count == 2

    def test_keys(self) -> None:
        registry = IndexRegistry()
        chunks = _make_chunks(5)
        embs = _random_embeddings(5, DIM)
        registry.build(StrategyName.LEXICHUNK, EmbeddingModelName.MINILM, chunks, embs)
        keys = registry.keys()
        assert len(keys) == 1
        assert keys[0].strategy == StrategyName.LEXICHUNK
        assert keys[0].model == EmbeddingModelName.MINILM
