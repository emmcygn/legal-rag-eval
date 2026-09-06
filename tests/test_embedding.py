"""Tests for EmbeddingPipeline and adapters."""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path
import pytest

from legal_rag_eval.embedding import (
    EmbeddingCache,
    EmbeddingPipeline,
    SentenceTransformerAdapter,
)
from legal_rag_eval.models import Chunk, EmbeddingModelName, StrategyName

# sentence-transformers is the ``[embeddings]`` extra, and these tests load a real model.
# The CI test job installs ``[dev]`` only, so they skip there; the retrieval benchmark job
# installs ``[embeddings]`` and runs them.
requires_sentence_transformers = pytest.mark.skipif(
    importlib.util.find_spec("sentence_transformers") is None,
    reason="sentence-transformers not installed (optional [embeddings] extra)",
)

SAMPLE_TEXTS = [
    "The Service Provider shall deliver services.",
    "Fees are due within 30 days of invoice date.",
    "This agreement is governed by the laws of England.",
]


class TestSentenceTransformerAdapter:
    @requires_sentence_transformers
    def test_minilm_shape(self) -> None:
        adapter = SentenceTransformerAdapter(EmbeddingModelName.MINILM)
        result = adapter.embed_texts(SAMPLE_TEXTS)
        assert result.shape == (3, 384)
        assert result.dtype == np.float32

    @requires_sentence_transformers
    def test_minilm_normalized(self) -> None:
        adapter = SentenceTransformerAdapter(EmbeddingModelName.MINILM)
        result = adapter.embed_texts(SAMPLE_TEXTS)
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_invalid_model_raises(self) -> None:
        with pytest.raises(ValueError, match="does not support"):
            SentenceTransformerAdapter(EmbeddingModelName.VOYAGE_LAW_2)


class TestEmbeddingCache:
    def test_put_then_get(self, tmp_path: Path) -> None:
        cache = EmbeddingCache(cache_dir=tmp_path / "cache")
        emb = np.random.randn(384).astype(np.float32)
        cache.put("test_model", "test text", emb)
        result = cache.get("test_model", "test text")
        assert result is not None
        np.testing.assert_array_equal(result, emb)

    def test_miss_returns_none(self, tmp_path: Path) -> None:
        cache = EmbeddingCache(cache_dir=tmp_path / "cache")
        assert cache.get("model", "nonexistent") is None

    def test_get_batch_mixed(self, tmp_path: Path) -> None:
        cache = EmbeddingCache(cache_dir=tmp_path / "cache")
        emb = np.random.randn(384).astype(np.float32)
        cache.put("model", "text_0", emb)
        # text_0 is cached, text_1 is not
        result, uncached = cache.get_batch("model", ["text_0", "text_1"])
        assert result is not None
        assert uncached == [1]

    def test_stats(self, tmp_path: Path) -> None:
        cache = EmbeddingCache(cache_dir=tmp_path / "cache")
        emb = np.random.randn(384).astype(np.float32)
        cache.put("model", "text", emb)
        cache.get("model", "text")  # hit
        cache.get("model", "miss")  # miss
        assert cache.stats == {"hits": 1, "misses": 1}

    def test_clear(self, tmp_path: Path) -> None:
        cache = EmbeddingCache(cache_dir=tmp_path / "cache")
        emb = np.random.randn(384).astype(np.float32)
        cache.put("model", "text", emb)
        cache.clear()
        assert cache.get("model", "text") is None


class TestEmbeddingPipeline:
    @requires_sentence_transformers
    def test_embed_texts_shape(self) -> None:
        pipeline = EmbeddingPipeline(use_cache=False)
        result = pipeline.embed_texts(SAMPLE_TEXTS, EmbeddingModelName.MINILM)
        assert result.shape == (3, 384)

    @requires_sentence_transformers
    def test_embed_with_cache(self, tmp_path: Path) -> None:
        pipeline = EmbeddingPipeline(cache_dir=tmp_path / "cache")
        # First call embeds
        result1 = pipeline.embed_texts(SAMPLE_TEXTS, EmbeddingModelName.MINILM)
        stats1 = pipeline.cache_stats
        assert stats1 is not None
        assert stats1["misses"] == 3

        # Second call should be from cache
        result2 = pipeline.embed_texts(SAMPLE_TEXTS, EmbeddingModelName.MINILM)
        stats2 = pipeline.cache_stats
        assert stats2 is not None
        assert stats2["hits"] == 3
        np.testing.assert_array_almost_equal(result1, result2)

    @requires_sentence_transformers
    def test_embed_chunks(self) -> None:
        chunks = [
            Chunk(
                id=f"test_{i}",
                text=t,
                document_id="doc",
                strategy=StrategyName.LEXICHUNK,
                index=i,
            )
            for i, t in enumerate(SAMPLE_TEXTS)
        ]
        pipeline = EmbeddingPipeline(use_cache=False)
        result = pipeline.embed_chunks(chunks, EmbeddingModelName.MINILM)
        assert result.shape == (3, 384)

    def test_voyage_without_key_raises(self) -> None:
        """Voyage model requires API key, so it raises without one."""
        from legal_rag_eval.embedding.voyage import VoyageEmbedderError

        pipeline = EmbeddingPipeline(use_cache=False)
        with pytest.raises(VoyageEmbedderError):
            pipeline.embed_texts(["test"], EmbeddingModelName.VOYAGE_LAW_2)
