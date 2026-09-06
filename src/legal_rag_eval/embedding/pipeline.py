"""Embedding pipeline with pluggable adapters and disk cache."""

from __future__ import annotations

import hashlib
import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from legal_rag_eval.models import EmbeddingModelName

if TYPE_CHECKING:
    from collections.abc import Sequence

    from legal_rag_eval.models import Chunk

logger = logging.getLogger(__name__)

# Model name -> HuggingFace model ID mapping
_MODEL_IDS: dict[EmbeddingModelName, str] = {
    EmbeddingModelName.MINILM: "sentence-transformers/all-MiniLM-L6-v2",
    EmbeddingModelName.BGE_BASE: "BAAI/bge-base-en-v1.5",
}

# Known dimensions (avoids loading model just to check)
_MODEL_DIMS: dict[EmbeddingModelName, int] = {
    EmbeddingModelName.MINILM: 384,
    EmbeddingModelName.BGE_BASE: 768,
    EmbeddingModelName.VOYAGE_LAW_2: 1024,
}


class SentenceTransformerAdapter:
    """Local embedding adapter using sentence-transformers.

    Lazily loads the model on first embed_texts() call. Supports
    all-MiniLM-L6-v2 (384d) and bge-base-en-v1.5 (768d).
    """

    def __init__(self, model_name: EmbeddingModelName) -> None:
        if model_name not in _MODEL_IDS:
            msg = (
                f"SentenceTransformerAdapter does not support {model_name}. "
                f"Use one of: {list(_MODEL_IDS.keys())}"
            )
            raise ValueError(msg)
        self.model_name = model_name
        self.dimension = _MODEL_DIMS[model_name]
        self._model_id = _MODEL_IDS[model_name]
        self._model: Any = None  # noqa: UP037

    def _load_model(self) -> Any:  # noqa: UP037
        """Load the sentence-transformers model (slow, cached after first call)."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading model: %s", self._model_id)
            self._model = SentenceTransformer(self._model_id)
        return self._model

    def embed_texts(self, texts: Sequence[str]) -> npt.NDArray[np.float32]:
        """Embed texts, returning (n, dimension) float32 array."""
        model = self._load_model()
        texts_list = list(texts)

        # BGE models recommend a prefix for better performance
        if self.model_name == EmbeddingModelName.BGE_BASE:
            texts_list = [
                t if t.startswith("Represent") else f"Represent this sentence: {t}"
                for t in texts_list
            ]

        embeddings = model.encode(
            texts_list,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=64,
        )
        return np.asarray(embeddings, dtype=np.float32)


class EmbeddingCache:
    """Disk-based embedding cache using .npy files.

    Cache key: hash(model_name + text) -> .npy file.
    Cache directory: .cache/embeddings/{model_name}/
    """

    def __init__(self, cache_dir: Path | None = None) -> None:
        self._cache_dir = cache_dir or Path(".cache/embeddings")
        self._hits = 0
        self._misses = 0

    def _key(self, model_name: str, text: str) -> str:
        content = f"{model_name}:{text}"
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

    def _path(self, model_name: str, key: str) -> Path:
        model_dir = self._cache_dir / model_name.replace("/", "_")
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir / f"{key}.npy"

    def get(self, model_name: str, text: str) -> npt.NDArray[np.float32] | None:
        """Look up a cached embedding. Returns None on miss."""
        key = self._key(model_name, text)
        path = self._path(model_name, key)
        if path.exists():
            self._hits += 1
            loaded: npt.NDArray[np.float32] = np.load(path)
            return loaded
        self._misses += 1
        return None

    def put(
        self,
        model_name: str,
        text: str,
        embedding: npt.NDArray[np.float32],
    ) -> None:
        """Store an embedding in the cache."""
        key = self._key(model_name, text)
        path = self._path(model_name, key)
        np.save(path, embedding)

    def get_batch(
        self, model_name: str, texts: Sequence[str]
    ) -> tuple[npt.NDArray[np.float32] | None, list[int]]:
        """Look up cached embeddings for a batch of texts.

        Returns:
            (cached_embeddings_or_None, list_of_uncached_indices)
        """
        dim: int | None = None
        cached: dict[int, npt.NDArray[np.float32]] = {}
        uncached: list[int] = []

        for i, text in enumerate(texts):
            emb = self.get(model_name, text)
            if emb is not None:
                cached[i] = emb
                if dim is None:
                    dim = emb.shape[0]
            else:
                uncached.append(i)

        if not cached:
            return None, uncached

        if dim is None:
            return None, uncached

        result = np.zeros((len(texts), dim), dtype=np.float32)
        for i, emb in cached.items():
            result[i] = emb

        return result, uncached

    @property
    def stats(self) -> dict[str, int]:
        """Return cache hit/miss statistics."""
        return {"hits": self._hits, "misses": self._misses}

    def clear(self) -> None:
        """Delete all cached embeddings."""
        if self._cache_dir.exists():
            shutil.rmtree(self._cache_dir)
        self._hits = 0
        self._misses = 0


class _VoyageAdapterWrapper:
    """Wraps Agent B's VoyageEmbedder (embed -> list[list[float]]) to match our interface."""

    def __init__(self, voyage: Any) -> None:  # noqa: UP037
        self._voyage = voyage

    @property
    def model_name(self) -> EmbeddingModelName:
        return EmbeddingModelName.VOYAGE_LAW_2

    @property
    def dimension(self) -> int:
        return _MODEL_DIMS[EmbeddingModelName.VOYAGE_LAW_2]

    def embed_texts(self, texts: Sequence[str]) -> npt.NDArray[np.float32]:
        result = self._voyage.embed(list(texts))
        return np.asarray(result, dtype=np.float32)


class EmbeddingPipeline:
    """Orchestrates embedding chunks using multiple models with caching.

    Usage:
        pipeline = EmbeddingPipeline(
            models=[EmbeddingModelName.MINILM, EmbeddingModelName.BGE_BASE],
        )
        embeddings = pipeline.embed_chunks(chunks, model_name)
    """

    def __init__(
        self,
        models: Sequence[EmbeddingModelName] | None = None,
        cache_dir: Path | None = None,
        use_cache: bool = True,
    ) -> None:
        self._adapters: dict[EmbeddingModelName, Any] = {}  # noqa: UP037
        self._cache = EmbeddingCache(cache_dir) if use_cache else None
        self._models = list(models or [EmbeddingModelName.MINILM])

    def _get_adapter(self, model_name: EmbeddingModelName) -> Any:  # noqa: UP037
        """Get or create an adapter for the given model."""
        if model_name not in self._adapters:
            if model_name in (
                EmbeddingModelName.MINILM,
                EmbeddingModelName.BGE_BASE,
            ):
                self._adapters[model_name] = SentenceTransformerAdapter(model_name)
            elif model_name == EmbeddingModelName.VOYAGE_LAW_2:
                from legal_rag_eval.embedding.voyage import VoyageEmbedder

                self._adapters[model_name] = _VoyageAdapterWrapper(VoyageEmbedder())
            else:
                msg = f"No adapter for model: {model_name}"
                raise ValueError(msg)
        return self._adapters[model_name]

    def embed_texts(
        self, texts: Sequence[str], model_name: EmbeddingModelName
    ) -> npt.NDArray[np.float32]:
        """Embed a list of texts using the specified model, with caching."""
        adapter = self._get_adapter(model_name)
        texts_list = list(texts)

        if self._cache is None:
            emb_result: npt.NDArray[np.float32] = adapter.embed_texts(texts_list)
            return emb_result

        # Check cache for batch
        cached_result, uncached_indices = self._cache.get_batch(model_name.value, texts_list)

        if not uncached_indices:
            assert cached_result is not None
            logger.info(
                "All %d embeddings served from cache (%s)",
                len(texts_list),
                model_name.value,
            )
            return cached_result

        # Embed uncached texts
        uncached_texts = [texts_list[i] for i in uncached_indices]
        logger.info(
            "Embedding %d texts (%d cached, %d new) with %s",
            len(texts_list),
            len(texts_list) - len(uncached_indices),
            len(uncached_indices),
            model_name.value,
        )
        new_embeddings: npt.NDArray[np.float32] = adapter.embed_texts(uncached_texts)

        # Store in cache
        for j, idx in enumerate(uncached_indices):
            self._cache.put(model_name.value, texts_list[idx], new_embeddings[j])

        # Merge cached + new
        if cached_result is not None:
            for j, idx in enumerate(uncached_indices):
                cached_result[idx] = new_embeddings[j]
            return cached_result

        # Nothing was cached, build fresh
        dim = _MODEL_DIMS.get(model_name, new_embeddings.shape[1])
        result = np.zeros((len(texts_list), dim), dtype=np.float32)
        for j, idx in enumerate(uncached_indices):
            result[idx] = new_embeddings[j]
        return result

    def embed_chunks(
        self, chunks: Sequence[Chunk], model_name: EmbeddingModelName
    ) -> npt.NDArray[np.float32]:
        """Convenience: embed chunk texts."""
        return self.embed_texts([c.text for c in chunks], model_name)

    @property
    def cache_stats(self) -> dict[str, int] | None:
        """Return cache statistics, or None if caching is disabled."""
        if self._cache:
            return self._cache.stats
        return None
