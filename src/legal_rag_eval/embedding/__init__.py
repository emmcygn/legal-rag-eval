"""Embedding pipeline and model adapters."""

from legal_rag_eval.embedding.pipeline import (
    EmbeddingCache,
    EmbeddingPipeline,
    SentenceTransformerAdapter,
)

__all__ = [
    "EmbeddingCache",
    "EmbeddingPipeline",
    "SentenceTransformerAdapter",
]
