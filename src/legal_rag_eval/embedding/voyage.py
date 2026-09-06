"""Voyage AI embedding adapter with rate limiting and error handling."""

from __future__ import annotations

import logging
import os
import time
from typing import Any

logger = logging.getLogger(__name__)

# Voyage API limits (as of 2025)
_DEFAULT_BATCH_SIZE = 128  # Max texts per API call
_MIN_RETRY_DELAY = 1.0  # Seconds
_MAX_RETRY_DELAY = 60.0  # Seconds
_MAX_RETRIES = 3

# Model dimensions
_MODEL_DIMENSIONS: dict[str, int] = {
    "voyage-law-2": 1024,
    "voyage-3": 1024,
    "voyage-3-lite": 512,
}


class VoyageEmbedderError(Exception):
    """Raised when Voyage embedding fails."""


class VoyageEmbedder:
    """Embedding adapter using Voyage AI's API.

    Requires the VOYAGE_API_KEY environment variable to be set.
    Uses voyage-law-2 by default, which is optimized for legal text.

    Features:
    - Automatic batching (respects API batch size limits)
    - Exponential backoff on rate limit errors
    - Separate document vs query embedding (Voyage supports input_type)
    - Clear error messages for common issues
    """

    def __init__(
        self,
        model_name: str = "voyage-law-2",
        api_key: str | None = None,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        max_retries: int = _MAX_RETRIES,
    ) -> None:
        self._model_name = model_name
        self._batch_size = batch_size
        self._max_retries = max_retries

        # Resolve API key
        self._api_key = api_key or os.getenv("VOYAGE_API_KEY")
        if not self._api_key:
            raise VoyageEmbedderError(
                "Voyage API key not found. Set the VOYAGE_API_KEY environment variable "
                "or pass api_key to VoyageEmbedder. Get a key at https://dash.voyageai.com/"
            )

        # Resolve dimension
        if model_name not in _MODEL_DIMENSIONS:
            logger.warning(
                "Unknown Voyage model '%s'. Assuming dimension 1024. Known models: %s",
                model_name,
                list(_MODEL_DIMENSIONS.keys()),
            )
        self._dimension = _MODEL_DIMENSIONS.get(model_name, 1024)

        # Initialize client lazily
        self._client: Any = None

    def _get_client(self) -> Any:
        """Lazy-initialize the Voyage client."""
        if self._client is None:
            try:
                import voyageai
            except ImportError:
                raise VoyageEmbedderError(
                    "voyageai package not installed. "
                    "Install with: pip install 'legal-rag-eval[voyage]'"
                ) from None
            self._client = voyageai.Client(api_key=self._api_key)
        return self._client

    @property
    def model_name(self) -> str:
        """Return the Voyage model identifier."""
        return self._model_name

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self._dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts as documents.

        Automatically batches requests and handles rate limits.
        """
        return self._embed_batched(texts, input_type="document")

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query text.

        Uses input_type='query' which Voyage optimizes differently
        from document embeddings.
        """
        results = self._embed_batched([text], input_type="query")
        return results[0]

    def _embed_batched(self, texts: list[str], input_type: str) -> list[list[float]]:
        """Embed texts in batches with rate limiting."""
        if not texts:
            return []

        all_embeddings: list[list[float]] = []
        client = self._get_client()

        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            embeddings = self._embed_with_retry(client, batch, input_type)
            all_embeddings.extend(embeddings)

        return all_embeddings

    def _embed_with_retry(
        self,
        client: Any,
        texts: list[str],
        input_type: str,
    ) -> list[list[float]]:
        """Embed a single batch with exponential backoff retry."""
        delay = _MIN_RETRY_DELAY

        for attempt in range(self._max_retries + 1):
            try:
                result = client.embed(
                    texts,
                    model=self._model_name,
                    input_type=input_type,
                )
                return result.embeddings  # type: ignore[no-any-return]

            except Exception as e:
                error_str = str(e).lower()

                # Don't retry auth errors
                if "401" in error_str or "403" in error_str or "unauthorized" in error_str:
                    raise VoyageEmbedderError(
                        f"Voyage API authentication failed. Check your VOYAGE_API_KEY. Error: {e}"
                    ) from e

                # Don't retry bad requests
                if "400" in error_str or "bad request" in error_str:
                    raise VoyageEmbedderError(
                        f"Voyage API bad request. Check input texts. Error: {e}"
                    ) from e

                # Retry on rate limits and server errors
                if attempt < self._max_retries:
                    if "429" in error_str or "rate" in error_str:
                        logger.warning(
                            "Voyage rate limit hit. Retrying in %.1fs (attempt %d/%d)",
                            delay,
                            attempt + 1,
                            self._max_retries,
                        )
                    else:
                        logger.warning(
                            "Voyage API error: %s. Retrying in %.1fs (attempt %d/%d)",
                            e,
                            delay,
                            attempt + 1,
                            self._max_retries,
                        )
                    time.sleep(delay)
                    delay = min(delay * 2, _MAX_RETRY_DELAY)
                else:
                    raise VoyageEmbedderError(
                        f"Voyage API failed after {self._max_retries + 1} attempts. Last error: {e}"
                    ) from e

        # Should not reach here, but satisfy type checker
        raise VoyageEmbedderError("Unexpected retry loop exit")  # pragma: no cover
