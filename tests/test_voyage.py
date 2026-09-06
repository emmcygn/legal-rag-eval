"""Tests for Voyage AI embedding adapter."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from legal_rag_eval.embedding.voyage import VoyageEmbedder, VoyageEmbedderError


class TestVoyageEmbedderInit:
    """Test initialization and configuration."""

    def test_missing_api_key_raises(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("VOYAGE_API_KEY", None)
            with pytest.raises(VoyageEmbedderError, match="VOYAGE_API_KEY"):
                VoyageEmbedder()

    def test_explicit_api_key(self) -> None:
        embedder = VoyageEmbedder(api_key="test-key")
        assert embedder.model_name == "voyage-law-2"
        assert embedder.dimension == 1024

    def test_env_api_key(self) -> None:
        with patch.dict(os.environ, {"VOYAGE_API_KEY": "env-key"}):
            embedder = VoyageEmbedder()
            assert embedder.model_name == "voyage-law-2"

    def test_custom_model(self) -> None:
        embedder = VoyageEmbedder(model_name="voyage-3-lite", api_key="test-key")
        assert embedder.dimension == 512


class TestVoyageEmbedderEmbed:
    """Test embedding with mocked API calls."""

    def _make_embedder(self) -> VoyageEmbedder:
        return VoyageEmbedder(api_key="test-key", batch_size=2, max_retries=1)

    def test_embed_empty_list(self) -> None:
        embedder = self._make_embedder()
        result = embedder.embed([])
        assert result == []

    def test_embed_single_batch(self) -> None:
        embedder = self._make_embedder()
        mock_client = MagicMock()
        mock_client.embed.return_value = MagicMock(embeddings=[[0.1, 0.2], [0.3, 0.4]])
        embedder._client = mock_client

        result = embedder.embed(["text1", "text2"])
        assert len(result) == 2
        mock_client.embed.assert_called_once_with(
            ["text1", "text2"],
            model="voyage-law-2",
            input_type="document",
        )

    def test_embed_multiple_batches(self) -> None:
        embedder = self._make_embedder()
        mock_client = MagicMock()
        mock_client.embed.side_effect = [
            MagicMock(embeddings=[[0.1], [0.2]]),
            MagicMock(embeddings=[[0.3]]),
        ]
        embedder._client = mock_client

        result = embedder.embed(["t1", "t2", "t3"])
        assert len(result) == 3
        assert mock_client.embed.call_count == 2

    def test_embed_query_uses_query_type(self) -> None:
        embedder = self._make_embedder()
        mock_client = MagicMock()
        mock_client.embed.return_value = MagicMock(embeddings=[[0.1, 0.2]])
        embedder._client = mock_client

        result = embedder.embed_query("test query")
        assert result == [0.1, 0.2]
        mock_client.embed.assert_called_once_with(
            ["test query"],
            model="voyage-law-2",
            input_type="query",
        )

    def test_auth_error_not_retried(self) -> None:
        embedder = self._make_embedder()
        mock_client = MagicMock()
        mock_client.embed.side_effect = Exception("401 Unauthorized")
        embedder._client = mock_client

        with pytest.raises(VoyageEmbedderError, match="authentication failed"):
            embedder.embed(["test"])
        assert mock_client.embed.call_count == 1  # No retry

    def test_rate_limit_retried(self) -> None:
        embedder = self._make_embedder()
        mock_client = MagicMock()
        mock_client.embed.side_effect = [
            Exception("429 rate limit exceeded"),
            MagicMock(embeddings=[[0.1]]),
        ]
        embedder._client = mock_client

        with patch("legal_rag_eval.embedding.voyage.time.sleep"):
            result = embedder.embed(["test"])
        assert len(result) == 1
        assert mock_client.embed.call_count == 2

    def test_max_retries_exceeded(self) -> None:
        embedder = self._make_embedder()
        mock_client = MagicMock()
        mock_client.embed.side_effect = Exception("500 internal server error")
        embedder._client = mock_client

        with (
            patch("legal_rag_eval.embedding.voyage.time.sleep"),
            pytest.raises(VoyageEmbedderError, match="failed after"),
        ):
            embedder.embed(["test"])


class TestVoyageEmbedderProtocol:
    """Test that VoyageEmbedder conforms to the Embedder protocol."""

    def test_has_required_interface(self) -> None:
        embedder = VoyageEmbedder(api_key="test-key")
        assert hasattr(embedder, "model_name")
        assert hasattr(embedder, "dimension")
        assert hasattr(embedder, "embed")
        assert hasattr(embedder, "embed_query")
