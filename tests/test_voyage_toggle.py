"""Tests for embedding model resolution, including the Voyage availability toggle.

``scaffolder.__main__._get_available_models`` was replaced by ``_resolve_models(config)``,
which reads the model list straight out of ``BenchmarkConfig.embedding_models`` instead of
hand-listing what's "available": it skips ``voyage-law-2`` when ``VOYAGE_API_KEY`` is unset,
raises ``ConfigError`` if nothing usable is left, and -- the actual point of the rewrite --
makes ``bge-base-en-v1.5`` selectable through configuration for the first time.
"""

from __future__ import annotations

import pytest

from scaffolder.__main__ import _resolve_models
from scaffolder.config import BenchmarkConfig, ConfigError
from scaffolder.models import EmbeddingModelName


def test_voyage_skipped_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    config = BenchmarkConfig(embedding_models=["all-MiniLM-L6-v2", "voyage-law-2"])
    models = _resolve_models(config)
    assert EmbeddingModelName.MINILM in models
    assert EmbeddingModelName.VOYAGE_LAW_2 not in models


def test_voyage_included_when_key_is_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    config = BenchmarkConfig(embedding_models=["all-MiniLM-L6-v2", "voyage-law-2"])
    models = _resolve_models(config)
    assert EmbeddingModelName.MINILM in models
    assert EmbeddingModelName.VOYAGE_LAW_2 in models


def test_config_error_when_only_voyage_and_no_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """If voyage is the only configured model and there's no key, nothing is left
    to run with -- that must surface as a ConfigError, not an empty, silent list."""
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    config = BenchmarkConfig(embedding_models=["voyage-law-2"])
    with pytest.raises(ConfigError, match="No usable embedding models"):
        _resolve_models(config)


def test_bge_base_is_selectable(monkeypatch: pytest.MonkeyPatch) -> None:
    """bge-base-en-v1.5 was previously not selectable at all; it must now resolve
    like any other configured model."""
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    config = BenchmarkConfig(embedding_models=["bge-base-en-v1.5"])
    models = _resolve_models(config)
    assert models == [EmbeddingModelName.BGE_BASE]


def test_bge_base_alongside_minilm_and_voyage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    config = BenchmarkConfig(
        embedding_models=["all-MiniLM-L6-v2", "bge-base-en-v1.5", "voyage-law-2"]
    )
    models = _resolve_models(config)
    assert models == [
        EmbeddingModelName.MINILM,
        EmbeddingModelName.BGE_BASE,
        EmbeddingModelName.VOYAGE_LAW_2,
    ]
