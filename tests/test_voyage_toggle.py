"""Tests for embedding model resolution, including the Voyage availability toggle.

``scaffolder.__main__._get_available_models`` was replaced by ``_resolve_models(config)``,
which reads the model list straight out of ``BenchmarkConfig.embedding_models`` instead of
hand-listing what's "available": it raises ``ConfigError`` if nothing usable is left, and --
the actual point of the rewrite -- makes ``bge-base-en-v1.5`` selectable through
configuration for the first time.

Paid access has two independent gates and needs BOTH: ``enable_voyage`` (from
``--enable-voyage`` or the config file) and ``VOYAGE_API_KEY``. An exported key is a
credential, not an instruction to spend money, so it must never on its own cause a run to
call a paid API -- the first two tests below pin exactly that.
"""

from __future__ import annotations

import pytest

from scaffolder.__main__ import _resolve_models
from scaffolder.config import BenchmarkConfig, ConfigError
from scaffolder.models import EmbeddingModelName


def test_voyage_skipped_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    config = BenchmarkConfig(embedding_models=["all-MiniLM-L6-v2", "voyage-law-2"], enable_voyage=True)
    models = _resolve_models(config)
    assert EmbeddingModelName.MINILM in models
    assert EmbeddingModelName.VOYAGE_LAW_2 not in models


def test_api_key_alone_does_not_select_voyage(monkeypatch: pytest.MonkeyPatch) -> None:
    """A present key must never be read as permission to spend money."""
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    config = BenchmarkConfig(embedding_models=["all-MiniLM-L6-v2", "voyage-law-2"])
    models = _resolve_models(config)
    assert EmbeddingModelName.MINILM in models
    assert EmbeddingModelName.VOYAGE_LAW_2 not in models


def test_voyage_included_with_key_and_explicit_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    config = BenchmarkConfig(embedding_models=["all-MiniLM-L6-v2", "voyage-law-2"], enable_voyage=True)
    models = _resolve_models(config)
    assert EmbeddingModelName.MINILM in models
    assert EmbeddingModelName.VOYAGE_LAW_2 in models


def test_config_error_when_only_voyage_and_no_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """If voyage is the only configured model and there's no key, nothing is left
    to run with -- that must surface as a ConfigError, not an empty, silent list."""
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    config = BenchmarkConfig(embedding_models=["voyage-law-2"], enable_voyage=True)
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
        embedding_models=["all-MiniLM-L6-v2", "bge-base-en-v1.5", "voyage-law-2"],
        enable_voyage=True,
    )
    models = _resolve_models(config)
    assert models == [
        EmbeddingModelName.MINILM,
        EmbeddingModelName.BGE_BASE,
        EmbeddingModelName.VOYAGE_LAW_2,
    ]
