"""Tests for Voyage model availability toggle."""

from __future__ import annotations

import pytest

from scaffolder.__main__ import _get_available_models
from scaffolder.models import EmbeddingModelName


def test_available_models_without_voyage(monkeypatch: object) -> None:
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)  # type: ignore[union-attr]
    models = _get_available_models()
    assert EmbeddingModelName.MINILM in models
    assert EmbeddingModelName.VOYAGE_LAW_2 not in models


def test_api_key_does_not_auto_select_voyage(monkeypatch: object) -> None:
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")  # type: ignore[union-attr]
    models = _get_available_models()
    assert EmbeddingModelName.VOYAGE_LAW_2 not in models


def test_available_models_with_explicit_voyage(monkeypatch: object) -> None:
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")  # type: ignore[union-attr]
    models = _get_available_models(enable_voyage=True)
    assert EmbeddingModelName.MINILM in models
    assert EmbeddingModelName.VOYAGE_LAW_2 in models


def test_explicit_voyage_requires_key(monkeypatch: object) -> None:
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)  # type: ignore[union-attr]
    with pytest.raises(ValueError, match="VOYAGE_API_KEY"):
        _get_available_models(enable_voyage=True)
