"""Compatibility layer for optional dependencies in the dashboard."""

from __future__ import annotations

import importlib
import os


def check_embed_available() -> bool:
    """Check if embedding dependencies are installed."""
    try:
        importlib.import_module("sentence_transformers")
        importlib.import_module("faiss")
        return True
    except ImportError:
        return False


def check_voyage_available() -> bool:
    """Check if Voyage AI dependencies and API key are available."""
    try:
        importlib.import_module("voyageai")
        return bool(os.getenv("VOYAGE_API_KEY"))
    except ImportError:
        return False


EMBED_AVAILABLE = check_embed_available()
VOYAGE_AVAILABLE = check_voyage_available()


def get_available_features() -> dict[str, bool]:
    """Return a dict of available features."""
    return {
        "embedding": EMBED_AVAILABLE,
        "voyage": VOYAGE_AVAILABLE,
        "chunking": True,
        "export": True,
    }
