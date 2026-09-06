"""Shared pytest fixtures for legal_rag_eval tests."""

from __future__ import annotations

import pytest

from legal_rag_eval.config import BenchmarkConfig


@pytest.fixture
def default_config() -> BenchmarkConfig:
    """A default BenchmarkConfig for testing."""
    return BenchmarkConfig()


@pytest.fixture
def sample_legal_text() -> str:
    """A short legal text snippet for testing."""
    return (
        "1. DEFINITIONS\n"
        '1.1 "Agreement" means this Master Service Agreement.\n'
        '1.2 "Confidential Information" means any information disclosed '
        "by either party that is marked as confidential.\n\n"
        "2. TERM AND TERMINATION\n"
        "2.1 This Agreement shall commence on the Effective Date and "
        "continue for a period of twelve (12) months.\n"
        "2.2 Either party may terminate this Agreement by giving "
        "ninety (90) days written notice.\n"
        "2.3 Termination shall not affect the rights and obligations "
        "set forth in Sections 3 (Confidentiality) and 5 (Limitation of Liability).\n"
    )
