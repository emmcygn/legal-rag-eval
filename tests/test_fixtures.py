"""Tests for FixtureManager."""

from __future__ import annotations

from pathlib import Path

import pytest

from legal_rag_eval.fixtures import FixtureManager
from legal_rag_eval.models import DocumentType, Jurisdiction


class TestFixtureManager:
    """Tests for fixture loading and discovery."""

    def test_load_all_returns_five_documents(self) -> None:
        manager = FixtureManager()
        docs = manager.load_all()
        assert len(docs) == 5

    def test_all_documents_have_non_empty_text(self) -> None:
        manager = FixtureManager()
        for doc in manager.load_all():
            assert len(doc.text) > 0
            assert doc.char_count > 0

    def test_all_documents_have_line_count(self) -> None:
        manager = FixtureManager()
        for doc in manager.load_all():
            assert doc.line_count >= 1

    def test_document_ids_match_filenames(self) -> None:
        manager = FixtureManager()
        manager.load_all()
        expected_ids = {
            "uk_service_agreement",
            "uk_terms_conditions",
            "us_msa",
            "us_terms_of_service",
            "eu_gdpr_excerpt",
        }
        assert set(manager.document_ids) == expected_ids

    def test_get_by_id(self) -> None:
        manager = FixtureManager()
        doc = manager.get_by_id("uk_service_agreement")
        assert doc.jurisdiction == Jurisdiction.UK
        assert doc.document_type == DocumentType.SERVICE_AGREEMENT

    def test_get_by_id_not_found_raises(self) -> None:
        manager = FixtureManager()
        with pytest.raises(KeyError, match="nonexistent"):
            manager.get_by_id("nonexistent")

    def test_get_by_jurisdiction(self) -> None:
        manager = FixtureManager()
        uk_docs = manager.get_by_jurisdiction(Jurisdiction.UK)
        assert len(uk_docs) == 2
        for doc in uk_docs:
            assert doc.jurisdiction == Jurisdiction.UK

    def test_get_by_jurisdiction_us(self) -> None:
        manager = FixtureManager()
        us_docs = manager.get_by_jurisdiction(Jurisdiction.US)
        assert len(us_docs) == 2

    def test_get_by_jurisdiction_eu(self) -> None:
        manager = FixtureManager()
        eu_docs = manager.get_by_jurisdiction(Jurisdiction.EU)
        assert len(eu_docs) == 1

    def test_lazy_loading(self) -> None:
        manager = FixtureManager()
        assert not manager._loaded
        docs = manager.get_all()
        assert manager._loaded
        assert len(docs) == 5

    def test_missing_directory_raises(self) -> None:
        manager = FixtureManager(documents_dir=Path("/nonexistent/path"))
        with pytest.raises(FileNotFoundError):
            manager.load_all()

    def test_document_is_frozen(self) -> None:
        manager = FixtureManager()
        doc = manager.get_by_id("us_msa")
        with pytest.raises(AttributeError):
            doc.text = "modified"  # type: ignore[misc]

    def test_char_count_matches_text_length(self) -> None:
        manager = FixtureManager()
        for doc in manager.load_all():
            assert doc.char_count == len(doc.text)
