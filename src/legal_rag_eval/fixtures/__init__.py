"""Fixture loading and document management."""

from __future__ import annotations

import logging
from pathlib import Path

from legal_rag_eval.models import Document, DocumentType, Jurisdiction

logger = logging.getLogger(__name__)

_DOCUMENTS_DIR = Path(__file__).parent / "documents"

_FIXTURE_METADATA: dict[str, tuple[Jurisdiction, DocumentType]] = {
    "uk_service_agreement.txt": (Jurisdiction.UK, DocumentType.SERVICE_AGREEMENT),
    "uk_terms_conditions.txt": (Jurisdiction.UK, DocumentType.TERMS_CONDITIONS),
    "us_msa.txt": (Jurisdiction.US, DocumentType.MSA),
    "us_terms_of_service.txt": (Jurisdiction.US, DocumentType.TERMS_OF_SERVICE),
    "eu_gdpr_excerpt.txt": (Jurisdiction.EU, DocumentType.GDPR_EXCERPT),
}


class FixtureManager:
    """Discovers and loads legal document fixtures from the documents/ directory.

    Usage:
        fm = FixtureManager()
        docs = fm.load_all()           # returns list[Document], sorted by id
        doc = fm.get_by_id("us_msa")   # returns single Document
        uk = fm.get_by_jurisdiction(Jurisdiction.UK)  # returns list[Document]
    """

    def __init__(self, documents_dir: Path | None = None) -> None:
        self._documents_dir = documents_dir or _DOCUMENTS_DIR
        self._documents: dict[str, Document] = {}
        self._loaded = False

    def load_all(self) -> list[Document]:
        """Discover and load all .txt files in the documents directory.

        Returns:
            List of Document instances, sorted by id.

        Raises:
            FileNotFoundError: If documents directory does not exist.
            ValueError: If a .txt file has no metadata mapping.
        """
        if not self._documents_dir.is_dir():
            msg = f"Documents directory not found: {self._documents_dir}"
            raise FileNotFoundError(msg)

        self._documents.clear()
        txt_files = sorted(self._documents_dir.glob("*.txt"))

        if not txt_files:
            logger.warning("No .txt files found in %s", self._documents_dir)
            return []

        for path in txt_files:
            doc = self._load_document(path)
            self._documents[doc.id] = doc
            logger.info(
                "Loaded %s (%s, %s) -- %d chars, %d lines",
                doc.id,
                doc.jurisdiction.value,
                doc.document_type.value,
                doc.char_count,
                doc.line_count,
            )

        self._loaded = True
        return self.get_all()

    def get_all(self) -> list[Document]:
        """Return all loaded documents, sorted by id."""
        if not self._loaded:
            self.load_all()
        return sorted(self._documents.values(), key=lambda d: d.id)

    def get_by_id(self, document_id: str) -> Document:
        """Return a specific document by id.

        Raises:
            KeyError: If document_id is not found.
        """
        if not self._loaded:
            self.load_all()
        if document_id not in self._documents:
            msg = f"Document '{document_id}' not found. Available: {sorted(self._documents.keys())}"
            raise KeyError(msg)
        return self._documents[document_id]

    def get_by_jurisdiction(self, jurisdiction: Jurisdiction) -> list[Document]:
        """Return all documents for a given jurisdiction."""
        if not self._loaded:
            self.load_all()
        return [d for d in self._documents.values() if d.jurisdiction == jurisdiction]

    @property
    def document_ids(self) -> list[str]:
        """Return sorted list of all document ids."""
        if not self._loaded:
            self.load_all()
        return sorted(self._documents.keys())

    def _load_document(self, path: Path) -> Document:
        """Load a single document from a file path."""
        filename = path.name
        if filename not in _FIXTURE_METADATA:
            msg = (
                f"No metadata mapping for '{filename}'. "
                f"Add it to _FIXTURE_METADATA in fixtures/__init__.py."
            )
            raise ValueError(msg)

        jurisdiction, document_type = _FIXTURE_METADATA[filename]
        text = path.read_text(encoding="utf-8")

        doc_id = path.stem  # e.g., "uk_service_agreement"

        return Document(
            id=doc_id,
            text=text,
            jurisdiction=jurisdiction,
            document_type=document_type,
            source=filename,
        )


__all__ = ["FixtureManager"]
