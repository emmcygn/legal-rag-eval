"""Strict loader for independently anchored evidence datasets."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "anchored_evidence_v1"
_VALID_GRADES = frozenset({1, 2, 3})
_VALID_AUTHORSHIP = frozenset({"ai_authored", "human_authored", "mixed", "unknown"})
_VALID_LEGAL_VALIDATION = frozenset({"not_human_reviewed", "human_reviewed", "unknown"})
_VALID_JURISDICTIONS = frozenset({"uk", "us", "eu"})
_VALID_PROVENANCE_KINDS = frozenset({"synthetic", "public_source", "customer_supplied"})
_VALID_LICENSE_STATUS = frozenset({"project_license", "declared_by_source", "unknown"})
_VALID_REVIEW_STATUS = frozenset({"not_human_reviewed", "human_reviewed", "unknown"})


class DatasetError(ValueError):
    """Raised when an evidence dataset is malformed or inconsistent."""


@dataclass(frozen=True, slots=True)
class CorpusMetadata:
    id: str
    title: str
    description: str
    authorship: str
    legal_validation: str
    held_out: bool
    customer_proof: bool


@dataclass(frozen=True, slots=True)
class DocumentRecord:
    id: str
    jurisdiction: str
    path: str
    sha256: str
    provenance_kind: str
    source_url: str | None
    license_status: str
    license_identifier: str
    review_status: str
    text: str


@dataclass(frozen=True, slots=True)
class EvidenceSpan:
    id: str
    document_id: str
    start: int
    end: int
    text: str
    grade: int


@dataclass(frozen=True, slots=True)
class AnswerSpan:
    document_id: str
    start: int
    end: int
    text: str


@dataclass(frozen=True, slots=True)
class EvidenceQuery:
    id: str
    text: str
    answerable: bool
    answer_spans: tuple[AnswerSpan, ...]
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EvidenceDataset:
    schema_version: str
    corpus: CorpusMetadata
    documents: tuple[DocumentRecord, ...]
    evidence: tuple[EvidenceSpan, ...]
    queries: tuple[EvidenceQuery, ...]
    manifest_sha256: str


def load_dataset(path: str | Path) -> EvidenceDataset:
    """Load and fully validate an anchored-evidence JSON dataset."""
    dataset_path = Path(path).resolve()
    try:
        raw_bytes = dataset_path.read_bytes()
    except FileNotFoundError as error:
        raise DatasetError(f"dataset file not found: {dataset_path.name}") from error
    try:
        payload = json.loads(raw_bytes)
    except json.JSONDecodeError as error:
        raise DatasetError(f"invalid dataset JSON: {error.msg}") from error
    if not isinstance(payload, dict):
        raise DatasetError("dataset root must be an object")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise DatasetError(f"schema_version must be {SCHEMA_VERSION}")

    corpus = _load_corpus(_mapping(payload, "corpus"))
    documents = _load_documents(_list(payload, "documents"), dataset_path.parent)
    if corpus.legal_validation == "human_reviewed" and any(
        document.review_status != "human_reviewed" for document in documents
    ):
        raise DatasetError(
            "human_reviewed corpus legal_validation requires every document to be human_reviewed"
        )
    documents_by_id = {document.id: document for document in documents}
    evidence = _load_evidence(_list(payload, "evidence"), documents_by_id)
    evidence_by_id = {span.id: span for span in evidence}
    queries = _load_queries(_list(payload, "queries"), documents_by_id, evidence_by_id)
    return EvidenceDataset(
        schema_version=SCHEMA_VERSION,
        corpus=corpus,
        documents=documents,
        evidence=evidence,
        queries=queries,
        manifest_sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )


def _load_corpus(data: dict[str, Any]) -> CorpusMetadata:
    corpus = CorpusMetadata(
        id=_text(data, "id"),
        title=_text(data, "title"),
        description=_text(data, "description"),
        authorship=_text(data, "authorship"),
        legal_validation=_text(data, "legal_validation"),
        held_out=_boolean(data, "held_out"),
        customer_proof=_boolean(data, "customer_proof"),
    )
    if corpus.authorship not in _VALID_AUTHORSHIP:
        raise DatasetError(f"invalid corpus authorship: {corpus.authorship}")
    if corpus.legal_validation not in _VALID_LEGAL_VALIDATION:
        raise DatasetError(f"invalid corpus legal_validation: {corpus.legal_validation}")
    return corpus


def _load_documents(
    items: list[Any],
    dataset_directory: Path,
) -> tuple[DocumentRecord, ...]:
    documents: list[DocumentRecord] = []
    seen_ids: set[str] = set()
    for item in items:
        data = _as_mapping(item, "document")
        document_id = _text(data, "id")
        if document_id in seen_ids:
            raise DatasetError(f"duplicate document id: {document_id}")
        seen_ids.add(document_id)
        relative_path = Path(_text(data, "path"))
        document_path = (dataset_directory / relative_path).resolve()
        if not document_path.is_relative_to(dataset_directory):
            raise DatasetError(f"document {document_id} path must remain within dataset directory")
        try:
            document_bytes = document_path.read_bytes()
        except FileNotFoundError as error:
            raise DatasetError(f"missing document for {document_id}: {relative_path}") from error
        try:
            text = document_bytes.decode("utf-8")
        except UnicodeDecodeError as error:
            raise DatasetError(f"document {document_id} must be UTF-8") from error
        expected_hash = _sha256(data, "sha256")
        actual_hash = hashlib.sha256(document_bytes).hexdigest()
        if actual_hash != expected_hash:
            raise DatasetError(f"document {document_id} hash mismatch")
        provenance = _mapping(data, "provenance")
        source_url = provenance.get("source_url")
        if source_url is not None and not isinstance(source_url, str):
            raise DatasetError(f"document {document_id} source_url must be a string or null")
        document = DocumentRecord(
            id=document_id,
            jurisdiction=_text(data, "jurisdiction"),
            path=relative_path.as_posix(),
            sha256=expected_hash,
            provenance_kind=_text(provenance, "kind"),
            source_url=source_url,
            license_status=_text(data, "license_status"),
            license_identifier=_text(data, "license_identifier"),
            review_status=_text(data, "review_status"),
            text=text,
        )
        if document.jurisdiction not in _VALID_JURISDICTIONS:
            raise DatasetError(f"document {document_id} has unsupported jurisdiction")
        if document.provenance_kind not in _VALID_PROVENANCE_KINDS:
            raise DatasetError(f"document {document_id} has invalid provenance kind")
        if document.provenance_kind == "synthetic" and document.source_url is not None:
            raise DatasetError(f"synthetic document {document_id} cannot claim a source URL")
        if document.provenance_kind == "public_source" and document.source_url is None:
            raise DatasetError(f"public document {document_id} requires a source URL")
        if document.license_status not in _VALID_LICENSE_STATUS:
            raise DatasetError(f"document {document_id} has invalid license_status")
        if document.review_status not in _VALID_REVIEW_STATUS:
            raise DatasetError(f"document {document_id} has invalid review_status")
        documents.append(document)
    if not documents:
        raise DatasetError("documents must not be empty")
    return tuple(documents)


def _load_evidence(
    items: list[Any],
    documents: dict[str, DocumentRecord],
) -> tuple[EvidenceSpan, ...]:
    evidence: list[EvidenceSpan] = []
    seen_ids: set[str] = set()
    for item in items:
        data = _as_mapping(item, "evidence")
        evidence_id = _text(data, "id")
        if evidence_id in seen_ids:
            raise DatasetError(f"duplicate evidence id: {evidence_id}")
        seen_ids.add(evidence_id)
        grade = data.get("grade")
        if not isinstance(grade, int) or isinstance(grade, bool) or grade not in _VALID_GRADES:
            raise DatasetError(f"evidence {evidence_id} grade must be 1, 2, or 3")
        span = EvidenceSpan(
            id=evidence_id,
            document_id=_text(data, "document_id"),
            start=_integer(data, "start"),
            end=_integer(data, "end"),
            text=_text(data, "text"),
            grade=grade,
        )
        _validate_span(span.document_id, span.start, span.end, span.text, documents, evidence_id)
        evidence.append(span)
    return tuple(evidence)


def _load_queries(
    items: list[Any],
    documents: dict[str, DocumentRecord],
    evidence: dict[str, EvidenceSpan],
) -> tuple[EvidenceQuery, ...]:
    queries: list[EvidenceQuery] = []
    seen_ids: set[str] = set()
    for item in items:
        data = _as_mapping(item, "query")
        query_id = _text(data, "id")
        if query_id in seen_ids:
            raise DatasetError(f"duplicate query id: {query_id}")
        seen_ids.add(query_id)
        answer_spans = tuple(
            _load_answer_span(_as_mapping(span, f"query {query_id} answer span"), documents)
            for span in _list(data, "answer_spans")
        )
        evidence_ids = tuple(_string_list(data, "evidence_ids"))
        if len(evidence_ids) != len(set(evidence_ids)):
            raise DatasetError(f"query {query_id} has duplicate evidence ids")
        unknown = [evidence_id for evidence_id in evidence_ids if evidence_id not in evidence]
        if unknown:
            raise DatasetError(f"query {query_id} references unknown evidence: {unknown}")
        answerable = _boolean(data, "answerable")
        if answerable and (not answer_spans or not evidence_ids):
            raise DatasetError(f"answerable query {query_id} needs answer spans and evidence")
        if not answerable and (answer_spans or evidence_ids):
            raise DatasetError(
                f"unanswerable query {query_id} cannot contain answer spans or evidence"
            )
        referenced_evidence = tuple(evidence[evidence_id] for evidence_id in evidence_ids)
        for answer_span in answer_spans:
            if not _answer_span_is_covered(answer_span, referenced_evidence):
                raise DatasetError(
                    f"query {query_id} answer span is not covered by referenced evidence"
                )
        queries.append(
            EvidenceQuery(
                id=query_id,
                text=_text(data, "text"),
                answerable=answerable,
                answer_spans=answer_spans,
                evidence_ids=evidence_ids,
            )
        )
    if not queries:
        raise DatasetError("queries must not be empty")
    return tuple(queries)


def _load_answer_span(
    data: dict[str, Any],
    documents: dict[str, DocumentRecord],
) -> AnswerSpan:
    span = AnswerSpan(
        document_id=_text(data, "document_id"),
        start=_integer(data, "start"),
        end=_integer(data, "end"),
        text=_text(data, "text"),
    )
    _validate_span(span.document_id, span.start, span.end, span.text, documents, "answer span")
    return span


def _validate_span(
    document_id: str,
    start: int,
    end: int,
    text: str,
    documents: dict[str, DocumentRecord],
    label: str,
) -> None:
    document = documents.get(document_id)
    if document is None:
        raise DatasetError(f"{label} references unknown document: {document_id}")
    if start < 0 or end <= start or end > len(document.text):
        raise DatasetError(f"{label} span is out of bounds")
    if document.text[start:end] != text:
        raise DatasetError(f"{label} span text mismatch")


def _answer_span_is_covered(
    answer_span: AnswerSpan,
    evidence: tuple[EvidenceSpan, ...],
) -> bool:
    intervals = sorted(
        (span.start, span.end) for span in evidence if span.document_id == answer_span.document_id
    )
    cursor = answer_span.start
    for start, end in intervals:
        if end <= cursor:
            continue
        if start > cursor:
            return False
        cursor = max(cursor, end)
        if cursor >= answer_span.end:
            return True
    return False


def _mapping(data: dict[str, Any], key: str) -> dict[str, Any]:
    return _as_mapping(data.get(key), key)


def _as_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DatasetError(f"{label} must be an object")
    return value


def _list(data: dict[str, Any], key: str) -> list[Any]:
    value = data.get(key)
    if not isinstance(value, list):
        raise DatasetError(f"{key} must be a list")
    return value


def _text(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise DatasetError(f"{key} must be a non-empty string")
    return value


def _boolean(data: dict[str, Any], key: str) -> bool:
    value = data.get(key)
    if not isinstance(value, bool):
        raise DatasetError(f"{key} must be a boolean")
    return value


def _integer(data: dict[str, Any], key: str) -> int:
    value = data.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise DatasetError(f"{key} must be an integer")
    return value


def _sha256(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise DatasetError(f"{key} must be a lowercase SHA-256 value")
    return value


def _string_list(data: dict[str, Any], key: str) -> list[str]:
    values = _list(data, key)
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise DatasetError(f"{key} must contain non-empty strings")
    return values
