from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

import pytest

from scaffolder.evidence.schema import DatasetError, load_dataset

if TYPE_CHECKING:
    from pathlib import Path


DOCUMENT_TEXT = "Payment is due in ten days. Notice must be written."


def _span(text: str) -> dict[str, object]:
    start = DOCUMENT_TEXT.index(text)
    return {"document_id": "contract", "start": start, "end": start + len(text), "text": text}


def _dataset_payload(document_hash: str) -> dict[str, object]:
    payment = _span("Payment is due in ten days.")
    notice = _span("Notice must be written.")
    return {
        "schema_version": "anchored_evidence_v1",
        "corpus": {
            "id": "test-corpus",
            "title": "Test corpus",
            "description": "Synthetic test data.",
            "authorship": "ai_authored",
            "legal_validation": "not_human_reviewed",
            "held_out": False,
            "customer_proof": False,
        },
        "documents": [
            {
                "id": "contract",
                "jurisdiction": "uk",
                "path": "documents/contract.txt",
                "sha256": document_hash,
                "provenance": {"kind": "synthetic", "source_url": None},
                "license_status": "project_license",
                "license_identifier": "MIT",
                "review_status": "not_human_reviewed",
            }
        ],
        "evidence": [
            {"id": "payment", "grade": 3, **payment},
            {"id": "notice", "grade": 2, **notice},
        ],
        "queries": [
            {
                "id": "combined",
                "text": "When is payment due and what notice is required?",
                "answerable": True,
                "answer_spans": [payment, notice],
                "evidence_ids": ["payment", "notice"],
            },
            {
                "id": "absent",
                "text": "Which court has exclusive jurisdiction?",
                "answerable": False,
                "answer_spans": [],
                "evidence_ids": [],
            },
        ],
    }


def _write_dataset(tmp_path: Path, mutate: object | None = None) -> Path:
    document_path = tmp_path / "documents" / "contract.txt"
    document_path.parent.mkdir()
    document_path.write_text(DOCUMENT_TEXT, encoding="utf-8", newline="")
    payload = _dataset_payload(hashlib.sha256(DOCUMENT_TEXT.encode()).hexdigest())
    if callable(mutate):
        mutate(payload)
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(json.dumps(payload), encoding="utf-8")
    return dataset_path


def test_load_dataset_validates_and_resolves_anchored_spans(tmp_path: Path) -> None:
    dataset = load_dataset(_write_dataset(tmp_path))

    assert dataset.schema_version == "anchored_evidence_v1"
    assert dataset.documents[0].text == DOCUMENT_TEXT
    assert dataset.evidence[0].text == "Payment is due in ten days."
    assert dataset.queries[0].evidence_ids == ("payment", "notice")
    assert len(dataset.manifest_sha256) == 64


def test_bundled_challenge_has_substantive_query_controls() -> None:
    dataset = load_dataset("src/scaffolder/data/synthetic_contracts_v1.json")

    assert len(dataset.queries) >= 12
    assert sum(query.answerable for query in dataset.queries) >= 10
    assert sum(not query.answerable for query in dataset.queries) >= 2
    assert any(len(query.evidence_ids) >= 2 for query in dataset.queries)
    assert any(
        "percentage" in query.text.casefold() for query in dataset.queries if not query.answerable
    )


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda data: data.update(schema_version="unknown"), "schema_version"),
        (lambda data: data["documents"][0].update(sha256="0" * 64), "hash mismatch"),
        (lambda data: data["evidence"][0].update(grade=4), "grade"),
        (lambda data: data["evidence"][0].update(end=999), "out of bounds"),
        (lambda data: data["evidence"][0].update(text="wrong"), "text mismatch"),
        (lambda data: data["queries"][0].update(evidence_ids=["missing"]), "unknown evidence"),
        (lambda data: data["queries"][1].update(evidence_ids=["payment"]), "unanswerable"),
        (lambda data: data["queries"].append(dict(data["queries"][0])), "duplicate query"),
        (lambda data: data["documents"].append(dict(data["documents"][0])), "duplicate document"),
        (lambda data: data["evidence"].append(dict(data["evidence"][0])), "duplicate evidence"),
    ],
)
def test_load_dataset_rejects_invalid_annotations(
    tmp_path: Path,
    mutate: object,
    message: str,
) -> None:
    with pytest.raises(DatasetError, match=message):
        load_dataset(_write_dataset(tmp_path, mutate))


def test_load_dataset_rejects_missing_document(tmp_path: Path) -> None:
    dataset_path = _write_dataset(tmp_path)
    (tmp_path / "documents" / "contract.txt").unlink()

    with pytest.raises(DatasetError, match="missing document"):
        load_dataset(dataset_path)


def test_load_dataset_rejects_path_escape(tmp_path: Path) -> None:
    def mutate(data: dict[str, object]) -> None:
        data["documents"][0]["path"] = "../contract.txt"

    with pytest.raises(DatasetError, match="within dataset directory"):
        load_dataset(_write_dataset(tmp_path, mutate))


def test_load_dataset_allows_truthful_human_reviewed_public_metadata(tmp_path: Path) -> None:
    def mutate(data: dict[str, object]) -> None:
        data["corpus"].update(
            authorship="human_authored",
            legal_validation="human_reviewed",
            held_out=True,
        )
        data["documents"][0].update(
            provenance={
                "kind": "public_source",
                "source_url": "https://github.com/emmcygn/legal-rag-eval",
            },
            license_status="declared_by_source",
            license_identifier="MIT",
            review_status="human_reviewed",
        )

    dataset = load_dataset(_write_dataset(tmp_path, mutate))

    assert dataset.corpus.authorship == "human_authored"
    assert dataset.corpus.held_out is True
    assert dataset.documents[0].provenance_kind == "public_source"


def test_load_dataset_rejects_corpus_document_review_contradiction(tmp_path: Path) -> None:
    def mutate(data: dict[str, object]) -> None:
        data["corpus"]["legal_validation"] = "human_reviewed"

    with pytest.raises(DatasetError, match="every document"):
        load_dataset(_write_dataset(tmp_path, mutate))


def test_load_dataset_rejects_unsupported_jurisdiction(tmp_path: Path) -> None:
    def mutate(data: dict[str, object]) -> None:
        data["documents"][0]["jurisdiction"] = "mars"

    with pytest.raises(DatasetError, match="jurisdiction"):
        load_dataset(_write_dataset(tmp_path, mutate))


def test_load_dataset_requires_evidence_to_cover_each_answer_span(tmp_path: Path) -> None:
    def mutate(data: dict[str, object]) -> None:
        data["queries"][0]["evidence_ids"] = ["notice"]

    with pytest.raises(DatasetError, match="answer span.*evidence"):
        load_dataset(_write_dataset(tmp_path, mutate))
