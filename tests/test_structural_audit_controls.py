"""Independent controls for known structural-metric defects."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import lexichunk
import pytest

import scaffolder.metrics.structural as structural
from scaffolder.models import (
    Chunk,
    ChunkSet,
    Document,
    DocumentType,
    Jurisdiction,
    StrategyName,
)

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def clear_ground_truth_cache() -> Iterator[None]:
    structural._gt_cache.clear()
    yield
    structural._gt_cache.clear()


def _document(document_id: str, text: str) -> Document:
    return Document(
        id=document_id,
        text=text,
        jurisdiction=Jurisdiction.UK,
        document_type=DocumentType.SERVICE_AGREEMENT,
        source="control.txt",
    )


def _chunk_set(
    document_id: str,
    texts: tuple[str, ...],
    metadata: tuple[dict[str, object], ...] | None = None,
) -> ChunkSet:
    chunk_metadata = metadata or tuple({} for _ in texts)
    chunks = tuple(
        Chunk(
            id=f"control_{index}",
            text=text,
            document_id=document_id,
            strategy=StrategyName.FIXED_SIZE,
            index=index,
            metadata=chunk_metadata[index],
        )
        for index, text in enumerate(texts)
    )
    return ChunkSet(
        strategy=StrategyName.FIXED_SIZE,
        document_id=document_id,
        chunks=chunks,
        elapsed_seconds=0.0,
    )


def _install_ground_truth(
    monkeypatch: pytest.MonkeyPatch,
    ground_truth: structural.GroundTruthStructure,
) -> None:
    monkeypatch.setattr(structural, "get_ground_truth", lambda _document: ground_truth)


def test_adjacent_target_section_is_resolved(monkeypatch: pytest.MonkeyPatch) -> None:
    ground_truth = structural.GroundTruthStructure(
        clauses=[],
        clause_boundaries=[],
        defined_terms=set(),
        cross_references=[{"text": "subject to the payment provision", "target": "Clause 99"}],
        max_hierarchy_depth=1,
    )
    _install_ground_truth(monkeypatch, ground_truth)
    chunk_set = _chunk_set(
        "xref-positive",
        ("This obligation is subject to the payment provision.", "Clause 99 — Payment terms."),
    )

    assert (
        structural.cross_ref_resolution_rate(
            chunk_set,
            _document("xref-positive", "source text"),
        )
        == 1.0
    )


def test_cross_reference_self_mention_without_target_is_not_resolved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ground_truth = structural.GroundTruthStructure(
        clauses=[],
        clause_boundaries=[],
        defined_terms=set(),
        cross_references=[{"text": "subject to Clause 99", "target": "Clause 99"}],
        max_hierarchy_depth=1,
    )
    _install_ground_truth(monkeypatch, ground_truth)
    chunk_set = _chunk_set(
        "xref-self-mention",
        ("This obligation is subject to Clause 99, but the target section is absent.",),
    )

    assert (
        structural.cross_ref_resolution_rate(
            chunk_set,
            _document("xref-self-mention", "source text"),
        )
        == 0.0
    )


@pytest.mark.parametrize(
    ("chunk_text", "metadata"),
    [
        pytest.param(
            'The "Confidential Information" must be returned.',
            {},
            id="quoted-use",
        ),
        pytest.param(
            "The Confidential Information must be returned.",
            {"defined_terms": ["Confidential Information"]},
            id="metadata-use",
        ),
    ],
)
def test_term_use_without_definition_is_not_preserved(
    monkeypatch: pytest.MonkeyPatch,
    chunk_text: str,
    metadata: dict[str, object],
) -> None:
    ground_truth = structural.GroundTruthStructure(
        clauses=[],
        clause_boundaries=[],
        defined_terms={"Confidential Information"},
        cross_references=[],
        max_hierarchy_depth=1,
    )
    _install_ground_truth(monkeypatch, ground_truth)
    chunk_set = _chunk_set(
        "definition-use-only",
        (chunk_text,),
        (metadata,),
    )

    assert (
        structural.definition_preservation_rate(
            chunk_set,
            _document("definition-use-only", "source text"),
        )
        == 0.0
    )


def test_same_document_id_cache_refreshes_after_text_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class TextEchoChunker:
        def chunk(self, text: str, document_id: str) -> list[object]:
            return [
                SimpleNamespace(
                    content=text,
                    defined_terms_used=[],
                    cross_references=[],
                    hierarchy_path=None,
                )
            ]

    monkeypatch.setattr(lexichunk, "LegalChunker", TextEchoChunker)
    first_document = _document("same-id", "first document text")
    second_document = _document("same-id", "second document text")

    first_ground_truth = structural.get_ground_truth(first_document)
    second_ground_truth = structural.get_ground_truth(second_document)

    assert first_ground_truth.clauses == ["first document text"]
    assert second_ground_truth.clauses == ["second document text"]
