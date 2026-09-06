"""Conservative matching between retrieval chunks and annotated gold sections."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from scaffolder.models import RelevanceGrade

if TYPE_CHECKING:
    from collections.abc import Sequence

    from scaffolder.models import Chunk, RelevantSection


def valid_relevant_sections(
    relevant_sections: Sequence[RelevantSection],
) -> tuple[RelevantSection, ...]:
    """Return uniquely identifiable, positively graded gold sections."""
    sections_by_identity: dict[tuple[str, str, str], RelevantSection] = {}

    for section in relevant_sections:
        if section.grade == RelevanceGrade.IRRELEVANT:
            continue

        identity = _section_identity(section)
        if identity is None:
            continue

        existing = sections_by_identity.get(identity)
        if existing is None or section.grade.value > existing.grade.value:
            sections_by_identity[identity] = section

    return tuple(sections_by_identity.values())


def matching_sections(
    chunk: Chunk,
    relevant_sections: Sequence[RelevantSection],
) -> tuple[RelevantSection, ...]:
    """Return positively graded gold sections with explicit evidence in a chunk."""
    chunk_span = _normalise_span(chunk.text)
    matched: list[RelevantSection] = []

    for section in valid_relevant_sections(relevant_sections):
        if section.document_id.strip() and chunk.document_id.strip() != section.document_id.strip():
            continue

        snippet = _normalise_span(section.text_snippet)
        if snippet and snippet in chunk_span:
            matched.append(section)
            continue

        section_label = _section_label(section.section_id)
        if section_label and _has_explicit_heading(chunk.text, *section_label):
            matched.append(section)

    return tuple(matched)


def _section_identity(section: RelevantSection) -> tuple[str, str, str] | None:
    document_id = section.document_id.strip()
    section_label = _section_label(section.section_id)
    if section_label:
        kind, label = section_label
        return (document_id, kind or "unqualified", label)

    snippet = _normalise_span(section.text_snippet)
    if snippet:
        return (document_id, "span", snippet)

    return None


def has_invalid_positive_gold(relevant_sections: Sequence[RelevantSection]) -> bool:
    """Return whether a positive-grade annotation lacks usable evidence identity."""
    return any(
        section.grade != RelevanceGrade.IRRELEVANT and _section_identity(section) is None
        for section in relevant_sections
    )


def _section_label(section_id: str) -> tuple[str | None, str] | None:
    """Normalise a query label without treating it as arbitrary body text."""
    label = section_id.casefold().strip()
    kind: str | None = None
    for prefix, section_kind in (
        ("clause_", "clause"),
        ("section_", "section"),
        ("definition_", "definition"),
        ("schedule_", "schedule"),
    ):
        if label.startswith(prefix):
            label = label.removeprefix(prefix)
            kind = section_kind
            break
    label = _normalise_span(label.replace("_", " "))
    return (kind, label) if label else None


def _has_explicit_heading(text: str, kind: str | None, label: str) -> bool:
    """Match a section label only at an explicit line-start heading."""
    if re.fullmatch(r"\d+(?:\.\d+)*", label):
        prefix = re.escape(kind) if kind else r"(?:clause|section|article|schedule)"
        return bool(
            re.search(
                rf"(?im)^[ \t]*(?:#{{1,6}}[ \t]*)?(?:{prefix}[ \t]+)?{re.escape(label)}"
                rf"(?=$|[ \t:—-]|\.(?!\d)|\)(?!\d))",
                text,
            )
        )

    label_pattern = re.escape(label).replace(r"\ ", r"[\s_-]+")
    prefix = re.escape(kind) if kind else r"(?:clause|section|article|definition|schedule)"
    return bool(
        re.search(
            rf"(?im)^[ \t]*(?:#{{1,6}}[ \t]*)?(?:{prefix}[ \t]*(?:[:#-][ \t]*|[ \t]+)"
            rf"{label_pattern}|{label_pattern})"
            rf"[ \t]*(?:[.:—-].*|$)",
            text,
        )
    )


def _normalise_span(value: str) -> str:
    return " ".join(value.casefold().split())
