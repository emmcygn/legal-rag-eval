"""Query annotation loading and resolution to gold clause spans.

Query YAML files (see ``queries/*.yaml``) name relevant clauses by their gold
*identifier* (e.g. ``"12.1"``, ``"article_6"``) rather than by prose. This module
has two jobs:

1. :func:`load_queries` parses those YAML files into :class:`AnnotatedQuery` /
   :class:`RelevantSection` objects — plain data, not yet checked against any
   gold annotation.
2. :func:`resolve_queries` maps each named identifier to the corresponding gold
   clause's *subtree* span (:meth:`scaffolder.gold.GoldAnnotation.subtree_span`),
   producing :class:`scaffolder.models.AnnotatedQuery` objects with concrete
   ``char_start``/``char_end`` offsets that :mod:`scaffolder.metrics.retrieval`
   and :mod:`scaffolder.retrieval.simulator` compare chunk spans against.

Splitting the two steps means a query can be parsed and validated for shape
without needing gold annotations loaded, while resolution is where an
unsatisfiable query (naming an identifier that does not exist) is caught.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from scaffolder import models

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from scaffolder.gold import GoldAnnotation

_VALID_RELEVANCE_GRADES = frozenset({1, 2, 3})


class QueryError(Exception):
    """Raised for any problem loading, parsing, or resolving a query annotation."""


@dataclass(frozen=True)
class RelevantSection:
    """One gold clause identifier named as relevant to a query.

    This is *not yet* a span: ``identifier`` is resolved to document offsets by
    :func:`resolve_queries`, which looks it up in the target document's gold
    annotation.
    """

    identifier: str
    relevance: int  # graded relevance: 3=exact, 2=partial, 1=background
    description: str = ""

    @property
    def section_id(self) -> str:
        """Legacy alias for ``identifier``, for code written against the pre-migration schema."""
        return self.identifier


@dataclass(frozen=True)
class AnnotatedQuery:
    """A query with ground-truth relevant-clause annotations, not yet resolved to spans."""

    id: str
    text: str
    document_id: str
    category: str  # free-form label, e.g. "clause_lookup"
    relevant_clauses: list[RelevantSection]
    notes: str = ""

    @property
    def failure_mode(self) -> str:
        """Legacy alias for ``category``, for code written against the pre-migration schema."""
        return self.category

    @property
    def relevant_sections(self) -> list[RelevantSection]:
        """Legacy alias for ``relevant_clauses``, for code written against the old schema."""
        return self.relevant_clauses


def _parse_relevant_clauses(query_id: str, raw_query: dict[str, Any]) -> list[RelevantSection]:
    """Parse the new ``relevant_clauses``/``identifier`` keys, or the legacy
    ``relevant_sections``/``section_id`` keys when the new ones are absent.
    """
    raw_clauses = raw_query.get("relevant_clauses")
    source_key, id_key = "relevant_clauses", "identifier"
    if raw_clauses is None:
        raw_clauses = raw_query.get("relevant_sections")
        source_key, id_key = "relevant_sections", "section_id"

    if not raw_clauses:
        raise QueryError(
            f"query {query_id!r} has no relevant clauses (expected a non-empty "
            f"'relevant_clauses' or legacy 'relevant_sections' list)"
        )

    sections: list[RelevantSection] = []
    for index, item in enumerate(raw_clauses):
        try:
            identifier = item[id_key]
            relevance = item["relevance"]
        except KeyError as exc:
            raise QueryError(
                f"entry #{index} of {source_key!r} in query {query_id!r} is missing key {exc}"
            ) from exc

        if relevance not in _VALID_RELEVANCE_GRADES:
            raise QueryError(
                f"entry #{index} of {source_key!r} in query {query_id!r} has relevance "
                f"{relevance!r}; must be one of {sorted(_VALID_RELEVANCE_GRADES)}"
            )

        sections.append(
            RelevantSection(
                identifier=identifier,
                relevance=relevance,
                description=item.get("description", ""),
            )
        )

    return sections


def load_queries(query_dir: str | Path) -> list[AnnotatedQuery]:
    """Load all query annotation files from a directory.

    Accepts both the current schema (``relevant_clauses`` entries keyed by
    ``identifier``) and the legacy schema (``relevant_sections`` entries keyed by
    ``section_id``), preferring the new keys when a query has both. ``category``
    is a free-form label; the legacy ``failure_mode`` key is accepted as an alias
    when ``category`` is absent.

    Raises ``QueryError`` if a query is missing required keys, has no relevant
    clauses, or a relevance grade is not 1, 2, or 3. Does not check identifiers
    against any gold annotation — that is :func:`resolve_queries`'s job.
    """
    query_dir = Path(query_dir)
    queries: list[AnnotatedQuery] = []

    for yaml_path in sorted(query_dir.glob("*.yaml")):
        with open(yaml_path, encoding="utf-8") as f:
            data: dict[str, Any] = yaml.safe_load(f) or {}

        document_id = data.get("document_id", yaml_path.stem)

        for q in data.get("queries", []):
            try:
                query_id = q["id"]
                text = q["text"]
            except KeyError as exc:
                raise QueryError(f"a query in {yaml_path.name} is missing key {exc}") from exc

            category = q.get("category")
            if category is None:
                category = q.get("failure_mode", "")

            sections = _parse_relevant_clauses(query_id, q)

            queries.append(
                AnnotatedQuery(
                    id=query_id,
                    text=text,
                    document_id=document_id,
                    category=category,
                    relevant_clauses=sections,
                    notes=q.get("notes", ""),
                )
            )

    return queries


def load_queries_for_document(query_dir: str | Path, document_id: str) -> list[AnnotatedQuery]:
    """Load queries for a specific document."""
    all_queries = load_queries(query_dir)
    return [q for q in all_queries if q.document_id == document_id]


def _infer_jurisdiction(document_id: str) -> models.Jurisdiction:
    """Infer jurisdiction from the document id's prefix, defaulting to UK."""
    if document_id.startswith("uk_"):
        return models.Jurisdiction.UK
    if document_id.startswith("us_"):
        return models.Jurisdiction.US
    if document_id.startswith("eu_"):
        return models.Jurisdiction.EU
    return models.Jurisdiction.UK


def resolve_queries(
    queries: Sequence[AnnotatedQuery],
    gold_by_document: Mapping[str, GoldAnnotation],
) -> list[models.AnnotatedQuery]:
    """Resolve raw query annotations to gold clause spans.

    For every :class:`RelevantSection`, looks up ``identifier`` in the query's
    document's gold annotation (``gold_by_document[query.document_id]``) and
    resolves it to that clause's *subtree* span
    (:meth:`~scaffolder.gold.GoldAnnotation.subtree_span`) — the clause's own
    span extended to cover its descendants — producing a
    :class:`scaffolder.models.RelevantSection` with concrete character offsets.
    Using the subtree span (not the clause's own span) means naming a parent
    clause counts its children's text too, so a query about "the termination
    clause" is not defeated by the fact that the interesting text lives in a
    numbered sub-clause.

    Raises ``QueryError``, naming the query id, document, and identifier, when:

    * ``query.document_id`` has no entry in ``gold_by_document``;
    * an ``identifier`` does not name any clause in that document's gold
      annotation;
    * a query has no relevant clauses;
    * a section's ``relevance`` is not one of 1, 2, 3.

    This raises eagerly, at load time, rather than resolving an unsatisfiable
    query to zero relevant sections: a silently-empty relevant-section set would
    contribute a guaranteed zero to every strategy's recall for that query
    without anyone noticing the annotation itself was broken.
    """
    resolved: list[models.AnnotatedQuery] = []

    for query in queries:
        gold = gold_by_document.get(query.document_id)
        if gold is None:
            raise QueryError(
                f"query {query.id!r} targets document {query.document_id!r}, which has no "
                f"gold annotation loaded"
            )

        if not query.relevant_clauses:
            raise QueryError(f"query {query.id!r} has no relevant clauses")

        sections: list[models.RelevantSection] = []
        for clause_ref in query.relevant_clauses:
            if clause_ref.relevance not in _VALID_RELEVANCE_GRADES:
                raise QueryError(
                    f"query {query.id!r} names identifier {clause_ref.identifier!r} in document "
                    f"{query.document_id!r} with relevance {clause_ref.relevance!r}; must be one "
                    f"of {sorted(_VALID_RELEVANCE_GRADES)}"
                )

            clause = gold.by_identifier.get(clause_ref.identifier)
            if clause is None:
                raise QueryError(
                    f"query {query.id!r} names identifier {clause_ref.identifier!r} which does "
                    f"not exist in the gold annotation for document {query.document_id!r}"
                )

            char_start, char_end = gold.subtree_span(clause)
            sections.append(
                models.RelevantSection(
                    document_id=query.document_id,
                    section_id=clause_ref.identifier,
                    char_start=char_start,
                    char_end=char_end,
                    grade=models.RelevanceGrade(clause_ref.relevance),
                    description=clause_ref.description,
                )
            )

        resolved.append(
            models.AnnotatedQuery(
                id=query.id,
                text=query.text,
                document_ids=[query.document_id],
                jurisdiction=_infer_jurisdiction(query.document_id),
                relevant_sections=tuple(sections),
                category=query.category,
            )
        )

    return resolved
