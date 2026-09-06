"""Tests for query annotation loading and resolution to gold clause spans.

Builds synthetic query YAML and gold annotations directly rather than
depending on the real ``queries/*.yaml`` or ``gold/*.json`` fixtures, both of
which are being rewritten concurrently.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import yaml

from scaffolder.gold import GoldAnnotation, GoldClause
from scaffolder.models import Jurisdiction, RelevanceGrade
from scaffolder.queries import (
    AnnotatedQuery,
    QueryError,
    RelevantSection,
    load_queries,
    load_queries_for_document,
    resolve_queries,
)

if TYPE_CHECKING:
    from pathlib import Path


def _gold_annotation(document_id: str = "test_doc") -> GoldAnnotation:
    """A minimal two-clause gold annotation: "1" (parent) contains "1.1" (child)."""
    clause_1 = GoldClause(
        identifier="1",
        level=1,
        parent=None,
        char_start=0,
        char_end=300,
        is_top_level=True,
        heading="Clause 1",
    )
    clause_1_1 = GoldClause(
        identifier="1.1",
        level=2,
        parent="1",
        char_start=300,
        char_end=600,
        is_top_level=False,
        heading="Clause 1.1",
    )
    return GoldAnnotation(
        document_id=document_id,
        source_file=f"{document_id}.txt",
        text_sha256="0" * 64,
        n_chars=5000,
        numbering_style="uk_decimal",
        clauses=(clause_1, clause_1_1),
        defined_terms=(),
        cross_references=(),
    )


class TestLoadQueriesNewSchema:
    def test_loads_relevant_clauses_with_identifier(self, tmp_path: Path) -> None:
        data = {
            "document_id": "uk_service_agreement",
            "queries": [
                {
                    "id": "uk_sa_q1",
                    "text": "What notice must a party give to terminate for convenience?",
                    "category": "clause_lookup",
                    "relevant_clauses": [
                        {
                            "identifier": "12.1",
                            "relevance": 3,
                            "description": "Termination for convenience — notice period",
                        },
                        {
                            "identifier": "12",
                            "relevance": 1,
                            "description": "Parent termination clause",
                        },
                    ],
                    "notes": "Tests clause fragmentation.",
                }
            ],
        }
        (tmp_path / "uk_service_agreement.yaml").write_text(yaml.dump(data))

        queries = load_queries(tmp_path)
        assert len(queries) == 1
        query = queries[0]
        assert query.id == "uk_sa_q1"
        assert query.document_id == "uk_service_agreement"
        assert query.category == "clause_lookup"
        assert len(query.relevant_clauses) == 2
        assert query.relevant_clauses[0].identifier == "12.1"
        assert query.relevant_clauses[0].relevance == 3

    def test_category_is_free_form(self, tmp_path: Path) -> None:
        """category is not restricted to a fixed enum of failure modes."""
        data = {
            "document_id": "doc",
            "queries": [
                {
                    "id": "q1",
                    "text": "text",
                    "category": "some_new_label_nobody_pre_registered",
                    "relevant_clauses": [{"identifier": "1", "relevance": 2}],
                }
            ],
        }
        (tmp_path / "doc.yaml").write_text(yaml.dump(data))

        queries = load_queries(tmp_path)
        assert queries[0].category == "some_new_label_nobody_pre_registered"


class TestLoadQueriesLegacySchema:
    def test_loads_relevant_sections_with_section_id(self, tmp_path: Path) -> None:
        data = {
            "document_id": "eu_gdpr_excerpt",
            "queries": [
                {
                    "id": "eu_gdpr_q1",
                    "text": "What are the lawful bases for processing personal data?",
                    "failure_mode": "clause_fragmentation",
                    "relevant_sections": [
                        {
                            "section_id": "article_6",
                            "relevance": 3,
                            "description": "Article 6",
                        }
                    ],
                }
            ],
        }
        (tmp_path / "eu_gdpr_excerpt.yaml").write_text(yaml.dump(data))

        queries = load_queries(tmp_path)
        assert len(queries) == 1
        query = queries[0]
        # New-schema attribute names work via the legacy data too.
        assert query.category == "clause_fragmentation"
        assert query.failure_mode == "clause_fragmentation"  # legacy alias property
        assert query.relevant_clauses[0].identifier == "article_6"
        assert query.relevant_clauses[0].section_id == "article_6"  # legacy alias property
        assert query.relevant_sections == query.relevant_clauses  # legacy alias property

    def test_new_keys_preferred_when_both_present(self, tmp_path: Path) -> None:
        data = {
            "document_id": "doc",
            "queries": [
                {
                    "id": "q1",
                    "text": "text",
                    "category": "new_category",
                    "failure_mode": "old_category",
                    "relevant_clauses": [{"identifier": "new_id", "relevance": 3}],
                    "relevant_sections": [{"section_id": "old_id", "relevance": 1}],
                }
            ],
        }
        (tmp_path / "doc.yaml").write_text(yaml.dump(data))

        queries = load_queries(tmp_path)
        query = queries[0]
        assert query.category == "new_category"
        assert query.relevant_clauses[0].identifier == "new_id"


class TestLoadQueriesValidation:
    def test_no_relevant_clauses_raises(self, tmp_path: Path) -> None:
        data = {
            "document_id": "doc",
            "queries": [{"id": "q1", "text": "text", "category": "x", "relevant_clauses": []}],
        }
        (tmp_path / "doc.yaml").write_text(yaml.dump(data))

        with pytest.raises(QueryError, match="q1"):
            load_queries(tmp_path)

    def test_invalid_relevance_raises(self, tmp_path: Path) -> None:
        data = {
            "document_id": "doc",
            "queries": [
                {
                    "id": "q1",
                    "text": "text",
                    "category": "x",
                    "relevant_clauses": [{"identifier": "1", "relevance": 5}],
                }
            ],
        }
        (tmp_path / "doc.yaml").write_text(yaml.dump(data))

        with pytest.raises(QueryError, match="relevance"):
            load_queries(tmp_path)

    def test_missing_id_raises(self, tmp_path: Path) -> None:
        data = {
            "document_id": "doc",
            "queries": [
                {
                    "text": "text",
                    "category": "x",
                    "relevant_clauses": [{"identifier": "1", "relevance": 1}],
                }
            ],
        }
        (tmp_path / "doc.yaml").write_text(yaml.dump(data))

        with pytest.raises(QueryError):
            load_queries(tmp_path)


class TestLoadQueriesForDocument:
    def test_filters_by_document(self, tmp_path: Path) -> None:
        data_a = {
            "document_id": "doc_a",
            "queries": [
                {
                    "id": "a1",
                    "text": "text",
                    "category": "x",
                    "relevant_clauses": [{"identifier": "1", "relevance": 1}],
                }
            ],
        }
        data_b = {
            "document_id": "doc_b",
            "queries": [
                {
                    "id": "b1",
                    "text": "text",
                    "category": "x",
                    "relevant_clauses": [{"identifier": "1", "relevance": 1}],
                }
            ],
        }
        (tmp_path / "a.yaml").write_text(yaml.dump(data_a))
        (tmp_path / "b.yaml").write_text(yaml.dump(data_b))

        result = load_queries_for_document(tmp_path, "doc_a")
        assert len(result) == 1
        assert result[0].id == "a1"

    def test_nonexistent_document_returns_empty(self, tmp_path: Path) -> None:
        assert load_queries_for_document(tmp_path, "nonexistent") == []


class TestResolveQueries:
    def test_resolves_to_subtree_span(self) -> None:
        gold = _gold_annotation("test_doc")
        query = AnnotatedQuery(
            id="q1",
            text="What does clause 1 say?",
            document_id="test_doc",
            category="clause_lookup",
            relevant_clauses=[RelevantSection(identifier="1", relevance=3, description="d")],
        )
        resolved = resolve_queries([query], {"test_doc": gold})
        assert len(resolved) == 1
        section = resolved[0].relevant_sections[0]
        # "1"'s own span is [0, 300) but its child "1.1" extends to 600, so the
        # subtree span used for relevance must cover the child too.
        assert (section.char_start, section.char_end) == (0, 600)
        assert section.grade == RelevanceGrade.EXACT
        assert section.document_id == "test_doc"
        assert section.section_id == "1"

    def test_leaf_clause_subtree_equals_own_span(self) -> None:
        gold = _gold_annotation("test_doc")
        query = AnnotatedQuery(
            id="q1",
            text="text",
            document_id="test_doc",
            category="x",
            relevant_clauses=[RelevantSection(identifier="1.1", relevance=2)],
        )
        resolved = resolve_queries([query], {"test_doc": gold})
        section = resolved[0].relevant_sections[0]
        assert (section.char_start, section.char_end) == (300, 600)

    def test_jurisdiction_inferred_from_document_prefix(self) -> None:
        gold = _gold_annotation("us_msa")
        query = AnnotatedQuery(
            id="q1",
            text="text",
            document_id="us_msa",
            category="x",
            relevant_clauses=[RelevantSection(identifier="1", relevance=1)],
        )
        resolved = resolve_queries([query], {"us_msa": gold})
        assert resolved[0].jurisdiction == Jurisdiction.US

    def test_unknown_identifier_raises_naming_query_document_and_identifier(self) -> None:
        gold = _gold_annotation("test_doc")
        query = AnnotatedQuery(
            id="uk_sa_q7",
            text="text",
            document_id="test_doc",
            category="x",
            relevant_clauses=[RelevantSection(identifier="99.9", relevance=3)],
        )
        with pytest.raises(QueryError) as exc_info:
            resolve_queries([query], {"test_doc": gold})

        message = str(exc_info.value)
        assert "uk_sa_q7" in message
        assert "test_doc" in message
        assert "99.9" in message

    def test_document_with_no_gold_raises(self) -> None:
        query = AnnotatedQuery(
            id="q1",
            text="text",
            document_id="missing_doc",
            category="x",
            relevant_clauses=[RelevantSection(identifier="1", relevance=1)],
        )
        with pytest.raises(QueryError, match="missing_doc"):
            resolve_queries([query], {})

    def test_no_relevant_clauses_raises(self) -> None:
        gold = _gold_annotation("test_doc")
        query = AnnotatedQuery(
            id="q1",
            text="text",
            document_id="test_doc",
            category="x",
            relevant_clauses=[],
        )
        with pytest.raises(QueryError, match="q1"):
            resolve_queries([query], {"test_doc": gold})

    def test_invalid_relevance_raises(self) -> None:
        gold = _gold_annotation("test_doc")
        query = AnnotatedQuery(
            id="q1",
            text="text",
            document_id="test_doc",
            category="x",
            relevant_clauses=[RelevantSection(identifier="1", relevance=0)],
        )
        with pytest.raises(QueryError):
            resolve_queries([query], {"test_doc": gold})

    def test_multiple_queries_resolved_independently(self) -> None:
        gold = _gold_annotation("test_doc")
        queries = [
            AnnotatedQuery(
                id="q1",
                text="text",
                document_id="test_doc",
                category="x",
                relevant_clauses=[RelevantSection(identifier="1", relevance=3)],
            ),
            AnnotatedQuery(
                id="q2",
                text="text",
                document_id="test_doc",
                category="x",
                relevant_clauses=[RelevantSection(identifier="1.1", relevance=2)],
            ),
        ]
        resolved = resolve_queries(queries, {"test_doc": gold})
        assert [q.id for q in resolved] == ["q1", "q2"]
