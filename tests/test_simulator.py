"""Tests for RetrievalSimulator and query loading."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from scaffolder.models import (
    AnnotatedQuery,
    Chunk,
    EmbeddingModelName,
    Jurisdiction,
    RelevanceGrade,
    RelevantSection,
    StrategyName,
)
from scaffolder.retrieval import IndexRegistry, RetrievalSimulator
from scaffolder.retrieval.simulator import _is_relevant, get_relevance_grade

if TYPE_CHECKING:
    from pathlib import Path

DIM = 64


def _random_embeddings(n: int, dim: int) -> np.ndarray:
    rng = np.random.default_rng(42)
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


def _make_chunks(n: int) -> list[Chunk]:
    return [
        Chunk(
            id=f"test_{i}",
            text=f"This is chunk number {i} about legal terms and obligations.",
            document_id="doc1",
            strategy=StrategyName.LEXICHUNK,
            index=i,
        )
        for i in range(n)
    ]


SAMPLE_QUERY = AnnotatedQuery(
    id="q1",
    text="What are the obligations?",
    document_ids=["doc1"],
    jurisdiction=Jurisdiction.UK,
    relevant_sections=(
        RelevantSection(
            document_id="doc1",
            section_id="clause_2",
            text_snippet="legal terms and obligations",
            grade=RelevanceGrade.EXACT,
        ),
    ),
    category="definition_lookup",
)


class TestIsRelevant:
    def test_substring_match(self) -> None:
        chunk = Chunk(
            id="c1",
            text="This contains legal terms and obligations here.",
            document_id="doc1",
            strategy=StrategyName.LEXICHUNK,
            index=0,
        )
        assert _is_relevant(chunk, SAMPLE_QUERY.relevant_sections)

    def test_section_id_match(self) -> None:
        chunk = Chunk(
            id="c1",
            text="2. The parties agree to the following.",
            document_id="doc1",
            strategy=StrategyName.LEXICHUNK,
            index=0,
        )
        assert _is_relevant(chunk, SAMPLE_QUERY.relevant_sections)

    def test_non_matching_returns_false(self) -> None:
        chunk = Chunk(
            id="c1",
            text="Completely unrelated text about weather.",
            document_id="doc1",
            strategy=StrategyName.LEXICHUNK,
            index=0,
        )
        assert not _is_relevant(chunk, SAMPLE_QUERY.relevant_sections)

    def test_wrong_document_returns_false(self) -> None:
        chunk = Chunk(
            id="c1",
            text="legal terms and obligations",
            document_id="doc2",
            strategy=StrategyName.LEXICHUNK,
            index=0,
        )
        assert not _is_relevant(chunk, SAMPLE_QUERY.relevant_sections)

    def test_section_id_requires_a_heading_not_an_incidental_number(self) -> None:
        section = RelevantSection(
            document_id="doc1",
            section_id="clause_1",
            text_snippet="",
            grade=RelevanceGrade.EXACT,
        )

        for text in ("The fee is $1 per user.", "10. Payment terms apply."):
            chunk = Chunk(
                id=text,
                text=text,
                document_id="doc1",
                strategy=StrategyName.LEXICHUNK,
                index=0,
            )
            assert not _is_relevant(chunk, (section,))

    def test_section_kind_does_not_match_a_different_heading_namespace(self) -> None:
        clause = RelevantSection(
            document_id="doc1",
            section_id="clause_1",
            text_snippet="",
            grade=RelevanceGrade.EXACT,
        )
        schedule = RelevantSection(
            document_id="doc1",
            section_id="schedule_1",
            text_snippet="",
            grade=RelevanceGrade.EXACT,
        )
        chunk = Chunk(
            id="c1",
            text="Schedule 1. Pricing",
            document_id="doc1",
            strategy=StrategyName.LEXICHUNK,
            index=0,
        )

        assert not _is_relevant(chunk, (clause,))
        assert _is_relevant(chunk, (schedule,))

    def test_binary_and_graded_matching_reject_partial_description_overlap(self) -> None:
        section = RelevantSection(
            document_id="doc1",
            section_id="clause_obligations",
            text_snippet="service obligations in detail",
            grade=RelevanceGrade.SAME_SECTION,
        )
        chunk = Chunk(
            id="c1",
            text="The service obligations are detailed elsewhere.",
            document_id="doc1",
            strategy=StrategyName.LEXICHUNK,
            index=0,
        )

        assert not _is_relevant(chunk, (section,))
        assert get_relevance_grade(chunk, (section,)) == RelevanceGrade.IRRELEVANT

    def test_irrelevant_grade_does_not_make_a_binary_hit(self) -> None:
        section = RelevantSection(
            document_id="doc1",
            section_id="clause_1",
            text_snippet="answer passage",
            grade=RelevanceGrade.IRRELEVANT,
        )
        chunk = Chunk(
            id="c1",
            text="This contains the answer passage.",
            document_id="doc1",
            strategy=StrategyName.LEXICHUNK,
            index=0,
        )

        assert not _is_relevant(chunk, (section,))
        assert get_relevance_grade(chunk, (section,)) == RelevanceGrade.IRRELEVANT


class TestGetRelevanceGrade:
    def test_exact_match(self) -> None:
        chunk = Chunk(
            id="c1",
            text="This contains legal terms and obligations here.",
            document_id="doc1",
            strategy=StrategyName.LEXICHUNK,
            index=0,
        )
        grade = get_relevance_grade(chunk, SAMPLE_QUERY.relevant_sections)
        assert grade == RelevanceGrade.EXACT

    def test_irrelevant(self) -> None:
        chunk = Chunk(
            id="c1",
            text="Unrelated text.",
            document_id="doc1",
            strategy=StrategyName.LEXICHUNK,
            index=0,
        )
        grade = get_relevance_grade(chunk, SAMPLE_QUERY.relevant_sections)
        assert grade == RelevanceGrade.IRRELEVANT


class TestLoadQueriesFromYAML:
    def test_loads_from_yaml(self, tmp_path: Path) -> None:
        yaml_content = """\
document_id: test_doc
queries:
  - id: test_q1
    text: "What is the definition?"
    failure_mode: clause_fragmentation
    relevant_sections:
      - section_id: "clause_1"
        relevance: 3
        description: "Main definition clause"
"""
        (tmp_path / "test_doc.yaml").write_text(yaml_content)
        from scaffolder.retrieval.simulator import load_queries_from_yaml

        queries = load_queries_from_yaml(str(tmp_path))
        assert len(queries) == 1
        assert queries[0].id == "test_q1"
        assert queries[0].relevant_sections[0].grade == RelevanceGrade.EXACT

    def test_empty_dir_returns_empty(self, tmp_path: Path) -> None:
        from scaffolder.retrieval.simulator import load_queries_from_yaml

        queries = load_queries_from_yaml(str(tmp_path))
        assert queries == []


class _FakeEmbeddingPipeline:
    """Fake embedding pipeline for testing."""

    def embed_texts(self, texts: list[str], model: EmbeddingModelName) -> np.ndarray:
        rng = np.random.default_rng(hash(model.value) % 2**31)
        vecs = rng.standard_normal((len(texts), DIM)).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / norms


class TestRetrievalSimulator:
    def test_run_produces_correct_count(self) -> None:
        chunks = _make_chunks(10)
        embs = _random_embeddings(10, DIM)

        registry = IndexRegistry()
        registry.build(StrategyName.LEXICHUNK, EmbeddingModelName.MINILM, chunks, embs)
        registry.build(StrategyName.RCTS, EmbeddingModelName.MINILM, chunks, embs)

        sim = RetrievalSimulator(
            index_registry=registry,
            embedding_pipeline=_FakeEmbeddingPipeline(),  # type: ignore[arg-type]
        )
        results = sim.run(
            queries=[SAMPLE_QUERY],
            strategies=[StrategyName.LEXICHUNK, StrategyName.RCTS],
            models=[EmbeddingModelName.MINILM],
            k=5,
        )
        # 1 query x 2 strategies x 1 model = 2 results
        assert len(results) == 2

    def test_run_results_have_hits(self) -> None:
        chunks = _make_chunks(10)
        embs = _random_embeddings(10, DIM)

        registry = IndexRegistry()
        registry.build(StrategyName.LEXICHUNK, EmbeddingModelName.MINILM, chunks, embs)

        sim = RetrievalSimulator(
            index_registry=registry,
            embedding_pipeline=_FakeEmbeddingPipeline(),  # type: ignore[arg-type]
        )
        results = sim.run(
            queries=[SAMPLE_QUERY],
            strategies=[StrategyName.LEXICHUNK],
            models=[EmbeddingModelName.MINILM],
            k=5,
        )
        assert len(results) == 1
        assert len(results[0].hits) == 5
