"""Tests for RetrievalSimulator and the load_queries_from_yaml wrapper.

These build synthetic gold/query fixtures under ``tmp_path`` rather than
depending on the real ``gold/*.json`` or ``queries/*.yaml`` (both are being
rewritten concurrently) or on ``lexichunk``.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import yaml

from legal_rag_eval.metrics.retrieval import is_relevant
from legal_rag_eval.models import (
    AnnotatedQuery,
    Chunk,
    EmbeddingModelName,
    Jurisdiction,
    RelevanceGrade,
    RelevantSection,
    StrategyName,
)
from legal_rag_eval.retrieval import IndexRegistry, RetrievalSimulator
from legal_rag_eval.retrieval.simulator import load_queries_from_yaml

if TYPE_CHECKING:
    from pathlib import Path

DIM = 64


def _random_embeddings(n: int, dim: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


def _make_chunks(n: int, document_id: str = "doc1") -> list[Chunk]:
    """Chunks with real, non-overlapping spans so relevance varies across them."""
    chunks = []
    for i in range(n):
        start = i * 100
        chunks.append(
            Chunk(
                id=f"test_{i}",
                text=f"This is chunk number {i} about legal terms and obligations.",
                document_id=document_id,
                strategy=StrategyName.LEXICHUNK,
                index=i,
                char_start=start,
                char_end=start + 100,
            )
        )
    return chunks


# A query whose single relevant section spans the same range as chunk index 2
# (char 200-300) from _make_chunks, so exactly one of the ten chunks matches.
SAMPLE_QUERY = AnnotatedQuery(
    id="q1",
    text="What are the obligations?",
    document_ids=["doc1"],
    jurisdiction=Jurisdiction.UK,
    relevant_sections=(
        RelevantSection(
            document_id="doc1",
            section_id="clause_2",
            char_start=200,
            char_end=300,
            grade=RelevanceGrade.EXACT,
        ),
    ),
    category="definition_lookup",
)


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
        registry.build(StrategyName.RCTS_512, EmbeddingModelName.MINILM, chunks, embs)

        sim = RetrievalSimulator(
            index_registry=registry,
            embedding_pipeline=_FakeEmbeddingPipeline(),  # type: ignore[arg-type]
        )
        results = sim.run(
            queries=[SAMPLE_QUERY],
            strategies=[StrategyName.LEXICHUNK, StrategyName.RCTS_512],
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

    def test_relevant_retrieved_matches_metrics_module(self) -> None:
        """Pin the simulator's relevant_retrieved count to metrics.retrieval.is_relevant.

        A prior version of this module kept its own near-duplicate relevance
        test, so RetrievalResult.relevant_retrieved and the metrics computed
        from it could diverge. Both must now come from exactly the same
        function.
        """
        chunks = _make_chunks(10)  # spans [0,100), [100,200), ..., [900,1000)
        embs = _random_embeddings(10, DIM)

        registry = IndexRegistry()
        registry.build(StrategyName.LEXICHUNK, EmbeddingModelName.MINILM, chunks, embs)

        sim = RetrievalSimulator(
            index_registry=registry,
            embedding_pipeline=_FakeEmbeddingPipeline(),  # type: ignore[arg-type]
        )
        # k=10 so every chunk is retrieved, guaranteeing the one matching chunk
        # (index 2, span [200,300)) is present regardless of embedding order.
        results = sim.run(
            queries=[SAMPLE_QUERY],
            strategies=[StrategyName.LEXICHUNK],
            models=[EmbeddingModelName.MINILM],
            k=10,
        )
        assert len(results) == 1
        result = results[0]

        expected = sum(
            1 for h in result.hits if is_relevant(h.chunk, result.query.relevant_sections)
        )
        assert result.relevant_retrieved == expected
        assert expected == 1  # exactly chunk index 2 overlaps [200, 300)

    def test_min_overlap_chars_threaded_to_relevant_retrieved(self) -> None:
        """A stricter min_overlap_chars than any available overlap must drop all matches.

        Uses a wider 300-char relevant span (chunks are 100 chars each) so that
        no single chunk can ever fully contain it: this keeps the "short
        clause" 50%-of-span fallback from kicking in and masking the intended
        absolute-threshold comparison (see matched_sections' docstring).
        """
        chunks = _make_chunks(10)  # spans [0,100), [100,200), ..., [900,1000)
        embs = _random_embeddings(10, DIM)
        wide_query = AnnotatedQuery(
            id="q_wide",
            text="wide span query",
            document_ids=["doc1"],
            jurisdiction=Jurisdiction.UK,
            relevant_sections=(
                RelevantSection(
                    document_id="doc1",
                    section_id="wide",
                    char_start=250,
                    char_end=550,  # 300 chars; max overlap with any one chunk is 100
                    grade=RelevanceGrade.EXACT,
                ),
            ),
            category="clause_lookup",
        )

        registry = IndexRegistry()
        registry.build(StrategyName.LEXICHUNK, EmbeddingModelName.MINILM, chunks, embs)

        sim = RetrievalSimulator(
            index_registry=registry,
            embedding_pipeline=_FakeEmbeddingPipeline(),  # type: ignore[arg-type]
        )
        lax = sim.run(
            queries=[wide_query],
            strategies=[StrategyName.LEXICHUNK],
            models=[EmbeddingModelName.MINILM],
            k=10,
            min_overlap_chars=50,
        )[0]
        strict = sim.run(
            queries=[wide_query],
            strategies=[StrategyName.LEXICHUNK],
            models=[EmbeddingModelName.MINILM],
            k=10,
            min_overlap_chars=1000,  # far larger than the 100-char max overlap available
        )[0]
        assert lax.relevant_retrieved == 4  # chunks [200,300),[300,400),[400,500),[500,600)
        assert strict.relevant_retrieved == 0


def _write_gold_file(gold_dir: Path, document_id: str) -> None:
    gold_dir.mkdir(parents=True, exist_ok=True)
    gold = {
        "document_id": document_id,
        "source_file": f"{document_id}.txt",
        "text_sha256": "0" * 64,
        "n_chars": 5000,
        "numbering_style": "uk_decimal",
        "clauses": [
            {
                "identifier": "1",
                "level": 1,
                "parent": None,
                "char_start": 0,
                "char_end": 300,
                "is_top_level": True,
                "heading": "Clause 1",
            },
            {
                "identifier": "1.1",
                "level": 2,
                "parent": "1",
                "char_start": 300,
                "char_end": 600,
                "is_top_level": False,
                "heading": "Clause 1.1",
            },
        ],
        "defined_terms": [],
        "cross_references": [],
    }
    (gold_dir / f"{document_id}.json").write_text(json.dumps(gold), encoding="utf-8")


class TestLoadQueriesFromYAML:
    def test_loads_and_resolves_from_yaml(self, tmp_path: Path) -> None:
        document_id = "test_doc"
        queries_dir = tmp_path / "queries"
        gold_dir = tmp_path / "gold"
        queries_dir.mkdir()
        _write_gold_file(gold_dir, document_id)

        yaml_content = {
            "document_id": document_id,
            "queries": [
                {
                    "id": "test_q1",
                    "text": "What is the definition?",
                    "category": "clause_lookup",
                    "relevant_clauses": [
                        {
                            "identifier": "1.1",
                            "relevance": 3,
                            "description": "Main definition clause",
                        }
                    ],
                }
            ],
        }
        (queries_dir / f"{document_id}.yaml").write_text(yaml.dump(yaml_content))

        queries = load_queries_from_yaml(str(queries_dir), str(gold_dir))
        assert len(queries) == 1
        query = queries[0]
        assert query.id == "test_q1"
        assert query.jurisdiction == Jurisdiction.UK  # "test_doc" has no uk_/us_/eu_ prefix -> UK
        section = query.relevant_sections[0]
        assert section.grade == RelevanceGrade.EXACT
        # "1.1" own span is [300, 600); it has no children, so subtree == own span.
        assert (section.char_start, section.char_end) == (300, 600)

    def test_empty_dir_returns_empty(self, tmp_path: Path) -> None:
        queries_dir = tmp_path / "queries"
        gold_dir = tmp_path / "gold"
        queries_dir.mkdir()
        gold_dir.mkdir()

        queries = load_queries_from_yaml(str(queries_dir), str(gold_dir))
        assert queries == []
