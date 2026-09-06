"""Shared data contracts for the LexiChunk evaluation harness.

All models used across the pipeline — documents, chunks, metrics, retrieval
types, and protocol interfaces — are defined here as the single source of truth.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    import numpy.typing as npt


# -- Enums ------------------------------------------------------------------


class Jurisdiction(str, enum.Enum):
    """Legal jurisdiction of a fixture document."""

    UK = "uk"
    US = "us"
    EU = "eu"


class DocumentType(str, enum.Enum):
    """Type of legal document."""

    SERVICE_AGREEMENT = "service_agreement"
    TERMS_CONDITIONS = "terms_conditions"
    MSA = "msa"
    TERMS_OF_SERVICE = "terms_of_service"
    GDPR_EXCERPT = "gdpr_excerpt"


class StrategyName(str, enum.Enum):
    """Chunking strategy identifier."""

    LEXICHUNK = "lexichunk"
    LEXICHUNK_CONTEXTUAL = "lexichunk_contextual"
    RCTS = "rcts"
    SENTENCE_SPLIT = "sentence_split"
    FIXED_SIZE = "fixed_size"


class EmbeddingModelName(str, enum.Enum):
    """Embedding model identifier."""

    MINILM = "all-MiniLM-L6-v2"
    BGE_BASE = "bge-base-en-v1.5"
    VOYAGE_LAW_2 = "voyage-law-2"


class RelevanceGrade(int, enum.Enum):
    """Graded relevance for query annotations."""

    EXACT = 3  # Chunk contains the exact answer passage
    SAME_SECTION = 2  # Chunk is from the correct section
    RELATED = 1  # Chunk is topically related
    IRRELEVANT = 0  # Chunk is not relevant


# -- Document ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Document:
    """A legal document loaded from fixtures."""

    id: str  # e.g. "uk_service_agreement"
    text: str  # full document text
    jurisdiction: Jurisdiction
    document_type: DocumentType
    source: str  # filename, e.g. "uk_service_agreement.txt"
    char_count: int = field(init=False)
    line_count: int = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "char_count", len(self.text))
        object.__setattr__(self, "line_count", self.text.count("\n") + 1)


# -- Chunk ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Chunk:
    """A single chunk produced by any strategy."""

    id: str  # unique: f"{strategy}_{doc_id}_{index}"
    text: str
    document_id: str
    strategy: StrategyName
    index: int  # position within the chunk set
    char_count: int = field(init=False)
    metadata: dict[str, object] = field(default_factory=dict, hash=False)
    # metadata may include: clause_type, confidence, defined_terms,
    # cross_references, section_hierarchy, embedded_text (contextual)

    def __post_init__(self) -> None:
        object.__setattr__(self, "char_count", len(self.text))


# -- ChunkSet ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ChunkSet:
    """All chunks produced by one strategy on one document."""

    strategy: StrategyName
    document_id: str
    chunks: tuple[Chunk, ...]  # immutable sequence
    elapsed_seconds: float  # wall-clock time for chunking

    @property
    def count(self) -> int:
        """Number of chunks in this set."""
        return len(self.chunks)

    @property
    def avg_chunk_size(self) -> float:
        """Average character count per chunk."""
        if not self.chunks:
            return 0.0
        return sum(c.char_count for c in self.chunks) / len(self.chunks)


# -- Strategy Result --------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StrategyResult:
    """Result of running one strategy across ALL documents."""

    strategy: StrategyName
    chunk_sets: tuple[ChunkSet, ...]  # one per document
    total_elapsed_seconds: float


# -- Structural Metrics -----------------------------------------------------


@dataclass(frozen=True, slots=True)
class StructuralMetrics:
    """Structural quality metrics for one ChunkSet."""

    strategy: StrategyName
    document_id: str
    clause_fragmentation_rate: float  # 0.0 = no fragmentation (best)
    definition_preservation_rate: float  # 1.0 = all preserved (best)
    cross_ref_resolution_rate: float  # 1.0 = all resolved (best)
    hierarchy_depth_retained: float  # 1.0 = full depth kept (best)
    chunk_size_cv: float  # coefficient of variation (lower = more uniform)
    chunk_count: int
    avg_chunk_chars: float


# -- Retrieval Types --------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RelevantSection:
    """A ground-truth relevant passage for a query."""

    document_id: str
    section_id: str  # e.g. "clause_5.2"
    text_snippet: str  # first 200 chars for fuzzy matching
    grade: RelevanceGrade


@dataclass(frozen=True, slots=True)
class AnnotatedQuery:
    """A query with human-annotated relevant chunks/sections."""

    id: str
    text: str
    document_ids: list[str]  # which documents this query targets
    jurisdiction: Jurisdiction
    relevant_sections: tuple[RelevantSection, ...]
    category: str  # e.g. "definition_lookup", "clause_search"


@dataclass(frozen=True, slots=True)
class RetrievalHit:
    """A single retrieved chunk with its similarity score."""

    chunk: Chunk
    score: float
    rank: int


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """Result of running one query against one strategy's index."""

    query: AnnotatedQuery
    strategy: StrategyName
    embedding_model: EmbeddingModelName
    hits: tuple[RetrievalHit, ...]  # ordered by rank
    relevant_retrieved: int  # how many of top-k were relevant
    total_relevant: int  # total annotated relevant


# -- Retrieval Metrics ------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RetrievalMetrics:
    """Retrieval quality metrics for one query x one strategy x one model."""

    query_id: str
    strategy: StrategyName
    embedding_model: EmbeddingModelName
    precision_at_1: float
    precision_at_3: float
    precision_at_5: float
    precision_at_10: float
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    recall_at_10: float
    mrr: float
    ndcg_at_10: float
    drm_hit: bool  # Document Retrieval Mismatch: pulled from wrong doc?


# -- Statistical Significance -----------------------------------------------


@dataclass(frozen=True, slots=True)
class SignificanceResult:
    """Result of a paired t-test comparing two strategies."""

    metric_name: str
    strategy_a: StrategyName  # always LexiChunk
    strategy_b: StrategyName  # the baseline
    mean_a: float
    mean_b: float
    improvement_pct: float  # ((mean_a - mean_b) / mean_b) * 100
    t_statistic: float
    p_value: float
    significant: bool  # p < 0.05
    effect_size: float  # Cohen's d
    n_queries: int


# -- Benchmark Result -------------------------------------------------------


@dataclass(slots=True)
class BenchmarkResult:
    """Top-level result container for a full benchmark run."""

    timestamp: str  # ISO 8601
    strategies: list[StrategyName] = field(default_factory=list)
    documents: list[str] = field(default_factory=list)
    models: list[EmbeddingModelName] = field(default_factory=list)
    structural_metrics: list[StructuralMetrics] = field(default_factory=list)
    retrieval_metrics: list[RetrievalMetrics] = field(default_factory=list)
    significance_results: list[SignificanceResult] = field(default_factory=list)
    strategy_results: list[StrategyResult] = field(default_factory=list)
    config: dict[str, object] = field(default_factory=dict)
    seed: int | None = None  # random seed pinned for this run, recorded for reproducibility


# -- Protocols --------------------------------------------------------------


@runtime_checkable
class ChunkingStrategy(Protocol):
    """Interface that every chunking strategy wrapper must implement."""

    name: StrategyName

    def chunk(self, document: Document) -> ChunkSet:
        """Chunk a document, returning a ChunkSet with timing info."""
        ...


@runtime_checkable
class Embedder(Protocol):
    """Interface for embedding adapters (local or API-based)."""

    model_name: EmbeddingModelName
    dimension: int

    def embed_texts(self, texts: Sequence[str]) -> npt.NDArray[np.float32]:
        """Embed a batch of texts. Returns array of shape (n, dimension)."""
        ...
