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
    """Chunking strategy identifier.

    RCTS appears three times on purpose. The audit of this harness showed that the
    headline retrieval gap was largely a chunk-size artefact: LexiChunk emits ~790-char
    chunks and the RCTS baseline was configured at 512. A baseline is only informative
    when its size is stated, so every RCTS variant carries its chunk size in its name and
    ``rcts_1024`` exists specifically as the size-matched control.
    """

    LEXICHUNK = "lexichunk"
    LEXICHUNK_CONTEXTUAL = "lexichunk_contextual"
    RCTS = "rcts"  # legacy, unsized label; kept so old result JSON still loads
    RCTS_512 = "rcts_512"
    RCTS_1024 = "rcts_1024"
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
    """A single chunk produced by any strategy.

    ``char_start``/``char_end`` are the chunk's span in the **sanitised** document text.
    They are filled in by :class:`~scaffolder.chunking.pipeline.ChunkingPipeline` using
    :func:`scaffolder.gold.locate_chunks`, which locates every strategy's chunks the same
    way — by matching text against the document. They are deliberately NOT taken from a
    chunker's self-reported offsets: those are a property of the chunker under test, and
    one LexiChunk build reports them incorrectly. ``None`` means the chunk could not be
    located (it is not a contiguous span of the source), which is itself reported.
    """

    id: str  # unique: f"{strategy}_{doc_id}_{index}"
    text: str
    document_id: str
    strategy: StrategyName
    index: int  # position within the chunk set
    char_count: int = field(init=False)
    metadata: dict[str, object] = field(default_factory=dict, hash=False)
    # metadata may include: clause_type, confidence, defined_terms,
    # cross_references, section_hierarchy, embedded_text (contextual)
    char_start: int | None = None
    char_end: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "char_count", len(self.text))

    @property
    def located(self) -> bool:
        """True when this chunk has a span in the source document."""
        return self.char_start is not None and self.char_end is not None

    def with_span(self, char_start: int | None, char_end: int | None) -> Chunk:
        """Return a copy of this chunk carrying the given document span."""
        return Chunk(
            id=self.id,
            text=self.text,
            document_id=self.document_id,
            strategy=self.strategy,
            index=self.index,
            metadata=self.metadata,
            char_start=char_start,
            char_end=char_end,
        )


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
class LegacyStructuralMetrics:
    """The original structural metrics. Retained for provenance only — DO NOT report.

    Every metric in this container derives its ground truth from LexiChunk's own parse of
    the document (``metrics.structural.get_ground_truth`` runs ``LegalChunker``). LexiChunk
    is therefore graded against itself and scores perfectly by construction, while the
    baselines' scores move whenever LexiChunk's parser changes even though the baselines
    have not. See ``docs/metrics.md`` for the measurements that replace these.
    """

    strategy: StrategyName
    document_id: str
    clause_fragmentation_rate: float  # 0.0 = no fragmentation (best)
    definition_preservation_rate: float  # 1.0 = all preserved (best)
    cross_ref_resolution_rate: float  # 1.0 = all resolved (best)
    hierarchy_depth_retained: float  # 1.0 = full depth kept (best)
    chunk_size_cv: float  # coefficient of variation (lower = more uniform)
    chunk_count: int
    avg_chunk_chars: float


#: Backwards-compatible alias so previously exported result JSON and the dashboard keep
#: loading. New code should use :class:`GoldStructuralMetrics`.
StructuralMetrics = LegacyStructuralMetrics


@dataclass(frozen=True, slots=True)
class GoldStructuralMetrics:
    """Structural metrics for one ChunkSet, scored against hand-checked gold annotations.

    Every number here is computed by comparing character spans in the document: the gold
    clause spans in ``gold/<document_id>.json`` against the chunk spans located by
    :func:`scaffolder.gold.locate_chunks`. Nothing is decided by searching for LexiChunk's
    output inside anything, and no metric consults a chunker's own metadata to decide
    whether that chunker succeeded.

    ``None`` means *not applicable to this strategy* (for example, heading recall for a
    splitter that exposes no headings) and must be rendered as "n/a", never as 0.
    """

    strategy: StrategyName
    document_id: str

    # -- how much of the strategy's output could be scored at all
    chunk_count: int
    avg_chunk_chars: float
    chunk_size_cv: float
    located_chunks: int
    localization_rate: float  # located / total chunks
    localization_coverage: float  # located characters / total chunk characters

    # -- segmentation quality
    n_leaf_clauses: int
    clause_fragmentation_rate: float  # lower better; leaf clause not >=80% inside one chunk
    n_top_level_clauses: int
    top_level_over_merge_rate: float  # lower better; chunk spanning >1 top-level clause
    sub_clause_grouping_rate: float  # informational, no direction

    # -- headings (n/a for strategies that expose none)
    heading_recall: float | None
    heading_precision: float | None
    n_gold_headings: int

    # -- definitions
    n_definition_uses: int
    definition_attachment_recall: float

    # -- cross references (n/a for strategies that emit none)
    n_gold_cross_refs: int
    xref_target_recall: float | None
    xref_target_precision: float | None


# -- Retrieval Types --------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RelevantSection:
    """A ground-truth relevant passage for a query, identified by its span.

    ``char_start``/``char_end`` are the subtree span of the gold clause named by
    ``section_id`` in ``gold/<document_id>.json``. Relevance is judged by overlap between
    this span and a retrieved chunk's span — never by looking for ``description`` inside
    the chunk text, which is what the previous judge did and which rewarded long chunks.
    ``description`` survives as documentation for the annotator and is never read by a
    metric.
    """

    document_id: str
    section_id: str  # a gold clause identifier, e.g. "12.3" or "article_6"
    char_start: int
    char_end: int
    grade: RelevanceGrade
    description: str = ""


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
    # All five documents share one index, so drm_hit is True for essentially every query
    # and carries no information on its own. The underlying rate does, so it is kept.
    drm_rate: float = 0.0
    n_relevant_sections: int = 0


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
    structural_metrics: list[GoldStructuralMetrics] = field(default_factory=list)
    legacy_structural_metrics: list[LegacyStructuralMetrics] = field(default_factory=list)
    retrieval_metrics: list[RetrievalMetrics] = field(default_factory=list)
    significance_results: list[SignificanceResult] = field(default_factory=list)
    comparisons: list[object] = field(default_factory=list)
    strategy_results: list[StrategyResult] = field(default_factory=list)
    config: dict[str, object] = field(default_factory=dict)
    seed: int | None = None  # random seed pinned for this run, recorded for reproducibility
    # Provenance: which build of the chunker under test produced these numbers.
    lexichunk_version: str | None = None
    lexichunk_commit: str | None = None


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
