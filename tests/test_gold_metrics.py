"""Unit tests for legal_rag_eval.metrics.gold.

Entirely synthetic: no dependency on real gold/*.json files or on lexichunk being
installed. A small fake contract, a hand-built GoldAnnotation, and hand-built ChunkSets
(with char_start/char_end already attached, as legal_rag_eval.chunking.pipeline.attach_spans
would do) exercise every metric via character-span overlap only.
"""

from __future__ import annotations

import hashlib

import pytest

from legal_rag_eval.gold import (
    GoldAnnotation,
    GoldClause,
    GoldCrossReference,
    GoldDefinedTerm,
    sanitize,
)
from legal_rag_eval.metrics.gold import (
    aggregate_gold_metrics,
    chunk_size_cv,
    clause_fragmentation_rate,
    compute_gold_structural_metrics,
    cross_reference_metrics,
    definition_attachment_recall,
    heading_metrics,
    normalize_identifier,
    sub_clause_grouping_rate,
    top_level_over_merge_rate,
)
from legal_rag_eval.models import (
    Chunk,
    ChunkSet,
    Document,
    DocumentType,
    GoldStructuralMetrics,
    Jurisdiction,
    StrategyName,
)

# -- Fixture document ----------------------------------------------------------------------
#
# A small fake contract with clauses 1, 1.1, 1.2, 2, 2.1. Clause 1 defines
# "Confidential Information"; clause 1.1 uses the term; clause 2 cross-references clause 1.1.

DOC_TEXT = (
    "1. DEFINITIONS\n"
    "Confidential Information means any information disclosed by a party.\n"
    "1.1 Confidentiality Obligation\n"
    "The receiving party shall keep Confidential Information secret.\n"
    "1.2 Exceptions\n"
    "This obligation does not apply to information already public.\n"
    "2. TERM\n"
    "This Agreement continues until terminated as set out in clause 1.1.\n"
    "2.1 Termination\n"
    "Either party may terminate this Agreement on notice.\n"
)

IDX_1 = DOC_TEXT.index("1. DEFINITIONS")
IDX_11 = DOC_TEXT.index("1.1 Confidentiality Obligation")
IDX_12 = DOC_TEXT.index("1.2 Exceptions")
IDX_2 = DOC_TEXT.index("2. TERM")
IDX_21 = DOC_TEXT.index("2.1 Termination")
DOC_END = len(DOC_TEXT)

DEF_START = DOC_TEXT.index("Confidential Information means")
DEF_END = DEF_START + len("Confidential Information means any information disclosed by a party.")

XREF_START = DOC_TEXT.index("clause 1.1")


def _sha256(text: str) -> str:
    return hashlib.sha256(sanitize(text).encode("utf-8")).hexdigest()


def _clause(
    identifier: str, level: int, parent: str | None, start: int, end: int, *, top: bool
) -> GoldClause:
    return GoldClause(
        identifier=identifier,
        level=level,
        parent=parent,
        char_start=start,
        char_end=end,
        is_top_level=top,
        heading=identifier,
    )


CLAUSE_1 = _clause("1", 1, None, IDX_1, IDX_11, top=True)
CLAUSE_11 = _clause("1.1", 2, "1", IDX_11, IDX_12, top=False)
CLAUSE_12 = _clause("1.2", 2, "1", IDX_12, IDX_2, top=False)
CLAUSE_2 = _clause("2", 1, None, IDX_2, IDX_21, top=True)
CLAUSE_21 = _clause("2.1", 2, "2", IDX_21, DOC_END, top=False)

DEFINED_TERM = GoldDefinedTerm(
    term="Confidential Information", definition_start=DEF_START, definition_end=DEF_END
)

XREF = GoldCrossReference(
    raw_text="clause 1.1", char_start=XREF_START, target_identifier="1.1", kind="internal_clause"
)
EXTERNAL_XREF = GoldCrossReference(
    raw_text="the Data Protection Act",
    char_start=0,
    target_identifier=None,
    kind="external_statute",
)


def _make_gold(
    *,
    defined_terms: tuple[GoldDefinedTerm, ...] = (DEFINED_TERM,),
    cross_references: tuple[GoldCrossReference, ...] = (XREF, EXTERNAL_XREF),
) -> GoldAnnotation:
    return GoldAnnotation(
        document_id="doc_a",
        source_file="doc_a.txt",
        text_sha256=_sha256(DOC_TEXT),
        n_chars=len(sanitize(DOC_TEXT)),
        numbering_style="uk_decimal",
        clauses=(CLAUSE_1, CLAUSE_11, CLAUSE_12, CLAUSE_2, CLAUSE_21),
        defined_terms=defined_terms,
        cross_references=cross_references,
    )


GOLD = _make_gold()

DOCUMENT = Document(
    id="doc_a",
    text=DOC_TEXT,
    jurisdiction=Jurisdiction.UK,
    document_type=DocumentType.SERVICE_AGREEMENT,
    source="doc_a.txt",
)


def _chunk(
    index: int,
    char_start: int | None,
    char_end: int | None,
    *,
    strategy: StrategyName = StrategyName.FIXED_SIZE,
    text: str | None = None,
    metadata: dict[str, object] | None = None,
) -> Chunk:
    if text is None:
        if char_start is not None and char_end is not None:
            text = DOC_TEXT[char_start:char_end]
        else:
            text = "?"
    chunk = Chunk(
        id=f"{strategy.value}_doc_a_{index}",
        text=text,
        document_id="doc_a",
        strategy=strategy,
        index=index,
        metadata=metadata or {},
    )
    return chunk.with_span(char_start, char_end)


def _chunk_set(chunks: list[Chunk], strategy: StrategyName = StrategyName.FIXED_SIZE) -> ChunkSet:
    return ChunkSet(
        strategy=strategy, document_id="doc_a", chunks=tuple(chunks), elapsed_seconds=0.0
    )


# -- Clause-aligned ("perfect") and whole-document chunk sets, used across several tests ----


def _perfect_chunk_set() -> ChunkSet:
    """One chunk per leaf clause, spans matching exactly."""
    return _chunk_set(
        [
            _chunk(0, IDX_11, IDX_12),
            _chunk(1, IDX_12, IDX_2),
            _chunk(2, IDX_21, DOC_END),
        ]
    )


def _whole_document_chunk_set() -> ChunkSet:
    return _chunk_set([_chunk(0, 0, DOC_END)])


# -- Circularity regression ------------------------------------------------------------------


class TestCircularityRegression:
    def test_perfect_clause_aligned_chunking_is_not_fragmented(self) -> None:
        assert clause_fragmentation_rate(_perfect_chunk_set(), GOLD) == 0.0

    def test_splitting_a_leaf_clause_produces_a_nonzero_computed_rate(self) -> None:
        """The regression this metric exists to catch: fragmentation is computed from gold
        spans, not always zero. Splitting clause 1.1 in half must move the number."""
        split_point = IDX_11 + (IDX_12 - IDX_11) // 2
        split_chunk_set = _chunk_set(
            [
                _chunk(0, IDX_11, split_point),
                _chunk(1, split_point, IDX_12),
                _chunk(2, IDX_12, IDX_2),
                _chunk(3, IDX_21, DOC_END),
            ]
        )
        rate = clause_fragmentation_rate(split_chunk_set, GOLD)
        assert rate != 0.0
        assert rate == pytest.approx(1 / 3)

    def test_over_merge_pair_the_old_harness_could_not_express(self) -> None:
        """A single whole-document chunk is 'perfect' by the old (LexiChunk-derived)
        fragmentation measure but is the worst possible case for over-merging."""
        whole = _whole_document_chunk_set()
        assert clause_fragmentation_rate(whole, GOLD) == 0.0
        assert top_level_over_merge_rate(whole, GOLD) == 1.0

    def test_perfect_clause_aligned_chunking_has_no_over_merge(self) -> None:
        perfect = _perfect_chunk_set()
        assert clause_fragmentation_rate(perfect, GOLD) == 0.0
        assert top_level_over_merge_rate(perfect, GOLD) == 0.0


# -- Fragmentation edge cases -----------------------------------------------------------------


class TestClauseFragmentationRate:
    def test_skips_leaf_clauses_shorter_than_20_chars(self) -> None:
        tiny_clause = _clause("9", 1, None, 0, 5, top=True)  # 5 chars, no chunks cover it
        gold = GoldAnnotation(
            document_id="doc_a",
            source_file="doc_a.txt",
            text_sha256=_sha256(DOC_TEXT),
            n_chars=len(sanitize(DOC_TEXT)),
            numbering_style="uk_decimal",
            clauses=(tiny_clause,),
            defined_terms=(),
            cross_references=(),
        )
        empty_chunk_set = _chunk_set([])
        assert clause_fragmentation_rate(empty_chunk_set, gold) == 0.0

    def test_no_chunks_means_all_eligible_clauses_fragmented(self) -> None:
        assert clause_fragmentation_rate(_chunk_set([]), GOLD) == 1.0


# -- Unlocated chunks --------------------------------------------------------------------------


class TestUnlocatedChunks:
    def test_unlocated_chunk_lowers_localization_rate_and_is_excluded_from_over_merge(
        self,
    ) -> None:
        chunk_set = _chunk_set(
            [
                _chunk(0, 0, DOC_END),  # whole doc, over-merges
                _chunk(1, None, None, text="could not be found anywhere"),
            ]
        )
        metrics = compute_gold_structural_metrics(chunk_set, DOCUMENT, GOLD)
        assert metrics.chunk_count == 2
        assert metrics.located_chunks == 1
        assert metrics.localization_rate == pytest.approx(0.5)
        # If the unlocated chunk were (wrongly) included in the denominator this would be 0.5.
        assert metrics.top_level_over_merge_rate == 1.0

    def test_unlocated_chunk_never_counts_as_a_grouping_success(self) -> None:
        chunk_set = _chunk_set([_chunk(0, None, None, text="unlocated")])
        assert sub_clause_grouping_rate(chunk_set, GOLD) == 0.0


# -- Sub-clause grouping (informational) --------------------------------------------------------


class TestSubClauseGroupingRate:
    def test_chunk_containing_two_siblings_counts_as_grouping(self) -> None:
        chunk_set = _chunk_set([_chunk(0, IDX_11, IDX_2)])  # covers 1.1 and 1.2 fully
        assert sub_clause_grouping_rate(chunk_set, GOLD) == 1.0

    def test_chunk_containing_only_one_sibling_does_not_count(self) -> None:
        chunk_set = _chunk_set([_chunk(0, IDX_11, IDX_12)])
        assert sub_clause_grouping_rate(chunk_set, GOLD) == 0.0


# -- Heading metrics -----------------------------------------------------------------------------


class TestHeadingMetrics:
    def test_no_section_hierarchy_metadata_is_not_applicable(self) -> None:
        chunk_set = _perfect_chunk_set()
        assert heading_metrics(chunk_set, GOLD) == (None, None, len(GOLD.clauses))

    def test_correct_and_wrong_claims(self) -> None:
        chunk_set = _chunk_set(
            [
                _chunk(0, IDX_11, IDX_12, metadata={"section_hierarchy": "1 > 1.1"}),  # correct
                _chunk(
                    1, IDX_12, IDX_2, metadata={"section_hierarchy": "1 > 1.1"}
                ),  # actually clause 1.2 -> wrong claim
                _chunk(2, IDX_21, DOC_END),  # no heading metadata at all
            ]
        )
        recall, precision, n_gold_headings = heading_metrics(chunk_set, GOLD)
        assert precision == pytest.approx(0.5)
        assert n_gold_headings == 3  # clauses 1.1, 1.2, 2.1 are each overlapped by >=50 chars
        assert recall == pytest.approx(1 / 3)


# -- Cross-reference metrics ----------------------------------------------------------------------


class TestCrossReferenceMetrics:
    def test_no_cross_reference_metadata_is_not_applicable(self) -> None:
        chunk_set = _perfect_chunk_set()
        recall, precision, n_gold_cross_refs = cross_reference_metrics(chunk_set, GOLD)
        assert (recall, precision) == (None, None)
        assert n_gold_cross_refs == 1  # the external ref is excluded

    def test_recall_precision_and_external_exclusion(self) -> None:
        chunk_set = _chunk_set(
            [
                _chunk(
                    0,
                    IDX_2,
                    IDX_21,
                    metadata={"cross_references": [{"raw_text": "clause 1.1", "target": "1.1"}]},
                ),
                _chunk(
                    1,
                    IDX_11,
                    IDX_12,
                    metadata={"cross_references": [{"raw_text": "clause 9.9", "target": "9.9"}]},
                ),
            ]
        )
        recall, precision, n_gold_cross_refs = cross_reference_metrics(chunk_set, GOLD)
        assert n_gold_cross_refs == 1
        assert recall == 1.0
        assert precision == pytest.approx(0.5)


# -- Definition attachment ------------------------------------------------------------------------


class TestDefinitionAttachmentRecall:
    def test_use_without_definition_in_chunk_or_metadata_is_not_attached(self) -> None:
        chunk_set = _chunk_set([_chunk(0, IDX_11, IDX_12)])
        recall, n_uses = definition_attachment_recall(chunk_set, DOCUMENT, GOLD)
        assert n_uses == 1
        assert recall == 0.0

    def test_use_with_definition_span_inside_chunk_is_attached(self) -> None:
        chunk_set = _chunk_set([_chunk(0, IDX_1, IDX_12)])  # covers clause 1 and clause 1.1
        recall, n_uses = definition_attachment_recall(chunk_set, DOCUMENT, GOLD)
        assert n_uses == 1
        assert recall == 1.0

    def test_bare_list_of_term_names_does_not_count(self) -> None:
        """Regression test: the superseded metric granted a free pass for a bare
        `defined_terms` list of names. This module must not repeat that."""
        chunk_set = _chunk_set(
            [_chunk(0, IDX_11, IDX_12, metadata={"defined_terms": ["Confidential Information"]})]
        )
        recall, n_uses = definition_attachment_recall(chunk_set, DOCUMENT, GOLD)
        assert n_uses == 1
        assert recall == 0.0

    def test_definitions_context_metadata_counts_as_attached(self) -> None:
        chunk_set = _chunk_set(
            [
                _chunk(
                    0,
                    IDX_11,
                    IDX_12,
                    metadata={
                        "defined_terms_context": {
                            "Confidential Information": "any information disclosed by a party."
                        }
                    },
                )
            ]
        )
        recall, n_uses = definition_attachment_recall(chunk_set, DOCUMENT, GOLD)
        assert n_uses == 1
        assert recall == 1.0

    def test_no_uses_is_vacuously_perfect(self) -> None:
        gold_no_terms = _make_gold(defined_terms=())
        chunk_set = _chunk_set([_chunk(0, IDX_11, IDX_12)])
        recall, n_uses = definition_attachment_recall(chunk_set, DOCUMENT, gold_no_terms)
        assert n_uses == 0
        assert recall == 1.0


# -- chunk_size_cv -----------------------------------------------------------------------------


class TestChunkSizeCV:
    def test_fewer_than_two_chunks_is_zero(self) -> None:
        assert chunk_size_cv(_chunk_set([_chunk(0, 0, DOC_END)])) == 0.0
        assert chunk_size_cv(_chunk_set([])) == 0.0

    def test_uniform_chunks_have_zero_cv(self) -> None:
        chunk_set = _chunk_set([_chunk(0, 0, 10, text="a" * 10), _chunk(1, 10, 20, text="b" * 10)])
        assert chunk_size_cv(chunk_set) == 0.0

    def test_nonuniform_chunks_have_positive_cv(self) -> None:
        chunk_set = _chunk_set([_chunk(0, 0, 5, text="a" * 5), _chunk(1, 5, 25, text="b" * 20)])
        assert chunk_size_cv(chunk_set) > 0.0


# -- normalize_identifier --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("clause 12.3", "12.3"),
        ("Clause 12.3.", "12.3"),
        ("article 6", "article_6"),
        ("Article 6", "article_6"),
        ("Section 4.02", "section_4.02"),
        ("SCHEDULE 1", "schedule_1"),
        ("paragraph 7", "7"),
        ("1.1", "1.1"),
        ("article_4", "article_4"),
        ("  1.1  ", "1.1"),
        # Roman numerals: US contracts number articles "Article VIII"; gold spells the
        # same clause "article_8". A bare roman numeral must convert to arabic whether
        # it comes from a "keep"-prefixed word (article/section/schedule), a
        # "strip"-prefixed word (clause/paragraph), or with no word at all.
        ("Article VIII", "article_8"),
        ("ARTICLE I", "article_1"),
        ("article iv", "article_4"),
        ("Section IX", "section_9"),
        ("Schedule II", "schedule_2"),
        ("clause vii", "7"),
        ("VIII", "8"),
        # A roman numeral immediately followed by a lettered romanette sub-clause.
        ("Article VIII(a)", "article_8(a)"),
        # "Exhibit" is not a gold prefix word (gold/SCHEMA.md has no "exhibit_"
        # identifiers -- exhibits are always external attachments, kind
        # "external_document", with no internal target to resolve), and a bare
        # letter like "A" is not a roman numeral, so this must pass through
        # unfolded rather than being misread as a number.
        ("Exhibit A", "exhibit a"),
        # A decimal number contains no roman-numeral letters, so it must pass through
        # unaffected, and a bare lettered romanette names a sub-clause, not a number.
        ("Section 7.02", "section_7.02"),
        ("(a)", "(a)"),
    ],
)
def test_normalize_identifier(raw: str, expected: str) -> None:
    assert normalize_identifier(raw) == expected


# -- aggregate_gold_metrics ---------------------------------------------------------------------


def _dummy_metrics(
    document_id: str, *, fragmentation: float, heading_recall: float | None
) -> GoldStructuralMetrics:
    return GoldStructuralMetrics(
        strategy=StrategyName.FIXED_SIZE,
        document_id=document_id,
        chunk_count=1,
        avg_chunk_chars=100.0,
        chunk_size_cv=0.0,
        located_chunks=1,
        localization_rate=1.0,
        localization_coverage=1.0,
        n_leaf_clauses=1,
        clause_fragmentation_rate=fragmentation,
        n_top_level_clauses=1,
        top_level_over_merge_rate=0.0,
        sub_clause_grouping_rate=0.0,
        heading_recall=heading_recall,
        heading_precision=heading_recall,
        n_gold_headings=1,
        n_definition_uses=0,
        definition_attachment_recall=1.0,
        n_gold_cross_refs=0,
        xref_target_recall=None,
        xref_target_precision=None,
    )


class TestAggregateGoldMetrics:
    def test_empty_sequence_is_empty_dict(self) -> None:
        assert aggregate_gold_metrics([]) == {}

    def test_averages_numeric_fields(self) -> None:
        metrics = [
            _dummy_metrics("doc_a", fragmentation=0.2, heading_recall=0.5),
            _dummy_metrics("doc_b", fragmentation=0.4, heading_recall=0.7),
        ]
        aggregated = aggregate_gold_metrics(metrics)
        assert aggregated["clause_fragmentation_rate"] == pytest.approx(0.3)
        assert aggregated["heading_recall"] == pytest.approx(0.6)
        assert aggregated["chunk_count"] == pytest.approx(1.0)

    def test_none_in_any_document_propagates(self) -> None:
        metrics = [
            _dummy_metrics("doc_a", fragmentation=0.2, heading_recall=0.5),
            _dummy_metrics("doc_b", fragmentation=0.4, heading_recall=None),
        ]
        aggregated = aggregate_gold_metrics(metrics)
        assert aggregated["heading_recall"] is None
        assert aggregated["xref_target_recall"] is None
        # Fields with no None anywhere still average normally.
        assert aggregated["clause_fragmentation_rate"] == pytest.approx(0.3)


# -- compute_gold_structural_metrics (integration of the above) --------------------------------


class TestComputeGoldStructuralMetrics:
    def test_full_computation_on_perfect_chunk_set(self) -> None:
        metrics = compute_gold_structural_metrics(_perfect_chunk_set(), DOCUMENT, GOLD)
        assert metrics.strategy == StrategyName.FIXED_SIZE
        assert metrics.document_id == "doc_a"
        assert metrics.chunk_count == 3
        assert metrics.located_chunks == 3
        assert metrics.localization_rate == 1.0
        assert metrics.clause_fragmentation_rate == 0.0
        assert metrics.top_level_over_merge_rate == 0.0
        assert metrics.n_leaf_clauses == 3
        assert metrics.n_top_level_clauses == 2
        assert metrics.heading_recall is None
        assert metrics.xref_target_recall is None
