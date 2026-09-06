"""Unit tests for legal_rag_eval.gold: loading, validation, and chunk localisation.

Pure unit tests -- no dependency on real gold/*.json files or on lexichunk.
Synthetic documents and gold dicts are built inline and written to tmp_path.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from legal_rag_eval.gold import (
    ChunkSpan,
    GoldAnnotation,
    GoldClause,
    GoldCrossReference,
    GoldDefinedTerm,
    GoldError,
    clear_gold_cache,
    load_all_gold,
    load_gold,
    localization_coverage,
    localization_rate,
    locate_chunks,
    overlap_chars,
    sanitize,
    verify_document,
)

# -- Helpers ------------------------------------------------------------------


def _write_gold(gold_dir: Path, document_id: str, data: dict[str, Any]) -> Path:
    gold_dir.mkdir(parents=True, exist_ok=True)
    path = gold_dir / f"{document_id}.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _sha256(text: str) -> str:
    return hashlib.sha256(sanitize(text).encode("utf-8")).hexdigest()


def _base_document() -> str:
    return (
        "1. DEFINITIONS\n"
        "This clause defines terms.\n"
        "1.1 First sub-clause about Widgets.\n"
        "1.2 Second sub-clause about Gadgets.\n"
        "2. TERM\n"
        "This clause is about the term of the agreement, see clause 1.1.\n"
    )


def _build_three_level_gold(text: str) -> dict[str, Any]:
    """A synthetic 3-level clause tree over a document with two top clauses."""
    idx_1 = text.index("1. DEFINITIONS")
    idx_1_body_end = text.index("1.1 First")
    idx_11 = text.index("1.1 First")
    idx_11_end = text.index("1.2 Second")
    idx_12 = text.index("1.2 Second")
    idx_12_end = text.index("2. TERM")
    idx_2 = text.index("2. TERM")

    return {
        "document_id": "doc_a",
        "source_file": "doc_a.txt",
        "text_sha256": _sha256(text),
        "n_chars": len(sanitize(text)),
        "numbering_style": "uk_decimal",
        "clauses": [
            {
                "identifier": "1",
                "level": 1,
                "parent": None,
                "char_start": idx_1,
                "char_end": idx_1_body_end,
                "is_top_level": True,
                "heading": "DEFINITIONS",
            },
            {
                # Own span excludes its child "1.1(a)": just the numbering label.
                "identifier": "1.1",
                "level": 2,
                "parent": "1",
                "char_start": idx_11,
                "char_end": idx_11 + 4,
                "is_top_level": False,
                "heading": "First sub-clause about Widgets.",
            },
            {
                "identifier": "1.1(a)",
                "level": 3,
                "parent": "1.1",
                "char_start": idx_11 + 4,
                "char_end": idx_11_end,
                "is_top_level": False,
                "heading": "First sub-clause about Widgets.",
            },
            {
                "identifier": "1.2",
                "level": 2,
                "parent": "1",
                "char_start": idx_12,
                "char_end": idx_12_end,
                "is_top_level": False,
                "heading": "Second sub-clause about Gadgets.",
            },
            {
                "identifier": "2",
                "level": 1,
                "parent": None,
                "char_start": idx_2,
                "char_end": len(text),
                "is_top_level": True,
                "heading": "TERM",
            },
        ],
        "defined_terms": [],
        "cross_references": [],
    }


# -- sanitize -------------------------------------------------------------


class TestSanitize:
    def test_strips_bom(self) -> None:
        assert sanitize("﻿hello") == "hello"

    def test_normalises_crlf(self) -> None:
        assert sanitize("a\r\nb\r\nc") == "a\nb\nc"

    def test_normalises_lone_cr(self) -> None:
        assert sanitize("a\rb\rc") == "a\nb\nc"

    def test_mixed_line_endings(self) -> None:
        assert sanitize("a\r\nb\rc\nd") == "a\nb\nc\nd"

    def test_nfc_normalisation(self) -> None:
        decomposed = "é"  # "e" + combining acute accent
        assert sanitize(decomposed) == "é"

    def test_idempotent(self) -> None:
        text = "﻿Hello\r\nWorld\rénd"
        once = sanitize(text)
        twice = sanitize(once)
        assert once == twice

    def test_no_bom_no_op(self) -> None:
        assert sanitize("plain text\nno issues") == "plain text\nno issues"


# -- Round trip loading -----------------------------------------------------


class TestLoadGoldRoundTrip:
    def setup_method(self) -> None:
        clear_gold_cache()

    def teardown_method(self) -> None:
        clear_gold_cache()

    def test_valid_gold_loads(self, tmp_path: Path) -> None:
        text = _base_document()
        data = _build_three_level_gold(text)
        _write_gold(tmp_path, "doc_a", data)

        gold = load_gold("doc_a", gold_dir=tmp_path)

        assert isinstance(gold, GoldAnnotation)
        assert gold.document_id == "doc_a"
        assert gold.numbering_style == "uk_decimal"
        assert len(gold.clauses) == 5
        assert all(isinstance(c, GoldClause) for c in gold.clauses)

    def test_by_identifier(self, tmp_path: Path) -> None:
        text = _base_document()
        data = _build_three_level_gold(text)
        _write_gold(tmp_path, "doc_a", data)
        gold = load_gold("doc_a", gold_dir=tmp_path)

        assert set(gold.by_identifier) == {"1", "1.1", "1.1(a)", "1.2", "2"}
        assert gold.by_identifier["1.1"].heading == "First sub-clause about Widgets."

    def test_children(self, tmp_path: Path) -> None:
        text = _base_document()
        data = _build_three_level_gold(text)
        _write_gold(tmp_path, "doc_a", data)
        gold = load_gold("doc_a", gold_dir=tmp_path)

        children_of_1 = {c.identifier for c in gold.children("1")}
        assert children_of_1 == {"1.1", "1.2"}
        children_of_11 = {c.identifier for c in gold.children("1.1")}
        assert children_of_11 == {"1.1(a)"}
        assert gold.children("1.2") == ()
        assert gold.children("nonexistent") == ()

    def test_leaf_clauses(self, tmp_path: Path) -> None:
        text = _base_document()
        data = _build_three_level_gold(text)
        _write_gold(tmp_path, "doc_a", data)
        gold = load_gold("doc_a", gold_dir=tmp_path)

        leaf_ids = {c.identifier for c in gold.leaf_clauses}
        assert leaf_ids == {"1.1(a)", "1.2", "2"}

    def test_top_level_clauses(self, tmp_path: Path) -> None:
        text = _base_document()
        data = _build_three_level_gold(text)
        _write_gold(tmp_path, "doc_a", data)
        gold = load_gold("doc_a", gold_dir=tmp_path)

        top_ids = {c.identifier for c in gold.top_level_clauses}
        assert top_ids == {"1", "2"}

    def test_subtree_span(self, tmp_path: Path) -> None:
        text = _base_document()
        data = _build_three_level_gold(text)
        _write_gold(tmp_path, "doc_a", data)
        gold = load_gold("doc_a", gold_dir=tmp_path)

        clause_1 = gold.by_identifier["1"]
        clause_11 = gold.by_identifier["1.1"]
        clause_2 = gold.by_identifier["2"]

        # subtree of "1" should extend to the end of its deepest descendant (1.2)
        start, end = gold.subtree_span(clause_1)
        assert start == clause_1.char_start
        assert end == gold.by_identifier["1.2"].char_end

        # subtree of "1.1" extends to include its child 1.1(a)
        start11, end11 = gold.subtree_span(clause_11)
        assert start11 == clause_11.char_start
        assert end11 == max(clause_11.char_end, gold.by_identifier["1.1(a)"].char_end)

        # leaf clause: subtree span equals its own span
        start2, end2 = gold.subtree_span(clause_2)
        assert (start2, end2) == (clause_2.char_start, clause_2.char_end)

    def test_defined_terms_and_cross_references_parsed(self, tmp_path: Path) -> None:
        text = _base_document()
        data = _build_three_level_gold(text)
        term_start = text.index("Widgets")
        term_end = term_start + len("Widgets")
        data["defined_terms"] = [{"term": "Widgets", "definition_span": [term_start, term_end]}]
        ref_text = "clause 1.1"
        ref_start = text.index(ref_text)
        data["cross_references"] = [
            {
                "raw_text": ref_text,
                "char_start": ref_start,
                "target_identifier": "1.1",
                "kind": "internal_clause",
            }
        ]
        _write_gold(tmp_path, "doc_a", data)
        gold = load_gold("doc_a", gold_dir=tmp_path)

        assert len(gold.defined_terms) == 1
        term = gold.defined_terms[0]
        assert isinstance(term, GoldDefinedTerm)
        assert term.term == "Widgets"
        assert term.definition_start == term_start
        assert term.definition_end == term_end

        assert len(gold.cross_references) == 1
        ref = gold.cross_references[0]
        assert isinstance(ref, GoldCrossReference)
        assert ref.raw_text == ref_text
        assert ref.char_end == ref_start + len(ref_text)
        assert ref.target_identifier == "1.1"

    def test_ignores_underscore_keys(self, tmp_path: Path) -> None:
        text = _base_document()
        data = _build_three_level_gold(text)
        data["_seed_unresolved"] = ["some", "debug", "info"]
        _write_gold(tmp_path, "doc_a", data)

        gold = load_gold("doc_a", gold_dir=tmp_path)
        assert gold.document_id == "doc_a"

    def test_load_all_gold(self, tmp_path: Path) -> None:
        text = _base_document()
        data_a = _build_three_level_gold(text)
        data_b = dict(data_a)
        data_b["document_id"] = "doc_b"
        _write_gold(tmp_path, "doc_a", data_a)
        _write_gold(tmp_path, "doc_b", data_b)

        all_gold = load_all_gold(gold_dir=tmp_path)
        assert set(all_gold) == {"doc_a", "doc_b"}
        assert all(isinstance(v, GoldAnnotation) for v in all_gold.values())

    def test_load_all_gold_missing_dir(self, tmp_path: Path) -> None:
        missing = tmp_path / "does_not_exist"
        with pytest.raises(GoldError):
            load_all_gold(gold_dir=missing)

    def test_cache_returns_same_object(self, tmp_path: Path) -> None:
        text = _base_document()
        data = _build_three_level_gold(text)
        _write_gold(tmp_path, "doc_a", data)

        first = load_gold("doc_a", gold_dir=tmp_path)
        second = load_gold("doc_a", gold_dir=tmp_path)
        assert first is second

    def test_clear_gold_cache(self, tmp_path: Path) -> None:
        text = _base_document()
        data = _build_three_level_gold(text)
        _write_gold(tmp_path, "doc_a", data)

        first = load_gold("doc_a", gold_dir=tmp_path)
        clear_gold_cache()
        second = load_gold("doc_a", gold_dir=tmp_path)
        assert first is not second
        assert first == second


# -- verify_document ----------------------------------------------------------


class TestVerifyDocument:
    def setup_method(self) -> None:
        clear_gold_cache()

    def teardown_method(self) -> None:
        clear_gold_cache()

    def test_matching_document_passes(self, tmp_path: Path) -> None:
        text = _base_document()
        data = _build_three_level_gold(text)
        _write_gold(tmp_path, "doc_a", data)
        gold = load_gold("doc_a", gold_dir=tmp_path)

        verify_document(gold, text)  # should not raise

    def test_bad_hash_raises(self, tmp_path: Path) -> None:
        text = _base_document()
        data = _build_three_level_gold(text)
        _write_gold(tmp_path, "doc_a", data)
        gold = load_gold("doc_a", gold_dir=tmp_path)

        with pytest.raises(GoldError, match="doc_a"):
            verify_document(gold, text + " some extra trailing text")

    def test_bad_hash_same_length_raises(self, tmp_path: Path) -> None:
        # Same character count as the original, but different content, so the
        # length check passes and only the hash check can catch the drift.
        text = _base_document()
        data = _build_three_level_gold(text)
        _write_gold(tmp_path, "doc_a", data)
        gold = load_gold("doc_a", gold_dir=tmp_path)

        mutated = "X" + text[1:]
        assert len(sanitize(mutated)) == gold.n_chars
        with pytest.raises(GoldError, match="doc_a"):
            verify_document(gold, mutated)

    def test_bad_length_raises(self, tmp_path: Path) -> None:
        text = _base_document()
        data = _build_three_level_gold(text)
        _write_gold(tmp_path, "doc_a", data)
        gold = load_gold("doc_a", gold_dir=tmp_path)

        with pytest.raises(GoldError):
            verify_document(gold, text[:-1])


# -- GoldError paths ------------------------------------------------------


class TestGoldErrors:
    def setup_method(self) -> None:
        clear_gold_cache()

    def teardown_method(self) -> None:
        clear_gold_cache()

    def _minimal_data(self, text: str) -> dict[str, Any]:
        return {
            "document_id": "doc_x",
            "source_file": "doc_x.txt",
            "text_sha256": _sha256(text),
            "n_chars": len(sanitize(text)),
            "numbering_style": "uk_decimal",
            "clauses": [],
            "defined_terms": [],
            "cross_references": [],
        }

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(GoldError):
            load_gold("nonexistent_doc", gold_dir=tmp_path)

    def test_unknown_top_level_key_raises(self, tmp_path: Path) -> None:
        text = "1. FOO\nsome text\n"
        data = self._minimal_data(text)
        data["mystery_field"] = 123
        _write_gold(tmp_path, "doc_x", data)

        with pytest.raises(GoldError):
            load_gold("doc_x", gold_dir=tmp_path)

    def test_missing_parent_raises(self, tmp_path: Path) -> None:
        text = "1. FOO\n1.1 sub\n"
        data = self._minimal_data(text)
        data["clauses"] = [
            {
                "identifier": "1.1",
                "level": 2,
                "parent": "1",  # never defined
                "char_start": 0,
                "char_end": 5,
                "is_top_level": False,
                "heading": "sub",
            }
        ]
        _write_gold(tmp_path, "doc_x", data)

        with pytest.raises(GoldError, match="parent"):
            load_gold("doc_x", gold_dir=tmp_path)

    def test_wrong_parent_level_raises(self, tmp_path: Path) -> None:
        text = "1. FOO\n1.1 sub\n1.1.1 subsub\n"
        data = self._minimal_data(text)
        data["clauses"] = [
            {
                "identifier": "1",
                "level": 1,
                "parent": None,
                "char_start": 0,
                "char_end": 5,
                "is_top_level": True,
                "heading": "FOO",
            },
            {
                # level 3 claiming "1" (level 1) as its parent -- should be level 2
                "identifier": "1.1.1",
                "level": 3,
                "parent": "1",
                "char_start": 5,
                "char_end": 10,
                "is_top_level": False,
                "heading": "subsub",
            },
        ]
        _write_gold(tmp_path, "doc_x", data)

        with pytest.raises(GoldError, match="level"):
            load_gold("doc_x", gold_dir=tmp_path)

    def test_duplicate_identifier_raises(self, tmp_path: Path) -> None:
        text = "1. FOO\n2. BAR\n"
        data = self._minimal_data(text)
        data["clauses"] = [
            {
                "identifier": "1",
                "level": 1,
                "parent": None,
                "char_start": 0,
                "char_end": 5,
                "is_top_level": True,
                "heading": "FOO",
            },
            {
                "identifier": "1",  # duplicate
                "level": 1,
                "parent": None,
                "char_start": 5,
                "char_end": 10,
                "is_top_level": True,
                "heading": "BAR",
            },
        ]
        _write_gold(tmp_path, "doc_x", data)

        with pytest.raises(GoldError, match="duplicate"):
            load_gold("doc_x", gold_dir=tmp_path)

    def test_overlapping_spans_raise(self, tmp_path: Path) -> None:
        text = "1. FOO\n2. BAR\n"
        data = self._minimal_data(text)
        data["clauses"] = [
            {
                "identifier": "1",
                "level": 1,
                "parent": None,
                "char_start": 0,
                "char_end": 10,
                "is_top_level": True,
                "heading": "FOO",
            },
            {
                "identifier": "2",
                "level": 1,
                "parent": None,
                "char_start": 5,  # overlaps clause "1" which ends at 10
                "char_end": 15,
                "is_top_level": True,
                "heading": "BAR",
            },
        ]
        _write_gold(tmp_path, "doc_x", data)

        with pytest.raises(GoldError):
            load_gold("doc_x", gold_dir=tmp_path)

    def test_unsorted_spans_raise(self, tmp_path: Path) -> None:
        text = "1. FOO\n2. BAR\n"
        data = self._minimal_data(text)
        data["clauses"] = [
            {
                "identifier": "2",
                "level": 1,
                "parent": None,
                "char_start": 10,
                "char_end": 14,
                "is_top_level": True,
                "heading": "BAR",
            },
            {
                "identifier": "1",
                "level": 1,
                "parent": None,
                "char_start": 0,  # precedes previous clause's char_start -- unsorted
                "char_end": 5,
                "is_top_level": True,
                "heading": "FOO",
            },
        ]
        _write_gold(tmp_path, "doc_x", data)

        with pytest.raises(GoldError):
            load_gold("doc_x", gold_dir=tmp_path)

    def test_out_of_bounds_span_raises(self, tmp_path: Path) -> None:
        text = "1. FOO\n"
        data = self._minimal_data(text)
        data["clauses"] = [
            {
                "identifier": "1",
                "level": 1,
                "parent": None,
                "char_start": 0,
                "char_end": len(text) + 1000,  # way past n_chars
                "is_top_level": True,
                "heading": "FOO",
            },
        ]
        _write_gold(tmp_path, "doc_x", data)

        with pytest.raises(GoldError):
            load_gold("doc_x", gold_dir=tmp_path)

    def test_inverted_span_raises(self, tmp_path: Path) -> None:
        text = "1. FOO\n"
        data = self._minimal_data(text)
        data["clauses"] = [
            {
                "identifier": "1",
                "level": 1,
                "parent": None,
                "char_start": 5,
                "char_end": 2,  # end before start
                "is_top_level": True,
                "heading": "FOO",
            },
        ]
        _write_gold(tmp_path, "doc_x", data)

        with pytest.raises(GoldError):
            load_gold("doc_x", gold_dir=tmp_path)

    def test_dangling_internal_cross_reference_raises(self, tmp_path: Path) -> None:
        text = "1. FOO\nsee clause 9.9\n"
        data = self._minimal_data(text)
        data["clauses"] = [
            {
                "identifier": "1",
                "level": 1,
                "parent": None,
                "char_start": 0,
                "char_end": 7,
                "is_top_level": True,
                "heading": "FOO",
            },
        ]
        ref_start = text.index("clause 9.9")
        data["cross_references"] = [
            {
                "raw_text": "clause 9.9",
                "char_start": ref_start,
                "target_identifier": "9.9",  # does not exist
                "kind": "internal_clause",
            }
        ]
        _write_gold(tmp_path, "doc_x", data)

        with pytest.raises(GoldError, match="does not resolve"):
            load_gold("doc_x", gold_dir=tmp_path)

    def test_invalid_json_raises(self, tmp_path: Path) -> None:
        tmp_path.mkdir(parents=True, exist_ok=True)
        (tmp_path / "doc_x.json").write_text("{not valid json", encoding="utf-8")

        with pytest.raises(GoldError):
            load_gold("doc_x", gold_dir=tmp_path)

    def test_top_level_not_an_object_raises(self, tmp_path: Path) -> None:
        tmp_path.mkdir(parents=True, exist_ok=True)
        (tmp_path / "doc_x.json").write_text("[1, 2, 3]", encoding="utf-8")

        with pytest.raises(GoldError, match="JSON object"):
            load_gold("doc_x", gold_dir=tmp_path)

    def test_missing_required_top_level_key_raises(self, tmp_path: Path) -> None:
        text = "1. FOO\n"
        data = self._minimal_data(text)
        del data["n_chars"]
        _write_gold(tmp_path, "doc_x", data)

        with pytest.raises(GoldError, match="missing required key"):
            load_gold("doc_x", gold_dir=tmp_path)

    def test_document_id_mismatch_raises(self, tmp_path: Path) -> None:
        text = "1. FOO\n"
        data = self._minimal_data(text)
        data["document_id"] = "some_other_id"
        _write_gold(tmp_path, "doc_x", data)

        with pytest.raises(GoldError, match="declares document_id"):
            load_gold("doc_x", gold_dir=tmp_path)

    def test_clauses_not_a_list_raises(self, tmp_path: Path) -> None:
        text = "1. FOO\n"
        data = self._minimal_data(text)
        data["clauses"] = {"identifier": "1"}
        _write_gold(tmp_path, "doc_x", data)

        with pytest.raises(GoldError, match="'clauses' must be a list"):
            load_gold("doc_x", gold_dir=tmp_path)

    def test_clause_missing_key_raises(self, tmp_path: Path) -> None:
        text = "1. FOO\n"
        data = self._minimal_data(text)
        data["clauses"] = [
            {
                "identifier": "1",
                "level": 1,
                "parent": None,
                "char_start": 0,
                "char_end": 5,
                "is_top_level": True,
                # "heading" deliberately omitted
            }
        ]
        _write_gold(tmp_path, "doc_x", data)

        with pytest.raises(GoldError, match="missing key"):
            load_gold("doc_x", gold_dir=tmp_path)

    def test_is_top_level_inconsistent_with_level_raises(self, tmp_path: Path) -> None:
        text = "1. FOO\n"
        data = self._minimal_data(text)
        data["clauses"] = [
            {
                "identifier": "1",
                "level": 1,
                "parent": None,
                "char_start": 0,
                "char_end": 5,
                "is_top_level": False,  # inconsistent: level == 1
                "heading": "FOO",
            }
        ]
        _write_gold(tmp_path, "doc_x", data)

        with pytest.raises(GoldError, match="is_top_level"):
            load_gold("doc_x", gold_dir=tmp_path)

    def test_no_parent_but_wrong_level_raises(self, tmp_path: Path) -> None:
        text = "1. FOO\n"
        data = self._minimal_data(text)
        data["clauses"] = [
            {
                "identifier": "1",
                "level": 2,  # no parent, but level != 1
                "parent": None,
                "char_start": 0,
                "char_end": 5,
                "is_top_level": False,
                "heading": "FOO",
            }
        ]
        _write_gold(tmp_path, "doc_x", data)

        with pytest.raises(GoldError, match="no parent"):
            load_gold("doc_x", gold_dir=tmp_path)

    def test_defined_terms_not_a_list_raises(self, tmp_path: Path) -> None:
        text = "1. FOO\n"
        data = self._minimal_data(text)
        data["defined_terms"] = {"term": "FOO"}
        _write_gold(tmp_path, "doc_x", data)

        with pytest.raises(GoldError, match="'defined_terms' must be a list"):
            load_gold("doc_x", gold_dir=tmp_path)

    def test_defined_term_missing_key_raises(self, tmp_path: Path) -> None:
        text = "1. FOO\n"
        data = self._minimal_data(text)
        data["defined_terms"] = [{"term": "FOO"}]  # missing definition_span
        _write_gold(tmp_path, "doc_x", data)

        with pytest.raises(GoldError, match="missing key"):
            load_gold("doc_x", gold_dir=tmp_path)

    def test_defined_term_invalid_span_shape_raises(self, tmp_path: Path) -> None:
        text = "1. FOO\n"
        data = self._minimal_data(text)
        data["defined_terms"] = [{"term": "FOO", "definition_span": [0, 1, 2]}]
        _write_gold(tmp_path, "doc_x", data)

        with pytest.raises(GoldError, match="invalid definition_span"):
            load_gold("doc_x", gold_dir=tmp_path)

    def test_defined_term_out_of_bounds_span_raises(self, tmp_path: Path) -> None:
        text = "1. FOO\n"
        data = self._minimal_data(text)
        data["defined_terms"] = [{"term": "FOO", "definition_span": [0, 10_000]}]
        _write_gold(tmp_path, "doc_x", data)

        with pytest.raises(GoldError, match="out-of-bounds"):
            load_gold("doc_x", gold_dir=tmp_path)

    def test_cross_references_not_a_list_raises(self, tmp_path: Path) -> None:
        text = "1. FOO\n"
        data = self._minimal_data(text)
        data["cross_references"] = {"raw_text": "FOO"}
        _write_gold(tmp_path, "doc_x", data)

        with pytest.raises(GoldError, match="'cross_references' must be a list"):
            load_gold("doc_x", gold_dir=tmp_path)

    def test_cross_reference_missing_key_raises(self, tmp_path: Path) -> None:
        text = "1. FOO\n"
        data = self._minimal_data(text)
        data["cross_references"] = [{"raw_text": "FOO", "char_start": 0}]  # missing keys
        _write_gold(tmp_path, "doc_x", data)

        with pytest.raises(GoldError, match="missing"):
            load_gold("doc_x", gold_dir=tmp_path)

    def test_cross_reference_unknown_kind_raises(self, tmp_path: Path) -> None:
        text = "1. FOO\n"
        data = self._minimal_data(text)
        data["cross_references"] = [
            {
                "raw_text": "FOO",
                "char_start": 0,
                "target_identifier": None,
                "kind": "made_up_kind",
            }
        ]
        _write_gold(tmp_path, "doc_x", data)

        with pytest.raises(GoldError, match="unknown kind"):
            load_gold("doc_x", gold_dir=tmp_path)

    def test_cross_reference_out_of_bounds_span_raises(self, tmp_path: Path) -> None:
        text = "1. FOO\n"
        data = self._minimal_data(text)
        data["cross_references"] = [
            {
                "raw_text": "way too long to fit in this tiny document",
                "char_start": 0,
                "target_identifier": None,
                "kind": "external_statute",
            }
        ]
        _write_gold(tmp_path, "doc_x", data)

        with pytest.raises(GoldError, match="out-of-bounds"):
            load_gold("doc_x", gold_dir=tmp_path)

    def test_external_cross_reference_no_target_ok(self, tmp_path: Path) -> None:
        text = "1. FOO\nsee GDPR Article 5\n"
        data = self._minimal_data(text)
        data["clauses"] = [
            {
                "identifier": "1",
                "level": 1,
                "parent": None,
                "char_start": 0,
                "char_end": 7,
                "is_top_level": True,
                "heading": "FOO",
            },
        ]
        ref_start = text.index("GDPR Article 5")
        data["cross_references"] = [
            {
                "raw_text": "GDPR Article 5",
                "char_start": ref_start,
                "target_identifier": None,
                "kind": "external_statute",
            }
        ]
        _write_gold(tmp_path, "doc_x", data)

        gold = load_gold("doc_x", gold_dir=tmp_path)
        assert gold.cross_references[0].target_identifier is None


# -- locate_chunks ----------------------------------------------------------


class TestLocateChunks:
    def test_exact_chunks_in_order(self) -> None:
        text = "AAAA BBBB CCCC DDDD"
        chunks = ["AAAA", "BBBB", "CCCC", "DDDD"]
        spans = locate_chunks(chunks, text)

        assert len(spans) == 4
        for chunk, span in zip(chunks, spans, strict=True):
            assert span is not None
            assert span.exact is True
            assert text[span.char_start : span.char_end] == chunk
            assert span.covered_chars == len(chunk)

    def test_chunks_out_of_order(self) -> None:
        text = "AAAA BBBB CCCC DDDD"
        # Chunk set emitted out of document order.
        chunks = ["CCCC", "AAAA", "DDDD", "BBBB"]
        spans = locate_chunks(chunks, text)

        assert all(span is not None for span in spans)
        expected_starts = {"AAAA": 0, "BBBB": 5, "CCCC": 10, "DDDD": 15}
        for chunk, span in zip(chunks, spans, strict=True):
            assert span is not None
            assert span.char_start == expected_starts[chunk]
            assert span.exact is True

    def test_synthesised_heading_prepended_suffix_path(self) -> None:
        real_text_segment = "The actual clause body text goes here in full."
        text = "PREAMBLE\n" + real_text_segment + "\nMORE TEXT"
        breadcrumb = "1.1 Confidentiality > "
        chunk = breadcrumb + real_text_segment

        spans = locate_chunks([chunk], text)
        assert len(spans) == 1
        span = spans[0]
        assert span is not None
        assert span.exact is False
        located_text = text[span.char_start : span.char_end]
        assert located_text == real_text_segment
        assert breadcrumb not in located_text
        assert span.covered_chars == len(real_text_segment)

    def test_trailer_appended_prefix_path(self) -> None:
        real_text_segment = "The actual clause body text goes here in full and is long."
        text = "PREAMBLE\n" + real_text_segment + "\nMORE TEXT"
        trailer = " [footer: page 3 of 10 -- confidential]"
        chunk = real_text_segment + trailer

        spans = locate_chunks([chunk], text)
        assert len(spans) == 1
        span = spans[0]
        assert span is not None
        assert span.exact is False
        located_text = text[span.char_start : span.char_end]
        assert located_text == real_text_segment
        assert trailer not in located_text
        assert span.covered_chars == len(real_text_segment)

    def test_chunk_nowhere_in_document_returns_none(self) -> None:
        text = "The quick brown fox jumps over the lazy dog."
        chunk = "This text shares nothing at all with the document above zzz."
        spans = locate_chunks([chunk], text)
        assert spans == (None,)

    def test_repeated_boilerplate_via_cursor(self) -> None:
        boilerplate = "STANDARD CLAUSE TEXT. "
        text = boilerplate + "unique one. " + boilerplate + "unique two. " + boilerplate
        chunks = [boilerplate, boilerplate, boilerplate]
        spans = locate_chunks(chunks, text)

        assert all(span is not None for span in spans)
        starts = [span.char_start for span in spans if span is not None]
        # Each successive occurrence should be located later than the previous one.
        assert starts == sorted(starts)
        assert len(set(starts)) == 3

    def test_empty_chunk_returns_none(self) -> None:
        text = "Some document text."
        spans = locate_chunks(["", "   ", "\n\t"], text)
        assert spans == (None, None, None)

    def test_mixed_empty_and_valid_chunks(self) -> None:
        text = "AAAA BBBB"
        spans = locate_chunks(["AAAA", "", "BBBB"], text)
        assert spans[0] is not None
        assert spans[1] is None
        assert spans[2] is not None

    def test_cursor_resets_after_failure(self) -> None:
        text = "AAAA BBBB CCCC"
        # Second chunk fails entirely; third chunk should still be found even
        # though the cursor reset to 0.
        chunks = ["BBBB", "totally absent text", "AAAA"]
        spans = locate_chunks(chunks, text)
        assert spans[0] is not None
        assert spans[1] is None
        assert spans[2] is not None
        assert spans[2].char_start == 0

    def test_far_apart_partial_matches_pick_a_side(self) -> None:
        # Build a chunk whose first half (a valid verbatim prefix) occurs at one
        # place in the document, and whose second half (a valid verbatim suffix)
        # occurs somewhere far away -- both individually clear the 50% coverage
        # bar, but they are much further apart than 2 * len(chunk), so the union
        # path (step 4) must not fire; one side is picked instead (step 2/3),
        # never None.
        prefix_part = "P" * 25
        suffix_part = "S" * 25
        chunk = prefix_part + suffix_part  # len 50; each half is exactly 50%
        text = "A" * 5 + prefix_part + "Q" + "M" * 300 + "R" + suffix_part + "Z" * 5

        prefix_start = text.index(prefix_part)
        suffix_start = text.rindex(suffix_part)
        assert suffix_start - (prefix_start + len(prefix_part)) > 2 * len(chunk)

        spans = locate_chunks([chunk], text)
        span = spans[0]
        assert span is not None
        assert span.exact is False
        assert span.covered_chars == 25
        located_text = text[span.char_start : span.char_end]
        assert located_text in (prefix_part, suffix_part)

    def test_far_apart_partial_matches_suffix_before_prefix(self) -> None:
        # Same idea as above but with the suffix-matching occurrence located
        # earlier in the document than the prefix-matching occurrence, to
        # exercise the opposite ordering of the two spans.
        prefix_part = "P" * 25
        suffix_part = "S" * 25
        chunk = prefix_part + suffix_part
        text = "A" * 5 + suffix_part + "R" + "M" * 300 + "Q" + prefix_part + "Z" * 5

        spans = locate_chunks([chunk], text)
        span = spans[0]
        assert span is not None
        assert span.exact is False
        assert span.covered_chars == 25

    def test_close_partial_matches_merge_into_union(self) -> None:
        # Both a verbatim prefix and a verbatim suffix are found, and their
        # occurrences in the document are close together (within 2 * len(chunk)),
        # so the located span should be their union, not just one side.
        prefix_part = "P" * 25
        suffix_part = "S" * 25
        chunk = prefix_part + suffix_part
        text = "A" * 5 + prefix_part + "Q" + "M" * 10 + "R" + suffix_part + "Z" * 5

        prefix_start = text.index(prefix_part)
        suffix_start = text.index(suffix_part)
        gap = suffix_start - (prefix_start + len(prefix_part))
        assert gap <= 2 * len(chunk)

        spans = locate_chunks([chunk], text)
        span = spans[0]
        assert span is not None
        assert span.exact is False
        assert span.char_start == prefix_start
        assert span.char_end == suffix_start + len(suffix_part)
        assert span.covered_chars == 25


class TestLocalizationHelpers:
    def test_localization_rate_all_located(self) -> None:
        text = "AAAA BBBB"
        spans = locate_chunks(["AAAA", "BBBB"], text)
        assert localization_rate(spans) == 1.0

    def test_localization_rate_partial(self) -> None:
        text = "AAAA BBBB"
        spans = locate_chunks(["AAAA", "nowhere to be found at all"], text)
        assert localization_rate(spans) == 0.5

    def test_localization_rate_empty(self) -> None:
        assert localization_rate(()) == 0.0

    def test_localization_coverage_full(self) -> None:
        text = "AAAA BBBB"
        chunks = ["AAAA", "BBBB"]
        spans = locate_chunks(chunks, text)
        assert localization_coverage(spans, chunks) == 1.0

    def test_localization_coverage_partial(self) -> None:
        text = "AAAA BBBB"
        chunks = ["AAAA", "not present anywhere in the text at all"]
        spans = locate_chunks(chunks, text)
        coverage = localization_coverage(spans, chunks)
        assert 0.0 < coverage < 1.0

    def test_localization_coverage_empty_chunks(self) -> None:
        assert localization_coverage((), ()) == 0.0

    def test_manual_chunk_span_covered_chars(self) -> None:
        span = ChunkSpan(chunk_index=0, char_start=0, char_end=4, covered_chars=4, exact=True)
        assert localization_coverage((span,), ["AAAA"]) == 1.0


class TestOverlapChars:
    def test_disjoint_spans(self) -> None:
        assert overlap_chars(0, 5, 10, 15) == 0

    def test_touching_spans(self) -> None:
        # [0, 5) and [5, 10) touch but do not overlap.
        assert overlap_chars(0, 5, 5, 10) == 0

    def test_nested_spans(self) -> None:
        assert overlap_chars(0, 10, 3, 7) == 4

    def test_partial_overlap(self) -> None:
        assert overlap_chars(0, 10, 5, 15) == 5

    def test_identical_spans(self) -> None:
        assert overlap_chars(2, 8, 2, 8) == 6

    def test_reversed_argument_order_symmetric(self) -> None:
        assert overlap_chars(5, 15, 0, 10) == 5
