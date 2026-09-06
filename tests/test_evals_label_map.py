"""Validation of the LEDGAR -> LexiChunk clause-type mapping.

Includes checks against the mapping actually shipped in
``evals/ledgar_label_map.yaml``: it is a hand-written legal artefact, so the
invariants that keep the accuracy number honest are asserted here rather than
left to review.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

import pytest
import yaml

from legal_rag_eval.evals.label_map import (
    DEFAULT_MAP_PATH,
    OUT_OF_SCOPE,
    load_label_map,
    parse_label_map,
    valid_clause_types,
)

if TYPE_CHECKING:
    from pathlib import Path

TYPES = {"definitions", "confidentiality", "payment", "boilerplate"}

MINIMAL = {
    "version": 1,
    "dataset": "test",
    "mappings": {
        "Definitions": {"lexichunk": "definitions", "rationale": "direct match"},
        "Confidentiality": {"lexichunk": "confidentiality", "rationale": "direct match"},
        "Base Salary": {"lexichunk": OUT_OF_SCOPE, "rationale": "no employment class"},
    },
}


def _mutate(**changes: object) -> dict[str, object]:
    """Return MINIMAL with its ``mappings`` replaced/extended."""
    doc = {k: v for k, v in MINIMAL.items() if k != "mappings"}
    doc["mappings"] = {**MINIMAL["mappings"], **changes}  # type: ignore[dict-item]
    return doc


class TestParsing:
    def test_parses_a_valid_document(self) -> None:
        mapping = parse_label_map(MINIMAL, known_clause_types=TYPES)
        assert len(mapping.entries) == 3
        assert mapping.version == 1
        assert mapping.source_dataset == "test"

    def test_target_resolves_in_scope_labels(self) -> None:
        mapping = parse_label_map(MINIMAL, known_clause_types=TYPES)
        assert mapping.target("Definitions") == "definitions"

    def test_target_is_none_for_out_of_scope_and_unknown_labels(self) -> None:
        mapping = parse_label_map(MINIMAL, known_clause_types=TYPES)
        assert mapping.target("Base Salary") is None
        assert mapping.target("Never Heard Of It") is None

    def test_in_scope_labels_and_targets(self) -> None:
        mapping = parse_label_map(MINIMAL, known_clause_types=TYPES)
        assert mapping.in_scope_labels == {"Definitions", "Confidentiality"}
        assert mapping.targets == {"definitions", "confidentiality"}

    def test_entry_in_scope_flag(self) -> None:
        mapping = parse_label_map(MINIMAL, known_clause_types=TYPES)
        entry = mapping.get("Base Salary")
        assert entry is not None
        assert not entry.in_scope


class TestValidation:
    def test_rejects_non_mapping_document(self) -> None:
        with pytest.raises(ValueError, match="top level"):
            parse_label_map(["not", "a", "mapping"], known_clause_types=TYPES)

    def test_rejects_missing_mappings_key(self) -> None:
        with pytest.raises(ValueError, match="'mappings'"):
            parse_label_map({"version": 1}, known_clause_types=TYPES)

    def test_rejects_empty_mappings(self) -> None:
        with pytest.raises(ValueError, match="'mappings'"):
            parse_label_map({"mappings": {}}, known_clause_types=TYPES)

    def test_rejects_unknown_clause_type(self) -> None:
        doc = _mutate(Bogus={"lexichunk": "not_a_clause_type", "rationale": "why"})
        with pytest.raises(ValueError, match="not a\n?\\s*LexiChunk ClauseType|ClauseType"):
            parse_label_map(doc, known_clause_types=TYPES)

    def test_rejects_missing_rationale(self) -> None:
        doc = _mutate(Bogus={"lexichunk": "payment"})
        with pytest.raises(ValueError, match="rationale"):
            parse_label_map(doc, known_clause_types=TYPES)

    def test_rejects_blank_rationale(self) -> None:
        doc = _mutate(Bogus={"lexichunk": "payment", "rationale": "   "})
        with pytest.raises(ValueError, match="rationale"):
            parse_label_map(doc, known_clause_types=TYPES)

    def test_rejects_missing_target(self) -> None:
        doc = _mutate(Bogus={"rationale": "why"})
        with pytest.raises(ValueError, match="lexichunk"):
            parse_label_map(doc, known_clause_types=TYPES)

    def test_rejects_scalar_entry(self) -> None:
        doc = _mutate(Bogus="payment")
        with pytest.raises(ValueError, match="must be a mapping"):
            parse_label_map(doc, known_clause_types=TYPES)

    def test_rejects_non_integer_version(self) -> None:
        doc = dict(MINIMAL)
        doc["version"] = "one"
        with pytest.raises(ValueError, match="version"):
            parse_label_map(doc, known_clause_types=TYPES)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_label_map(tmp_path / "nope.yaml")

    def test_loads_from_disk(self, tmp_path: Path) -> None:
        target = tmp_path / "map.yaml"
        target.write_text(yaml.safe_dump(MINIMAL), encoding="utf-8")
        assert len(load_label_map(target, known_clause_types=TYPES).entries) == 3


class TestValidClauseTypes:
    def test_includes_the_documented_classes(self) -> None:
        types = valid_clause_types()
        assert {"definitions", "confidentiality", "indemnification", "unknown"} <= types
        assert len(types) >= 27


class TestShippedMap:
    """Invariants for the mapping actually used to produce published numbers."""

    @pytest.fixture(scope="class")
    @classmethod
    def shipped(cls):  # type: ignore[no-untyped-def]
        if not DEFAULT_MAP_PATH.is_file():
            pytest.skip(f"{DEFAULT_MAP_PATH} not present")
        return load_label_map(DEFAULT_MAP_PATH)

    def test_covers_all_hundred_ledgar_labels(self, shipped) -> None:  # type: ignore[no-untyped-def]
        assert len(shipped.entries) == 100

    def test_every_target_is_a_real_clause_type(self, shipped) -> None:  # type: ignore[no-untyped-def]
        known = valid_clause_types()
        for entry in shipped.entries.values():
            assert entry.lexichunk == OUT_OF_SCOPE or entry.lexichunk in known

    def test_never_maps_gold_onto_unknown(self, shipped) -> None:  # type: ignore[no-untyped-def]
        # `unknown` is the classifier's failure output. As a gold label it would
        # reward the classifier for giving up, so it must never appear.
        assert all(e.lexichunk != "unknown" for e in shipped.entries.values())

    def test_never_maps_onto_positional_classes(self, shipped) -> None:  # type: ignore[no-untyped-def]
        # `preamble`/`recitals` are assigned from document position, which an
        # isolated LEDGAR provision does not have.
        positional = {"preamble", "recitals"}
        assert all(e.lexichunk not in positional for e in shipped.entries.values())

    def test_every_entry_has_a_substantive_rationale(self, shipped) -> None:  # type: ignore[no-untyped-def]
        for entry in shipped.entries.values():
            assert len(entry.rationale) >= 5, entry.ledgar_label

    def test_mapping_is_many_to_one_not_one_to_one(self, shipped) -> None:  # type: ignore[no-untyped-def]
        counts = Counter(e.lexichunk for e in shipped.entries.values() if e.in_scope)
        assert any(n > 1 for n in counts.values())

    def test_keeps_a_usable_share_in_scope(self, shipped) -> None:  # type: ignore[no-untyped-def]
        # A near-empty mapping would make the accuracy number vacuous; a
        # near-total one would mean labels were forced.
        share = len(shipped.in_scope_labels) / len(shipped.entries)
        assert 0.2 <= share <= 0.9

    def test_uses_a_broad_slice_of_the_taxonomy(self, shipped) -> None:  # type: ignore[no-untyped-def]
        assert len(shipped.targets) >= 10
