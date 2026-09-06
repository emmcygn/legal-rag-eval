"""Validate the committed gold annotations against the fixture documents.

These are the harness's ground truth. Every structural metric and every relevance
judgement is an overlap computation against the spans in ``gold/``, so a span that is off
by a paragraph is not a cosmetic problem — it silently changes every published number.
This module is the gate: it runs in CI on every push and fails loudly if a fixture and its
annotation have drifted apart.

Nothing here imports ``lexichunk``. The point of the gold files is that they were made
without it.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from scaffolder.gold import GoldAnnotation, load_gold, sanitize, verify_document

REPO_ROOT = Path(__file__).resolve().parent.parent
GOLD_DIR = REPO_ROOT / "gold"
FIXTURE_DIR = REPO_ROOT / "src" / "scaffolder" / "fixtures" / "documents"

DOCUMENT_IDS = [
    "eu_gdpr_excerpt",
    "uk_service_agreement",
    "uk_terms_conditions",
    "us_msa",
    "us_terms_of_service",
]


def _text(document_id: str) -> str:
    return sanitize((FIXTURE_DIR / f"{document_id}.txt").read_text(encoding="utf-8"))


@pytest.fixture(scope="module", params=DOCUMENT_IDS)
def annotated(request: pytest.FixtureRequest) -> tuple[str, GoldAnnotation, str]:
    """Yield (document_id, gold annotation, sanitised text) for each fixture."""
    document_id = str(request.param)
    return document_id, load_gold(document_id, GOLD_DIR), _text(document_id)


def test_every_fixture_has_a_gold_file() -> None:
    """A fixture without gold cannot be scored, so it must not exist silently."""
    fixtures = sorted(p.stem for p in FIXTURE_DIR.glob("*.txt"))
    annotations = sorted(p.stem for p in GOLD_DIR.glob("*.json"))
    assert fixtures == sorted(DOCUMENT_IDS)
    assert annotations == sorted(DOCUMENT_IDS)


def test_document_hash_matches(annotated: tuple[str, GoldAnnotation, str]) -> None:
    """The annotation is pinned to an exact document revision."""
    document_id, gold, text = annotated
    verify_document(gold, text)
    assert gold.text_sha256 == hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert gold.n_chars == len(text)
    assert gold.document_id == document_id


def test_clause_spans_are_in_bounds(annotated: tuple[str, GoldAnnotation, str]) -> None:
    document_id, gold, text = annotated
    assert gold.clauses, f"{document_id} has no annotated clauses"
    for clause in gold.clauses:
        assert 0 <= clause.char_start < clause.char_end <= len(text), (
            f"{document_id}:{clause.identifier} span "
            f"[{clause.char_start}, {clause.char_end}) is out of bounds"
        )


def test_clause_spans_are_ordered_and_non_overlapping(
    annotated: tuple[str, GoldAnnotation, str],
) -> None:
    """Clause spans partition the numbered body; a clause excludes its descendants."""
    document_id, gold, _ = annotated
    previous_end = 0
    previous_id = "<start>"
    for clause in gold.clauses:
        assert clause.char_start >= previous_end, (
            f"{document_id}: {clause.identifier} starts at {clause.char_start}, "
            f"before {previous_id} ends at {previous_end}"
        )
        previous_end = clause.char_end
        previous_id = clause.identifier


def test_clause_identifiers_are_unique(annotated: tuple[str, GoldAnnotation, str]) -> None:
    document_id, gold, _ = annotated
    identifiers = [c.identifier for c in gold.clauses]
    duplicates = {i for i in identifiers if identifiers.count(i) > 1}
    assert not duplicates, f"{document_id} has duplicate identifiers: {sorted(duplicates)}"


def test_parents_exist_and_are_one_level_up(annotated: tuple[str, GoldAnnotation, str]) -> None:
    document_id, gold, _ = annotated
    seen: dict[str, int] = {}
    for clause in gold.clauses:
        if clause.parent is None:
            assert clause.level == 1, (
                f"{document_id}:{clause.identifier} has no parent but level {clause.level}"
            )
            assert clause.is_top_level
        else:
            assert clause.parent in seen, (
                f"{document_id}:{clause.identifier} names parent {clause.parent!r}, "
                f"which does not appear earlier"
            )
            assert seen[clause.parent] == clause.level - 1, (
                f"{document_id}:{clause.identifier} is level {clause.level} but its parent "
                f"{clause.parent} is level {seen[clause.parent]}"
            )
            assert not clause.is_top_level
        seen[clause.identifier] = clause.level


def test_subtree_spans_contain_every_descendant(
    annotated: tuple[str, GoldAnnotation, str],
) -> None:
    """The derived subtree span must actually cover the children the metrics rely on."""
    document_id, gold, _ = annotated
    for clause in gold.clauses:
        start, end = gold.subtree_span(clause)
        assert start == clause.char_start
        assert end >= clause.char_end
        for child in gold.children(clause.identifier):
            child_start, child_end = gold.subtree_span(child)
            assert start <= child_start and child_end <= end, (
                f"{document_id}: subtree of {clause.identifier} does not contain "
                f"{child.identifier}"
            )


def test_top_level_clauses_exist_and_cover_the_body(
    annotated: tuple[str, GoldAnnotation, str],
) -> None:
    document_id, gold, text = annotated
    top = gold.top_level_clauses
    assert top, f"{document_id} has no top-level clauses"
    covered = sum(end - start for start, end in (gold.subtree_span(c) for c in top))
    assert covered > 0.3 * len(text), (
        f"{document_id}: top-level clause subtrees cover only {covered} of {len(text)} "
        f"characters — the annotation is probably missing most of the document"
    )


def test_top_level_subtrees_do_not_overlap(annotated: tuple[str, GoldAnnotation, str]) -> None:
    document_id, gold, _ = annotated
    spans = sorted(gold.subtree_span(c) for c in gold.top_level_clauses)
    for (a_start, a_end), (b_start, b_end) in zip(spans, spans[1:], strict=False):
        assert a_end <= b_start, (
            f"{document_id}: top-level subtrees [{a_start},{a_end}) and "
            f"[{b_start},{b_end}) overlap"
        )


def test_defined_term_spans_contain_their_term(
    annotated: tuple[str, GoldAnnotation, str],
) -> None:
    document_id, gold, text = annotated
    for term in gold.defined_terms:
        assert 0 <= term.definition_start < term.definition_end <= len(text), (
            f"{document_id}: definition span for {term.term!r} is out of bounds"
        )
        span_text = text[term.definition_start : term.definition_end]
        assert term.term.lower() in span_text.lower(), (
            f"{document_id}: definition span for {term.term!r} does not contain the term. "
            f"Span begins: {span_text[:80]!r}"
        )


def test_defined_terms_are_unique(annotated: tuple[str, GoldAnnotation, str]) -> None:
    document_id, gold, _ = annotated
    terms = [t.term.lower() for t in gold.defined_terms]
    duplicates = {t for t in terms if terms.count(t) > 1}
    assert not duplicates, f"{document_id} defines these terms more than once: {sorted(duplicates)}"


def test_cross_reference_raw_text_matches_the_document(
    annotated: tuple[str, GoldAnnotation, str],
) -> None:
    document_id, gold, text = annotated
    for ref in gold.cross_references:
        actual = text[ref.char_start : ref.char_start + len(ref.raw_text)]
        assert actual == ref.raw_text, (
            f"{document_id}: cross-reference at {ref.char_start} should read "
            f"{ref.raw_text!r} but the document has {actual!r}"
        )


def test_internal_cross_reference_targets_resolve(
    annotated: tuple[str, GoldAnnotation, str],
) -> None:
    document_id, gold, _ = annotated
    for ref in gold.cross_references:
        if ref.kind.startswith("internal_"):
            assert ref.target_identifier is not None, (
                f"{document_id}: {ref.raw_text!r} is marked {ref.kind} but has no target"
            )
            assert ref.target_identifier in gold.by_identifier, (
                f"{document_id}: {ref.raw_text!r} targets {ref.target_identifier!r}, "
                f"which is not an annotated clause"
            )
        else:
            assert ref.target_identifier is None, (
                f"{document_id}: {ref.raw_text!r} is external ({ref.kind}) but names an "
                f"internal target {ref.target_identifier!r}"
            )


def test_no_seed_markers_remain(annotated: tuple[str, GoldAnnotation, str]) -> None:
    """`_seed_unresolved` marks work the hand-check pass was supposed to finish."""
    document_id, _, _ = annotated
    raw = (GOLD_DIR / f"{document_id}.json").read_text(encoding="utf-8")
    assert "_seed_unresolved" not in raw, (
        f"{document_id}.json still contains _seed_unresolved markers left by the seeder; "
        f"the hand-check pass has not finished."
    )
