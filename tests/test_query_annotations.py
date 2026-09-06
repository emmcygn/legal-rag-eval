"""Validate the committed query annotations against the gold clause spans.

The previous annotation set contained queries whose relevant sections were not in the
target document at all — five of twenty-two were unsatisfiable by any strategy, so they
contributed a guaranteed zero to every strategy and to every paired difference while still
counting toward n. Two others were so broad that a majority of all chunks were judged
relevant. This module makes both failure modes impossible to commit.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from legal_rag_eval.gold import load_all_gold
from legal_rag_eval.queries import load_queries
from legal_rag_eval.retrieval.simulator import load_queries_from_yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
GOLD_DIR = REPO_ROOT / "gold"
QUERY_DIR = REPO_ROOT / "queries"
FIXTURE_DIR = REPO_ROOT / "src" / "legal_rag_eval" / "fixtures" / "documents"

MIN_QUERIES = 30
MIN_QUERIES_PER_DOCUMENT = 6

# `category` records the kind of question, not a chunker failure mode. The five failure
# modes the previous schema used map almost one-to-one onto the structural metrics this
# project uses to argue for LexiChunk, so the query set was labelled by the claims it was
# meant to support. See queries/schema.md.
ALLOWED_CATEGORIES = frozenset(
    {
        "definition_lookup",
        "clause_lookup",
        "numeric_lookup",
        "conditional",
        "multi_clause",
        "cross_reference",
    }
)


def test_queries_load_and_resolve() -> None:
    """Every annotated clause identifier resolves to a real gold span."""
    queries = load_queries_from_yaml(str(QUERY_DIR), str(GOLD_DIR))
    assert len(queries) >= MIN_QUERIES, (
        f"{len(queries)} queries; the annotation set is meant to be at least {MIN_QUERIES}"
    )


def test_query_ids_are_unique() -> None:
    raw = load_queries(QUERY_DIR)
    ids = [q.id for q in raw]
    duplicates = {i for i in ids if ids.count(i) > 1}
    assert not duplicates, f"duplicate query ids: {sorted(duplicates)}"


def test_every_document_has_queries() -> None:
    raw = load_queries(QUERY_DIR)
    documented = {q.document_id for q in raw}
    fixtures = {p.stem for p in FIXTURE_DIR.glob("*.txt")}
    assert fixtures <= documented, f"documents with no queries: {sorted(fixtures - documented)}"


def test_queries_are_spread_across_documents() -> None:
    """Per-document minimum, so 30 queries cannot all come from one document.

    Leave-one-document-out on the old set moved the headline delta from +0.109 to
    +0.024 because one document carried it. An even spread does not make the queries
    independent, but it stops a single document from being the whole result.
    """
    counts: dict[str, int] = {}
    for query in load_queries(QUERY_DIR):
        counts[query.document_id] = counts.get(query.document_id, 0) + 1
    thin = {d: n for d, n in counts.items() if n < MIN_QUERIES_PER_DOCUMENT}
    assert not thin, f"documents with fewer than {MIN_QUERIES_PER_DOCUMENT} queries: {thin}"


def test_categories_are_from_the_documented_set() -> None:
    unknown = {
        f"{q.id}: {q.category!r}"
        for q in load_queries(QUERY_DIR)
        if q.category not in ALLOWED_CATEGORIES
    }
    assert not unknown, (
        "queries with a category outside queries/schema.md's set "
        f"{sorted(ALLOWED_CATEGORIES)}:\n  " + "\n  ".join(sorted(unknown))
    )


def test_every_relevant_clause_exists_in_gold() -> None:
    """The check the old harness lacked: an identifier must name a real clause."""
    gold_by_document = load_all_gold(GOLD_DIR)
    missing: list[str] = []
    for query in load_queries(QUERY_DIR):
        gold = gold_by_document.get(query.document_id)
        assert gold is not None, f"{query.id}: no gold file for {query.document_id}"
        for section in query.relevant_sections:
            if section.section_id not in gold.by_identifier:
                missing.append(f"{query.id} -> {query.document_id}:{section.section_id}")
    assert not missing, "relevant clauses that do not exist in gold:\n  " + "\n  ".join(missing)


def test_every_query_has_at_least_one_exact_relevant_clause() -> None:
    """A query with no grade-3 clause has no right answer to find."""
    for query in load_queries(QUERY_DIR):
        grades = [s.relevance for s in query.relevant_sections]
        assert grades, f"{query.id} has no relevant clauses"
        assert max(grades) == 3, f"{query.id} has no grade-3 (exact) relevant clause"
        assert all(g in (1, 2, 3) for g in grades), f"{query.id} has an out-of-range grade"


def test_relevant_spans_are_not_degenerate() -> None:
    """No query may mark a large fraction of its document as relevant.

    A query whose relevant spans cover most of the document is satisfied by retrieving
    almost anything, which inflates every strategy's precision equally and adds noise to
    every paired test. The threshold is deliberately generous; the previous set had queries
    where a majority of all chunks were judged relevant.
    """
    gold_by_document = load_all_gold(GOLD_DIR)
    offenders: list[str] = []
    for query in load_queries_from_yaml(str(QUERY_DIR), str(GOLD_DIR)):
        document_id = query.document_ids[0]
        gold = gold_by_document[document_id]
        # union of relevant spans, in case annotations nest
        spans = sorted((s.char_start, s.char_end) for s in query.relevant_sections)
        covered = 0
        current_start, current_end = spans[0]
        for start, end in spans[1:]:
            if start > current_end:
                covered += current_end - current_start
                current_start, current_end = start, end
            else:
                current_end = max(current_end, end)
        covered += current_end - current_start
        fraction = covered / gold.n_chars
        if fraction > 0.25:
            offenders.append(f"{query.id}: {fraction:.0%} of {document_id}")
    assert not offenders, "queries covering too much of their document:\n  " + "\n  ".join(
        offenders
    )


def test_no_query_marks_too_many_gold_clauses_relevant() -> None:
    """The same degeneracy check, counted in clauses rather than characters.

    A query may name a handful of short clauses (small character coverage) and still be
    so broad that a third of the document's clause set answers it. Both counts are
    capped so neither route to a degenerate query is open.
    """
    gold_by_document = load_all_gold(GOLD_DIR)
    offenders: list[str] = []
    for query in load_queries(QUERY_DIR):
        gold = gold_by_document[query.document_id]
        # Count the subtree, since resolution scores a named parent's descendants too.
        relevant: set[str] = set()
        for section in query.relevant_sections:
            stack = [gold.by_identifier[section.section_id]]
            while stack:
                clause = stack.pop()
                relevant.add(clause.identifier)
                stack.extend(gold.children(clause.identifier))
        fraction = len(relevant) / len(gold.clauses)
        if fraction > 0.30:
            offenders.append(
                f"{query.id}: {len(relevant)}/{len(gold.clauses)} clauses "
                f"({fraction:.0%}) of {query.document_id}"
            )
    assert not offenders, "queries marking too many gold clauses relevant:\n  " + "\n  ".join(
        offenders
    )


def test_query_text_is_a_question_a_reader_would_ask() -> None:
    """Cheap sanity gate: non-empty, not a restatement of the clause identifier."""
    for query in load_queries(QUERY_DIR):
        assert len(query.text.strip()) >= 15, f"{query.id}: query text is too short"
        for section in query.relevant_sections:
            assert section.section_id not in query.text, (
                f"{query.id} names its answer's identifier {section.section_id!r} in the "
                f"query text, which leaks the target"
            )


@pytest.mark.parametrize("field", ["text", "category"])
def test_required_fields_are_present(field: str) -> None:
    for query in load_queries(QUERY_DIR):
        value = getattr(query, field, "") or getattr(query, "failure_mode", "")
        assert str(value).strip(), f"{query.id} is missing {field}"
