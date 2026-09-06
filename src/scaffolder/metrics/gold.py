"""Structural metrics scored against hand-checked gold annotations.

Every metric in this module is computed from character-span overlap between the gold
spans recorded in ``gold/<document_id>.json`` (see ``gold/SCHEMA.md`` and
:mod:`scaffolder.gold`) and the chunk spans attached by
:func:`scaffolder.chunking.pipeline.attach_spans`. This replaces
:mod:`scaffolder.metrics.structural`, whose ground truth was LexiChunk's own parse of the
document -- LexiChunk therefore scored perfectly by construction and the baselines' scores
moved whenever LexiChunk's parser changed, even though the baselines had not.

Rules every metric here obeys:

1. No metric decides a score by searching chunk *text* for gold content (no ``in``, no
   word-overlap, no substring matching of ``chunk.text`` against anything). Scoring is
   span-overlap only. The one exception is definition-usage detection
   (:func:`definition_attachment_recall`), which checks whether a defined *term string* is
   used in the document -- that is detecting usage in the document's own prose, not grading
   a chunker against itself, so it is done on ``sanitize(document.text)[chunk.char_start:
   chunk.char_end]``, never on ``chunk.text``, so a strategy that injects synthesised text
   (e.g. a contextual header) gains nothing from it.
2. No metric reads chunker-emitted metadata to decide whether that chunker "succeeded",
   except the two metrics that are explicitly *about* what a strategy chooses to expose
   (:func:`heading_metrics`, :func:`cross_reference_metrics`). Those return ``None`` (not
   applicable) for a strategy that exposes nothing -- never ``0``, which would look like a
   graded failure rather than an untested claim.
3. A chunk that :attr:`~scaffolder.models.Chunk.located` is ``False`` for is excluded from
   every span-based numerator and denominator. It still counts against
   ``localization_rate`` and is never treated as a success.
"""

from __future__ import annotations

import re
from dataclasses import fields
from typing import TYPE_CHECKING

from scaffolder.gold import overlap_chars, sanitize, verify_document
from scaffolder.models import GoldStructuralMetrics

if TYPE_CHECKING:
    from collections.abc import Sequence

    from scaffolder.gold import GoldAnnotation, GoldClause
    from scaffolder.models import Chunk, ChunkSet, Document

# -- Small shared helpers -----------------------------------------------------------------


def _located_chunks(chunk_set: ChunkSet) -> list[Chunk]:
    """Chunks in ``chunk_set`` that were successfully localised in the document."""
    return [chunk for chunk in chunk_set.chunks if chunk.located]


def _span(chunk: Chunk) -> tuple[int, int]:
    """``(char_start, char_end)`` for a located chunk.

    Only call this on a chunk that has passed a ``chunk.located`` check; it asserts rather
    than returning ``Optional`` so callers (and mypy) don't have to re-check afterwards.
    """
    assert chunk.char_start is not None and chunk.char_end is not None
    return chunk.char_start, chunk.char_end


# -- Identifier normalisation --------------------------------------------------------------

# Words that a UK/US contract uses to introduce a numbered unit. Per gold/SCHEMA.md, plain
# decimal clause numbers ("clause 12.3") carry no word prefix in the gold identifier
# ("12.3"), while article/schedule/section numbers do ("article_4", "schedule_1",
# "section_1.01"). "paragraph" patterns like "clause" -- lettered romanettes are folded into
# their parent clause per SCHEMA.md and never get their own "paragraph_" identifier.
_STRIP_PREFIX_WORDS = ("clause", "paragraph")
_KEEP_PREFIX_WORDS = ("section", "article", "schedule")


def normalize_identifier(identifier: str) -> str:
    """Normalise a clause identifier or cross-reference target for comparison.

    Applies, in order: casefold, collapse internal whitespace to single spaces, strip a
    leading "clause "/"paragraph " word entirely (these carry no prefix in gold
    identifiers -- ``"clause 12.3"`` normalises to ``"12.3"``, matching
    ``target_identifier`` in gold/SCHEMA.md's own example), or fold a leading
    "section "/"article "/"schedule " word into an underscore-joined prefix (these DO
    carry a prefix in gold identifiers -- ``"article 6"`` -> ``"article_6"``,
    ``"Section 4.02"`` -> ``"section_4.02"``), then strip a trailing ``"."``.

    Range: any string in, any string out -- this is a normalisation, not a metric.
    Does not measure anything by itself; it exists so that a claimed identifier coming
    from a strategy's own text (a heading breadcrumb, a cross-reference's raw target
    string) can be compared for equality against a gold ``identifier``/``target_identifier``
    without either side needing to match the other's exact casing or wording.
    """
    text = " ".join(identifier.strip().casefold().split())

    for word in _STRIP_PREFIX_WORDS:
        prefix = f"{word} "
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()
            break
    else:
        for word in _KEEP_PREFIX_WORDS:
            prefix = f"{word} "
            if text.startswith(prefix):
                text = f"{word}_{text[len(prefix) :].strip()}"
                break

    return text.rstrip(".")


# Leading numbering token of one hierarchy-path component. A strategy's breadcrumb reads
# like ``"1.1 - In this Agreement, the following terms..."`` or ``"Section 4.02 - Payment"``,
# so the identifier has to be cut off the front before it can be compared with a gold one.
_PATH_TOKEN_RE = re.compile(
    r"""^\s*
    (?:(?P<word>article|section|schedule|chapter|clause|recital|part)\s+)?
    (?P<num>
        \([a-z]{1,3}\)            # a lettered romanette, e.g. (a), (iv)
      | \d+(?:\.\d+)*             # a decimal number, e.g. 4 or 1.02 or 2.5.1
      | [ivxlcdm]+(?=\b)          # a roman numeral, e.g. IV
    )
    (?P<sub>\([a-z]{1,3}\))?      # a romanette appended to a number, e.g. 1.2(a)
    """,
    re.IGNORECASE | re.VERBOSE,
)

_ROMAN_VALUES = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}

#: Prefixes gold identifiers use for named units. A strategy's breadcrumb often omits the
#: word (``"7.2"`` where gold says ``"section_7.2"``), which is a difference in naming
#: convention, not a wrong claim, so identifier lookup falls back to the bare suffix.
_GOLD_PREFIXES = ("section_", "article_", "schedule_", "chapter_", "recital_", "part_")


def _roman_to_int(token: str) -> int | None:
    """Convert a roman numeral to an integer, or return None if it is not one."""
    token = token.casefold()
    if not token or any(ch not in _ROMAN_VALUES for ch in token):
        return None
    total = 0
    previous = 0
    for ch in reversed(token):
        value = _ROMAN_VALUES[ch]
        total += -value if value < previous else value
        previous = max(previous, value)
    return total


def _component_token(component: str) -> str | None:
    """Extract the numbering token that opens one hierarchy-path component.

    ``"1.1 - In this Agreement..."`` -> ``"1.1"``; ``"Section 4.02 - Payment"`` ->
    ``"section 4.02"``; ``"(a) - references to clauses..."`` -> ``"(a)"``;
    ``"ARTICLE IV - Fees"`` -> ``"article 4"``. Returns ``None`` for a component that
    opens with no numbering at all (``"preamble"``), which is not a claim about any
    annotated clause.
    """
    match = _PATH_TOKEN_RE.match(component)
    if match is None:
        return None
    num = match.group("num")
    if not num.startswith("(") and "." not in num and not num.isdigit():
        arabic = _roman_to_int(num)
        if arabic is None:
            return None
        num = str(arabic)
    token = f"{num}{match.group('sub') or ''}"
    word = match.group("word")
    return f"{word} {token}" if word else token


def claimed_identifier(hierarchy_path: str) -> str | None:
    """Compose the clause identifier a strategy's breadcrumb claims for a chunk.

    A breadcrumb is a ``">"``-separated path whose components carry a numbering token
    followed by heading text, for example
    ``"2 - Services > 2.5 - The Client shall: > (a) - provide the Supplier with..."``.
    Reading only the last component would yield ``"(a)"``, which names nothing; the path
    has to be walked so the romanette is attached to the sub-clause that owns it, giving
    ``"2.5(a)"``. A component whose token replaces rather than extends the accumulated
    identifier (a new numbered clause) resets it.

    Returns ``None`` when no component carries a numbering token — the strategy has made
    no claim about an annotated clause.
    """
    accumulated: str | None = None
    for component in hierarchy_path.split(">"):
        token = _component_token(component)
        if token is None:
            continue
        if token.startswith("(") and accumulated is not None:
            accumulated = f"{accumulated}{token}"
        else:
            accumulated = token
    return accumulated


def _gold_lookup(gold: GoldAnnotation) -> dict[str, list[GoldClause]]:
    """Index gold clauses by normalised identifier and by their bare numbering suffix.

    Both forms are indexed because gold identifiers follow each document's own convention
    (``"12.3"`` in a UK agreement, ``"section_7.2"`` in a US one) while a strategy's
    breadcrumb usually carries only the number. Treating ``"7.2"`` as a wrong claim about
    ``section_7.2`` would measure naming convention, not segmentation. Suffix keys can be
    ambiguous — ``chapter_1`` and ``article_1`` both reduce to ``1`` — so every key maps to
    a list and the caller disambiguates by span overlap.
    """
    index: dict[str, list[GoldClause]] = {}
    for clause in gold.clauses:
        keys = {normalize_identifier(clause.identifier)}
        for prefix in _GOLD_PREFIXES:
            if clause.identifier.startswith(prefix):
                keys.add(normalize_identifier(clause.identifier[len(prefix) :]))
        for key in keys:
            index.setdefault(key, []).append(clause)
    return index


# -- 1. Clause fragmentation ----------------------------------------------------------------


def clause_fragmentation_rate(chunk_set: ChunkSet, gold: GoldAnnotation) -> float:
    """Fraction of gold leaf clauses that no single chunk substantially covers.

    Formula: over eligible leaf clauses (``gold.leaf_clauses`` at least 20 characters
    long -- leaf clauses are the atomic units; a non-leaf clause's own span is often just
    a heading line and would be trivially contained by any chunk touching it), a clause is
    *fragmented* when::

        max(overlap_chars(chunk_span, clause_span) for chunk in located chunks)
            / (clause.char_end - clause.char_start) < 0.80

    Returns fragmented / eligible; ``0.0`` when there are no eligible leaf clauses.

    Range: ``[0.0, 1.0]``. Direction: lower is better (less fragmentation).

    Does not measure: over-merging. A single chunk spanning the entire document contains
    100% of every leaf clause and scores ``0.0`` here -- see
    :func:`top_level_over_merge_rate` for the metric that penalises that case. Unlocated
    chunks never contribute a covering chunk, so they can only push this rate up, never
    down.
    """
    located = _located_chunks(chunk_set)
    eligible = [
        clause for clause in gold.leaf_clauses if (clause.char_end - clause.char_start) >= 20
    ]
    if not eligible:
        return 0.0

    fragmented = 0
    for clause in eligible:
        clause_len = clause.char_end - clause.char_start
        best_overlap = 0
        for chunk in located:
            start, end = _span(chunk)
            overlap = overlap_chars(start, end, clause.char_start, clause.char_end)
            best_overlap = max(best_overlap, overlap)
            if best_overlap / clause_len >= 0.80:
                break
        if best_overlap / clause_len < 0.80:
            fragmented += 1

    return fragmented / len(eligible)


# -- 2. Top-level over-merge ------------------------------------------------------------------


def top_level_over_merge_rate(
    chunk_set: ChunkSet, gold: GoldAnnotation, *, min_overlap_chars: int = 50
) -> float:
    """Fraction of located chunks that substantially span more than one top-level clause.

    Formula: over located chunks, a chunk *over-merges* when its span overlaps the
    **subtree spans** (``gold.subtree_span``, i.e. a top-level clause plus all its
    descendants) of more than one top-level gold clause by at least ``min_overlap_chars``
    characters each. Returns over-merging chunks / located chunks.

    Returns ``0.0`` when there are no located chunks -- check ``localization_rate``
    alongside this metric to distinguish "no over-merging happened" from "nothing could be
    scored".

    Range: ``[0.0, 1.0]``. Direction: lower is better (less swallowing of multiple clauses
    into one chunk).

    Does not measure: fragmentation. This is the metric the superseded harness lacked
    entirely -- it is what catches a chunker that merges several clauses into one chunk,
    which :func:`clause_fragmentation_rate` alone cannot see (a whole-document chunk scores
    a perfect ``0.0`` fragmentation rate).
    """
    located = _located_chunks(chunk_set)
    if not located:
        return 0.0

    subtree_spans = [gold.subtree_span(clause) for clause in gold.top_level_clauses]

    over_merging = 0
    for chunk in located:
        start, end = _span(chunk)
        hits = sum(
            1
            for subtree_start, subtree_end in subtree_spans
            if overlap_chars(start, end, subtree_start, subtree_end) >= min_overlap_chars
        )
        if hits > 1:
            over_merging += 1

    return over_merging / len(located)


# -- 3. Sub-clause grouping (informational) ----------------------------------------------------


def sub_clause_grouping_rate(chunk_set: ChunkSet, gold: GoldAnnotation) -> float:
    """Fraction of located chunks that fully contain two or more sibling leaf clauses.

    Formula: over located chunks, a chunk counts when it contains (overlap
    ``>= 95%`` of the clause's own characters) two or more leaf clauses that share the
    same ``parent``. Returns that count / located chunks.

    Range: ``[0.0, 1.0]``. Direction: **informational, no direction.** A high value is
    neither good nor bad on its own -- it says a strategy tends to keep sibling
    sub-clauses in one chunk, which trades off directly against
    :func:`top_level_over_merge_rate` (grouping enough siblings together eventually spans
    more than one top-level clause). Report it next to that metric, not instead of it.

    Does not measure: whether grouping siblings is *appropriate* for a given clause --
    that depends on document semantics this harness does not model.
    """
    located = _located_chunks(chunk_set)
    if not located:
        return 0.0

    siblings = [clause for clause in gold.leaf_clauses if clause.parent is not None]

    grouping = 0
    for chunk in located:
        start, end = _span(chunk)
        parent_counts: dict[str, int] = {}
        for clause in siblings:
            clause_len = clause.char_end - clause.char_start
            if clause_len <= 0:
                continue
            ratio = overlap_chars(start, end, clause.char_start, clause.char_end) / clause_len
            if ratio >= 0.95:
                assert clause.parent is not None
                parent_counts[clause.parent] = parent_counts.get(clause.parent, 0) + 1
        if any(count >= 2 for count in parent_counts.values()):
            grouping += 1

    return grouping / len(located)


# -- 4. Heading recall / precision (n/a for strategies with no heading metadata) ---------------


def heading_metrics(
    chunk_set: ChunkSet, gold: GoldAnnotation
) -> tuple[float | None, float | None, int]:
    """Recall/precision of a strategy's self-reported ``section_hierarchy`` claims.

    Only strategies that *expose* a heading can be scored here, since this is one of the
    two metrics that is explicitly about what a strategy chooses to surface (see the
    module docstring, rule 2). If no chunk in ``chunk_set`` carries a
    ``metadata["section_hierarchy"]`` string, returns ``(None, None, len(gold.clauses))``
    -- not applicable, never a ``0``.

    Otherwise, for each *located* chunk with that key, the breadcrumb is walked by
    :func:`claimed_identifier` to compose the identifier it claims (so a trailing ``"(a)"``
    is attached to the sub-clause that owns it rather than read on its own), and that
    identifier is resolved against the gold clauses by :func:`_gold_lookup`, which accepts
    both the document's own naming convention and the bare numbering suffix. The claim is
    **correct** when some gold clause with that identifier exists AND its own span overlaps
    the chunk's span by at least 1 character.

    Only chunks that overlap some gold clause at all make a scoreable claim. A chunk lying
    entirely in unannotated front matter (a title block or recitals, which
    ``gold/SCHEMA.md`` deliberately excludes) is skipped rather than counted as a wrong
    claim, since there is no annotation for it to be right about.

    * ``precision`` = correct claims / total claims made by located chunks (``1.0`` when a
      strategy exposes the key but no located chunk ends up making a claim -- vacuously,
      nothing wrong was claimed).
    * ``recall`` = (distinct gold clauses correctly claimed) / (gold clauses that some
      located chunk overlaps by >= 50 characters) -- ``n_gold_headings`` reports that
      denominator, so a strategy is not penalised for headings on clauses nobody's chunks
      even reached. ``1.0`` when that denominator is 0.

    Range: each of ``recall``/``precision`` is ``None`` or in ``[0.0, 1.0]``. Direction:
    higher is better for both.

    Does not measure: heading quality or wording -- only whether the claimed identifier
    resolves to the right gold clause. Unlocated chunks never contribute a claim (rule 3).
    """
    has_headings = any(
        isinstance(chunk.metadata.get("section_hierarchy"), str) for chunk in chunk_set.chunks
    )
    if not has_headings:
        return None, None, len(gold.clauses)

    located = _located_chunks(chunk_set)
    lookup = _gold_lookup(gold)

    claims: list[tuple[Chunk, str]] = []
    for chunk in located:
        raw = chunk.metadata.get("section_hierarchy")
        if not isinstance(raw, str) or not raw.strip():
            continue
        start, end = _span(chunk)
        touches_annotated_body = any(
            overlap_chars(start, end, clause.char_start, clause.char_end) >= 1
            for clause in gold.clauses
        )
        if not touches_annotated_body:
            continue
        claimed = claimed_identifier(raw)
        if claimed:
            claims.append((chunk, claimed))

    correct_identifiers: set[str] = set()
    correct_claims = 0
    for chunk, claimed in claims:
        candidates = lookup.get(normalize_identifier(claimed), [])
        start, end = _span(chunk)
        for gold_clause in candidates:
            if overlap_chars(start, end, gold_clause.char_start, gold_clause.char_end) >= 1:
                correct_claims += 1
                correct_identifiers.add(gold_clause.identifier)
                break

    precision = correct_claims / len(claims) if claims else 1.0

    overlapped_gold_ids = {
        clause.identifier
        for clause in gold.clauses
        if any(
            overlap_chars(*_span(chunk), clause.char_start, clause.char_end) >= 50
            for chunk in located
        )
    }
    n_gold_headings = len(overlapped_gold_ids)
    recall = (
        len(correct_identifiers & overlapped_gold_ids) / n_gold_headings if n_gold_headings else 1.0
    )

    return recall, precision, n_gold_headings


# -- 5. Definition attachment ------------------------------------------------------------------


def _has_definition_metadata(chunk: Chunk, term: str) -> bool:
    """True when ``chunk.metadata`` carries the actual definition text for ``term``.

    Checks ``metadata["defined_terms_context"]`` and ``metadata["definitions"]``, each
    expected to be a ``dict[str, str]`` mapping term -> definition text. A bare list of
    term *names* (no definition text) does not count -- knowing a term is defined
    somewhere is not the same as the chunk carrying its definition.
    """
    for key in ("defined_terms_context", "definitions"):
        mapping = chunk.metadata.get(key)
        if isinstance(mapping, dict):
            value = mapping.get(term)
            if isinstance(value, str) and value.strip():
                return True
    return False


def definition_attachment_recall(
    chunk_set: ChunkSet, document: Document, gold: GoldAnnotation
) -> tuple[float, int]:
    """Fraction of defined-term *uses* for which the definition is available in-chunk.

    For each located chunk and each gold defined term, a **use** is counted when the term
    string occurs (case-insensitively, on word boundaries) anywhere in
    ``sanitize(document.text)[chunk.char_start:chunk.char_end]`` at an offset outside the
    term's own ``definition_span`` -- checked on the document text sliced by the chunk's
    span, never on ``chunk.text``, so a strategy cannot manufacture uses (or attachment) by
    injecting text (rule 1 in the module docstring). At most one use is counted per
    (chunk, term) pair.

    A use is **attached** when the definition is available to a reader of that chunk:
    either the chunk's span overlaps the term's ``definition_span`` by at least 80% of the
    definition's characters, or the chunk's metadata carries the definition text itself
    (see :func:`_has_definition_metadata`).

    Returns ``(recall, n_definition_uses)`` where recall = attached uses / total uses,
    and ``1.0`` when there are no uses at all (vacuous).

    Range: ``[0.0, 1.0]``. Direction: higher is better.

    Does not measure: over-merging. A strategy that emits one enormous chunk per document
    trivially attaches every definition to every use and scores perfectly here -- read
    this metric next to :func:`top_level_over_merge_rate`, not in isolation.
    """
    text = sanitize(document.text)
    located = _located_chunks(chunk_set)

    total_uses = 0
    attached_uses = 0

    for term in gold.defined_terms:
        pattern = re.compile(rf"\b{re.escape(term.term)}\b", re.IGNORECASE)
        definition_len = term.definition_end - term.definition_start

        for chunk in located:
            start, end = _span(chunk)
            chunk_text = text[start:end]
            used = any(
                not (term.definition_start <= start + match.start() < term.definition_end)
                for match in pattern.finditer(chunk_text)
            )
            if not used:
                continue

            total_uses += 1
            overlap = overlap_chars(start, end, term.definition_start, term.definition_end)
            has_span_overlap = definition_len > 0 and overlap / definition_len >= 0.80
            if has_span_overlap or _has_definition_metadata(chunk, term.term):
                attached_uses += 1

    if total_uses == 0:
        return 1.0, 0
    return attached_uses / total_uses, total_uses


# -- 6. Cross-reference recall / precision (n/a for strategies with no cross-ref metadata) -----


def cross_reference_metrics(
    chunk_set: ChunkSet, gold: GoldAnnotation
) -> tuple[float | None, float | None, int]:
    """Recall/precision of a strategy's self-reported cross-reference targets.

    Only strategies that emit cross-reference targets can be scored (rule 2). If no chunk
    carries a ``metadata["cross_references"]`` list, returns
    ``(None, None, n_gold_cross_refs)`` where ``n_gold_cross_refs`` is the number of gold
    cross-references with a non-``null`` ``target_identifier`` (internal references only --
    external references, e.g. to a statute, have no in-document target to check and are
    excluded from both this count and every numerator/denominator below).

    Otherwise: for each located chunk, the gold references *in scope* are those whose
    ``char_start`` falls inside the chunk's span.

    * A gold reference is **recalled** when the chunk in whose scope it falls emits an
      entry (``{"raw_text": ..., "target": ...}``) whose normalised target
      (:func:`normalize_identifier`) equals the reference's normalised
      ``target_identifier``. ``recall`` = recalled references / ``n_gold_cross_refs``
      (``1.0`` when there are no internal gold references).
    * ``precision`` = emitted entries whose normalised target matches some gold reference
      in scope of that same chunk / total emitted entries across all located chunks
      (``1.0`` when no located chunk emits anything).

    Range: each of ``recall``/``precision`` is ``None`` or in ``[0.0, 1.0]``. Direction:
    higher is better for both.

    Does not measure: reference *kind* (schedule vs. definition vs. clause) or whether the
    ``raw_text`` itself was extracted correctly -- only whether the claimed target
    resolves to the right gold clause. A reference whose owning chunk was never located is
    simply not recalled (rule 3); it does not shrink ``n_gold_cross_refs``.
    """
    internal_refs = [ref for ref in gold.cross_references if ref.target_identifier is not None]
    n_gold_cross_refs = len(internal_refs)

    has_cross_refs = any(
        isinstance(chunk.metadata.get("cross_references"), list) for chunk in chunk_set.chunks
    )
    if not has_cross_refs:
        return None, None, n_gold_cross_refs

    located = _located_chunks(chunk_set)

    recalled_indices: set[int] = set()
    total_emitted = 0
    correct_emitted = 0

    for chunk in located:
        start, end = _span(chunk)
        refs_in_scope = [
            (index, ref) for index, ref in enumerate(internal_refs) if start <= ref.char_start < end
        ]

        emitted_targets: list[str] = []
        emitted_raw = chunk.metadata.get("cross_references")
        if isinstance(emitted_raw, list):
            for entry in emitted_raw:
                if isinstance(entry, dict):
                    target = entry.get("target")
                    if isinstance(target, str) and target:
                        emitted_targets.append(normalize_identifier(target))

        scope_targets = {
            normalize_identifier(ref.target_identifier)
            for _, ref in refs_in_scope
            if ref.target_identifier is not None
        }

        for target in emitted_targets:
            total_emitted += 1
            if target in scope_targets:
                correct_emitted += 1

        emitted_target_set = set(emitted_targets)
        for index, ref in refs_in_scope:
            assert ref.target_identifier is not None
            if normalize_identifier(ref.target_identifier) in emitted_target_set:
                recalled_indices.add(index)

    recall = len(recalled_indices) / n_gold_cross_refs if n_gold_cross_refs else 1.0
    precision = correct_emitted / total_emitted if total_emitted else 1.0

    return recall, precision, n_gold_cross_refs


# -- 7. Chunk size uniformity (purely descriptive) ----------------------------------------------


def chunk_size_cv(chunk_set: ChunkSet) -> float:
    """Coefficient of variation (population std / mean) of chunk character counts.

    Returns ``0.0`` for fewer than 2 chunks, or when the mean is 0.

    Range: ``[0.0, inf)``. Direction: **purely descriptive, not a quality signal.** A low
    value only means the chunks are uniformly sized, which is exactly what a naive
    fixed-size splitter gives you by construction -- it says nothing about whether chunk
    boundaries land anywhere sensible. Read it alongside the segmentation metrics above,
    never as a substitute for them.
    """
    sizes = [chunk.char_count for chunk in chunk_set.chunks]
    if len(sizes) < 2:
        return 0.0
    mean = sum(sizes) / len(sizes)
    if mean == 0:
        return 0.0
    variance = sum((size - mean) ** 2 for size in sizes) / len(sizes)
    std_dev = float(variance**0.5)
    return std_dev / mean


# -- Aggregation ---------------------------------------------------------------------------


def compute_gold_structural_metrics(
    chunk_set: ChunkSet, document: Document, gold: GoldAnnotation
) -> GoldStructuralMetrics:
    """Compute every gold-derived structural metric for one chunk set on one document.

    Calls :func:`scaffolder.gold.verify_document` first so a fixture edit can never be
    silently scored against stale gold spans. Every field is documented on
    :class:`~scaffolder.models.GoldStructuralMetrics`; ``None`` there means "not
    applicable to this strategy", never "scored zero".
    """
    verify_document(gold, document.text)

    chunks = chunk_set.chunks
    located = _located_chunks(chunk_set)
    total = len(chunks)

    localization_rate_value = len(located) / total if total else 0.0
    total_chars = sum(chunk.char_count for chunk in chunks)
    located_chars = sum(end - start for start, end in (_span(chunk) for chunk in located))
    localization_coverage_value = located_chars / total_chars if total_chars else 0.0

    heading_recall, heading_precision, n_gold_headings = heading_metrics(chunk_set, gold)
    definition_recall, n_definition_uses = definition_attachment_recall(chunk_set, document, gold)
    xref_recall, xref_precision, n_gold_cross_refs = cross_reference_metrics(chunk_set, gold)

    return GoldStructuralMetrics(
        strategy=chunk_set.strategy,
        document_id=chunk_set.document_id,
        chunk_count=total,
        avg_chunk_chars=chunk_set.avg_chunk_size,
        chunk_size_cv=chunk_size_cv(chunk_set),
        located_chunks=len(located),
        localization_rate=localization_rate_value,
        localization_coverage=localization_coverage_value,
        n_leaf_clauses=len(gold.leaf_clauses),
        clause_fragmentation_rate=clause_fragmentation_rate(chunk_set, gold),
        n_top_level_clauses=len(gold.top_level_clauses),
        top_level_over_merge_rate=top_level_over_merge_rate(chunk_set, gold),
        sub_clause_grouping_rate=sub_clause_grouping_rate(chunk_set, gold),
        heading_recall=heading_recall,
        heading_precision=heading_precision,
        n_gold_headings=n_gold_headings,
        n_definition_uses=n_definition_uses,
        definition_attachment_recall=definition_recall,
        n_gold_cross_refs=n_gold_cross_refs,
        xref_target_recall=xref_recall,
        xref_target_precision=xref_precision,
    )


def aggregate_gold_metrics(metrics: Sequence[GoldStructuralMetrics]) -> dict[str, float | None]:
    """Macro-average every numeric field of ``metrics`` (one strategy, many documents).

    Every field of :class:`~scaffolder.models.GoldStructuralMetrics` except ``strategy``
    and ``document_id`` is averaged unweighted across the given metrics. If *any* document
    is ``None`` (not applicable) for a field, the aggregate for that field is ``None`` --
    an n/a on one document is not something a macro-average is allowed to paper over.

    Returns an empty dict when ``metrics`` is empty.
    """
    if not metrics:
        return {}

    skip = ("strategy", "document_id")
    field_names = [f.name for f in fields(GoldStructuralMetrics) if f.name not in skip]

    aggregated: dict[str, float | None] = {}
    for name in field_names:
        values = [getattr(metric, name) for metric in metrics]
        if any(value is None for value in values):
            aggregated[name] = None
        else:
            aggregated[name] = sum(float(value) for value in values) / len(values)

    return aggregated
