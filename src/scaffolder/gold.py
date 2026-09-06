"""Gold annotation loading, validation, and chunk-span localisation.

Gold annotations record where a fixture document's *own* numbering places clause
boundaries, defined terms, and cross-references. They are produced independently
of LexiChunk (see ``gold/SCHEMA.md``) and are the ground truth every structural
metric is scored against.

``sanitize`` is the single implementation of the text normalisation every offset
in ``gold/*.json`` is relative to, mirroring ``LegalChunker._sanitize_input``:
strip the BOM, normalise line endings, then apply Unicode NFC normalisation.

``locate_chunks`` maps arbitrary chunk texts (from any strategy) back onto
document offsets *without* trusting any chunker's self-reported spans -- some
strategies (e.g. LexiChunk's heading breadcrumbs) prepend synthesised text that
is not verbatim in the document, so localisation must be robust to a chunk being
a near-match rather than an exact substring.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


class GoldError(Exception):
    """Raised for any problem loading, validating, or verifying gold annotations."""


# -- Text normalisation -------------------------------------------------------


def sanitize(text: str) -> str:
    """Normalise raw fixture text the same way ``LegalChunker._sanitize_input`` does.

    Every offset recorded in a gold annotation is relative to the output of this
    function, applied in this exact order: strip BOMs, normalise line endings,
    then apply Unicode NFC normalisation.
    """
    text = text.replace("\ufeff", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return unicodedata.normalize("NFC", text)


# -- Data model ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GoldClause:
    """A single annotated clause span (excludes descendant spans)."""

    identifier: str
    level: int
    parent: str | None
    char_start: int
    char_end: int
    is_top_level: bool
    heading: str


@dataclass(frozen=True, slots=True)
class GoldDefinedTerm:
    """A defined term and the span of its definition."""

    term: str
    definition_start: int
    definition_end: int


@dataclass(frozen=True, slots=True)
class GoldCrossReference:
    """An explicit reference from one part of the document to another."""

    raw_text: str
    char_start: int
    target_identifier: str | None
    kind: str

    @property
    def char_end(self) -> int:
        """Exclusive end offset of ``raw_text`` in the sanitised document."""
        return self.char_start + len(self.raw_text)


@dataclass(frozen=True, slots=True)
class GoldAnnotation:
    """The full gold annotation for one fixture document."""

    document_id: str
    source_file: str
    text_sha256: str
    n_chars: int
    numbering_style: str
    clauses: tuple[GoldClause, ...]
    defined_terms: tuple[GoldDefinedTerm, ...]
    cross_references: tuple[GoldCrossReference, ...]
    _by_identifier: Mapping[str, GoldClause] = field(init=False, repr=False, compare=False)
    _children: Mapping[str, tuple[GoldClause, ...]] = field(init=False, repr=False, compare=False)
    _leaf_clauses: tuple[GoldClause, ...] = field(init=False, repr=False, compare=False)
    _top_level_clauses: tuple[GoldClause, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        by_identifier = {clause.identifier: clause for clause in self.clauses}
        children: dict[str, list[GoldClause]] = {clause.identifier: [] for clause in self.clauses}
        for clause in self.clauses:
            if clause.parent is not None:
                children[clause.parent].append(clause)
        children_tuples = {key: tuple(value) for key, value in children.items()}
        leaf_clauses = tuple(
            clause for clause in self.clauses if not children_tuples.get(clause.identifier)
        )
        top_level_clauses = tuple(clause for clause in self.clauses if clause.is_top_level)
        object.__setattr__(self, "_by_identifier", by_identifier)
        object.__setattr__(self, "_children", children_tuples)
        object.__setattr__(self, "_leaf_clauses", leaf_clauses)
        object.__setattr__(self, "_top_level_clauses", top_level_clauses)

    @property
    def by_identifier(self) -> Mapping[str, GoldClause]:
        """Map of clause identifier to clause."""
        return self._by_identifier

    def children(self, identifier: str) -> tuple[GoldClause, ...]:
        """Direct children of the clause with the given identifier."""
        return self._children.get(identifier, ())

    @property
    def leaf_clauses(self) -> tuple[GoldClause, ...]:
        """Clauses with no children, in document order."""
        return self._leaf_clauses

    @property
    def top_level_clauses(self) -> tuple[GoldClause, ...]:
        """Clauses with ``is_top_level`` set, in document order."""
        return self._top_level_clauses

    def subtree_span(self, clause: GoldClause) -> tuple[int, int]:
        """The clause's own span extended to the end of its last descendant.

        Clause spans are ordered and non-overlapping, so a subtree is always
        contiguous, but this walks the descendant set explicitly rather than
        assuming adjacency in the ``clauses`` list.
        """
        end = clause.char_end
        stack = list(self.children(clause.identifier))
        while stack:
            current = stack.pop()
            end = max(end, current.char_end)
            stack.extend(self.children(current.identifier))
        return clause.char_start, end


def verify_document(gold: GoldAnnotation, text: str) -> None:
    """Verify that ``text`` (after sanitisation) still matches ``gold``.

    Raises ``GoldError`` if the sanitised character count or hash no longer
    matches, meaning the fixture changed since the annotation was made.
    """
    sanitized = sanitize(text)
    if len(sanitized) != gold.n_chars:
        raise GoldError(
            f"document {gold.document_id!r} has {len(sanitized)} sanitised characters but its "
            f"gold annotation expects {gold.n_chars}; the fixture changed since the annotation "
            f"was made"
        )
    digest = hashlib.sha256(sanitized.encode("utf-8")).hexdigest()
    if digest != gold.text_sha256:
        raise GoldError(
            f"document {gold.document_id!r} does not hash to its gold annotation's "
            f"text_sha256 ({digest} != {gold.text_sha256}); the fixture changed since the "
            f"annotation was made"
        )


# -- Loading & validation -------------------------------------------------------

_TOP_LEVEL_KEYS = {
    "document_id",
    "source_file",
    "text_sha256",
    "n_chars",
    "numbering_style",
    "clauses",
    "defined_terms",
    "cross_references",
}

_CROSS_REF_KINDS = {
    "internal_clause",
    "internal_schedule",
    "internal_definition",
    "external_statute",
    "external_document",
}

_gold_cache: dict[tuple[Path, str], GoldAnnotation] = {}


def clear_gold_cache() -> None:
    """Clear the module-level gold annotation cache. Intended for tests."""
    _gold_cache.clear()


def load_gold(document_id: str, gold_dir: str | Path = "gold") -> GoldAnnotation:
    """Load and validate the gold annotation for ``document_id``.

    Results are cached by ``(resolved gold_dir, document_id)``; use
    ``clear_gold_cache`` to reset between tests.
    """
    resolved_dir = Path(gold_dir).resolve()
    cache_key = (resolved_dir, document_id)
    cached = _gold_cache.get(cache_key)
    if cached is not None:
        return cached

    path = resolved_dir / f"{document_id}.json"
    if not path.is_file():
        raise GoldError(f"gold annotation file not found for document {document_id!r}: {path}")

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise GoldError(
            f"gold annotation file for {document_id!r} is not valid JSON: {path}"
        ) from exc

    annotation = _parse_gold(document_id, raw, path)
    _gold_cache[cache_key] = annotation
    return annotation


def load_all_gold(gold_dir: str | Path = "gold") -> dict[str, GoldAnnotation]:
    """Load every ``gold/<document_id>.json`` file in ``gold_dir``."""
    resolved_dir = Path(gold_dir).resolve()
    if not resolved_dir.is_dir():
        raise GoldError(f"gold directory not found: {resolved_dir}")

    return {
        path.stem: load_gold(path.stem, gold_dir=resolved_dir)
        for path in sorted(resolved_dir.glob("*.json"))
    }


def _parse_gold(document_id: str, raw: object, path: Path) -> GoldAnnotation:
    if not isinstance(raw, dict):
        raise GoldError(f"gold annotation for {document_id!r} must be a JSON object: {path}")

    unknown_keys = {key for key in raw if not key.startswith("_")} - _TOP_LEVEL_KEYS
    if unknown_keys:
        raise GoldError(
            f"gold annotation for {document_id!r} has unknown top-level keys "
            f"{sorted(unknown_keys)}: {path}"
        )

    try:
        file_document_id = raw["document_id"]
        source_file = raw["source_file"]
        text_sha256 = raw["text_sha256"]
        n_chars = raw["n_chars"]
        numbering_style = raw["numbering_style"]
        raw_clauses = raw["clauses"]
    except KeyError as exc:
        raise GoldError(
            f"gold annotation for {document_id!r} is missing required key {exc}: {path}"
        ) from exc
    raw_defined_terms = raw.get("defined_terms", [])
    raw_cross_references = raw.get("cross_references", [])

    if file_document_id != document_id:
        raise GoldError(
            f"gold annotation file {path} declares document_id {file_document_id!r}, "
            f"expected {document_id!r}"
        )

    clauses = _parse_clauses(document_id, raw_clauses, n_chars, path)
    by_identifier = {clause.identifier: clause for clause in clauses}
    defined_terms = _parse_defined_terms(document_id, raw_defined_terms, n_chars, path)
    cross_references = _parse_cross_references(
        document_id, raw_cross_references, n_chars, by_identifier, path
    )

    return GoldAnnotation(
        document_id=document_id,
        source_file=source_file,
        text_sha256=text_sha256,
        n_chars=n_chars,
        numbering_style=numbering_style,
        clauses=clauses,
        defined_terms=defined_terms,
        cross_references=cross_references,
    )


def _parse_clauses(
    document_id: str, raw_clauses: object, n_chars: int, path: Path
) -> tuple[GoldClause, ...]:
    if not isinstance(raw_clauses, list):
        raise GoldError(f"'clauses' must be a list in gold annotation for {document_id!r}: {path}")

    clauses: list[GoldClause] = []
    seen: dict[str, GoldClause] = {}
    previous_end = -1
    for index, item in enumerate(raw_clauses):
        try:
            identifier = item["identifier"]
            level = item["level"]
            parent = item["parent"]
            char_start = item["char_start"]
            char_end = item["char_end"]
            is_top_level = item["is_top_level"]
            heading = item["heading"]
        except KeyError as exc:
            raise GoldError(
                f"clause #{index} in gold annotation for {document_id!r} is missing key "
                f"{exc}: {path}"
            ) from exc

        if identifier in seen:
            raise GoldError(
                f"duplicate clause identifier {identifier!r} in gold annotation for "
                f"{document_id!r}: {path}"
            )

        if not (0 <= char_start < char_end <= n_chars):
            raise GoldError(
                f"clause {identifier!r} in {document_id!r} has an out-of-bounds or inverted "
                f"span [{char_start}, {char_end}) for n_chars={n_chars}: {path}"
            )

        if char_start < previous_end:
            raise GoldError(
                f"clauses in gold annotation for {document_id!r} are unsorted or overlap at "
                f"{identifier!r} (char_start={char_start} precedes the previous clause's "
                f"char_end={previous_end}): {path}"
            )

        if bool(is_top_level) != (level == 1):
            raise GoldError(
                f"clause {identifier!r} in {document_id!r} has is_top_level={is_top_level} "
                f"inconsistent with level={level}: {path}"
            )

        if parent is None:
            if level != 1:
                raise GoldError(
                    f"clause {identifier!r} in {document_id!r} has no parent but level="
                    f"{level} (expected level 1 for a clause without a parent): {path}"
                )
        else:
            parent_clause = seen.get(parent)
            if parent_clause is None:
                raise GoldError(
                    f"clause {identifier!r} in {document_id!r} has parent {parent!r} which is "
                    f"absent or does not appear earlier in 'clauses': {path}"
                )
            if parent_clause.level != level - 1:
                raise GoldError(
                    f"clause {identifier!r} in {document_id!r} has parent {parent!r} at level "
                    f"{parent_clause.level}, expected exactly one level up (level {level - 1}): "
                    f"{path}"
                )

        clause = GoldClause(
            identifier=identifier,
            level=level,
            parent=parent,
            char_start=char_start,
            char_end=char_end,
            is_top_level=is_top_level,
            heading=heading,
        )
        clauses.append(clause)
        seen[identifier] = clause
        previous_end = char_end

    return tuple(clauses)


def _parse_defined_terms(
    document_id: str, raw_terms: object, n_chars: int, path: Path
) -> tuple[GoldDefinedTerm, ...]:
    if not isinstance(raw_terms, list):
        raise GoldError(
            f"'defined_terms' must be a list in gold annotation for {document_id!r}: {path}"
        )

    terms: list[GoldDefinedTerm] = []
    for index, item in enumerate(raw_terms):
        try:
            term = item["term"]
            span = item["definition_span"]
        except KeyError as exc:
            raise GoldError(
                f"defined_term #{index} in gold annotation for {document_id!r} is missing key "
                f"{exc}: {path}"
            ) from exc

        if not (isinstance(span, list) and len(span) == 2):
            raise GoldError(
                f"defined_term {term!r} in {document_id!r} has an invalid definition_span "
                f"(expected a [start, end] pair): {path}"
            )
        start, end = span
        if not (0 <= start < end <= n_chars):
            raise GoldError(
                f"defined_term {term!r} in {document_id!r} has an out-of-bounds or inverted "
                f"definition_span [{start}, {end}) for n_chars={n_chars}: {path}"
            )
        terms.append(GoldDefinedTerm(term=term, definition_start=start, definition_end=end))

    return tuple(terms)


def _parse_cross_references(
    document_id: str,
    raw_refs: object,
    n_chars: int,
    by_identifier: Mapping[str, GoldClause],
    path: Path,
) -> tuple[GoldCrossReference, ...]:
    if not isinstance(raw_refs, list):
        raise GoldError(
            f"'cross_references' must be a list in gold annotation for {document_id!r}: {path}"
        )

    refs: list[GoldCrossReference] = []
    for index, item in enumerate(raw_refs):
        try:
            raw_text = item["raw_text"]
            char_start = item["char_start"]
            target_identifier = item["target_identifier"]
            kind = item["kind"]
        except KeyError as exc:
            raise GoldError(
                f"cross_reference #{index} in gold annotation for {document_id!r} is missing "
                f"key {exc}: {path}"
            ) from exc

        if kind not in _CROSS_REF_KINDS:
            raise GoldError(
                f"cross_reference #{index} in {document_id!r} has unknown kind {kind!r}: {path}"
            )

        char_end = char_start + len(raw_text)
        if not (char_start >= 0 and char_end <= n_chars):
            raise GoldError(
                f"cross_reference #{index} in {document_id!r} has an out-of-bounds span "
                f"[{char_start}, {char_end}) for n_chars={n_chars}: {path}"
            )

        if kind.startswith("internal_") and target_identifier not in by_identifier:
            raise GoldError(
                f"cross_reference #{index} in {document_id!r} of kind {kind!r} has "
                f"target_identifier {target_identifier!r} which does not resolve to any "
                f"clause in this file: {path}"
            )

        refs.append(
            GoldCrossReference(
                raw_text=raw_text,
                char_start=char_start,
                target_identifier=target_identifier,
                kind=kind,
            )
        )

    return tuple(refs)


# -- Chunk span localisation ----------------------------------------------------


@dataclass(frozen=True, slots=True)
class ChunkSpan:
    """Where a chunk's text was located in the sanitised document."""

    chunk_index: int
    char_start: int
    char_end: int
    covered_chars: int
    exact: bool


def _smallest_suffix_index(chunk: str, text: str) -> int:
    """Smallest ``i`` such that ``chunk[i:]`` occurs in ``text``.

    ``chunk[i:]`` occurring in ``text`` is monotone in ``i`` (a substring of a
    substring is a substring), so this holds trivially at ``i == len(chunk)``
    (the empty string) and binary search finds the smallest working ``i``.
    """
    lo, hi = 0, len(chunk)
    while lo < hi:
        mid = (lo + hi) // 2
        if chunk[mid:] in text:
            hi = mid
        else:
            lo = mid + 1
    return lo


def _largest_prefix_index(chunk: str, text: str) -> int:
    """Largest ``j`` such that ``chunk[:j]`` occurs in ``text``.

    Symmetric to ``_smallest_suffix_index``: monotone decreasing in ``j``, holds
    trivially at ``j == 0``.
    """
    lo, hi = 0, len(chunk)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if chunk[:mid] in text:
            lo = mid
        else:
            hi = mid - 1
    return lo


def _locate_substring(candidate: str, text: str, cursor: int) -> int:
    idx = text.find(candidate, cursor)
    if idx == -1:
        idx = text.find(candidate)
    return idx


def _locate_suffix(chunk: str, text: str, cursor: int) -> tuple[int, int, int] | None:
    i = _smallest_suffix_index(chunk, text)
    covered = len(chunk) - i
    if covered < 0.5 * len(chunk):
        return None
    candidate = chunk[i:]
    idx = _locate_substring(candidate, text, cursor)
    if idx == -1:
        return None
    return idx, idx + len(candidate), covered


def _locate_prefix(chunk: str, text: str, cursor: int) -> tuple[int, int, int] | None:
    j = _largest_prefix_index(chunk, text)
    covered = j
    if covered < 0.5 * len(chunk):
        return None
    candidate = chunk[:j]
    idx = _locate_substring(candidate, text, cursor)
    if idx == -1:
        return None
    return idx, idx + len(candidate), covered


def _span_distance(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    """How far apart two spans are: 0 if they touch or overlap."""
    if a_end <= b_start:
        return b_start - a_end
    if b_end <= a_start:
        return a_start - b_end
    return 0


def _locate_one(chunk: str, text: str, cursor: int) -> tuple[int, int, int, bool] | None:
    exact_idx = _locate_substring(chunk, text, cursor)
    if exact_idx != -1:
        return exact_idx, exact_idx + len(chunk), len(chunk), True

    suffix = _locate_suffix(chunk, text, cursor)
    prefix = _locate_prefix(chunk, text, cursor)

    if suffix is not None and prefix is not None:
        s_start, s_end, s_covered = suffix
        p_start, p_end, p_covered = prefix
        if _span_distance(s_start, s_end, p_start, p_end) <= 2 * len(chunk):
            start = min(s_start, p_start)
            end = max(s_end, p_end)
            covered = max(s_covered, p_covered)
            return start, end, covered, False
        # Both matched but land far apart in the document: they cannot both be
        # describing the same occurrence, so trust whichever covers more of
        # the chunk's own text.
        return (
            (s_start, s_end, s_covered, False)
            if s_covered >= p_covered
            else (
                p_start,
                p_end,
                p_covered,
                False,
            )
        )

    if suffix is not None:
        start, end, covered = suffix
        return start, end, covered, False

    if prefix is not None:
        start, end, covered = prefix
        return start, end, covered, False

    return None


def locate_chunks(chunk_texts: Sequence[str], text: str) -> tuple[ChunkSpan | None, ...]:
    """Locate every chunk's text in the sanitised document text.

    Strategy-reported offsets are never used: this locates chunks purely from
    their text, uniformly across strategies. See the module docstring and
    ``gold/SCHEMA.md`` for why (some strategies synthesise non-verbatim text
    such as heading breadcrumbs).
    """
    spans: list[ChunkSpan | None] = []
    cursor = 0
    for index, chunk in enumerate(chunk_texts):
        if not chunk.strip():
            spans.append(None)
            cursor = 0
            continue

        located = _locate_one(chunk, text, cursor)
        if located is None:
            spans.append(None)
            cursor = 0
            continue

        char_start, char_end, covered_chars, exact = located
        spans.append(ChunkSpan(index, char_start, char_end, covered_chars, exact))
        cursor = max(cursor, char_end)

    return tuple(spans)


def localization_rate(spans: Sequence[ChunkSpan | None]) -> float:
    """Fraction of chunks that were located at all."""
    if not spans:
        return 0.0
    return sum(1 for span in spans if span is not None) / len(spans)


def localization_coverage(spans: Sequence[ChunkSpan | None], chunk_texts: Sequence[str]) -> float:
    """Fraction of total chunk-text characters accounted for by located spans."""
    total_chars = sum(len(chunk) for chunk in chunk_texts)
    if total_chars == 0:
        return 0.0
    covered = sum(span.covered_chars for span in spans if span is not None)
    return covered / total_chars


def overlap_chars(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    """Number of characters two ``[start, end)`` spans have in common."""
    return max(0, min(a_end, b_end) - max(a_start, b_start))
