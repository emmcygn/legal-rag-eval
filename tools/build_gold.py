#!/usr/bin/env python3
"""Seed gold annotation JSON files from each fixture document's own numbering.

Standalone, stdlib-only. Never imports lexichunk or legal_rag_eval. See gold/README.md
and gold/SCHEMA.md for the contract this script implements.

Usage:
    python tools/build_gold.py                 # writes gold/*.json for all fixtures
    python tools/build_gold.py --doc us_msa     # one document
    python tools/build_gold.py --check          # re-seed in memory, diff vs gold/, exit 1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = REPO_ROOT / "src" / "legal_rag_eval" / "fixtures" / "documents"
GOLD_DIR = REPO_ROOT / "gold"

DOCS = {
    "eu_gdpr_excerpt": "eu_regulation",
    "uk_service_agreement": "uk_decimal",
    "uk_terms_conditions": "uk_decimal",
    "us_msa": "us_article_section",
    "us_terms_of_service": "us_section_decimal",
}

# ---------------------------------------------------------------------------
# Sanitisation (must match LegalChunker._sanitize_input exactly)
# ---------------------------------------------------------------------------


def sanitize(raw: str) -> str:
    text = raw.replace("﻿", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = unicodedata.normalize("NFC", text)
    return text


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------

ROMAN_VALUES = [
    (1000, "M"),
    (900, "CM"),
    (500, "D"),
    (400, "CD"),
    (100, "C"),
    (90, "XC"),
    (50, "L"),
    (40, "XL"),
    (10, "X"),
    (9, "IX"),
    (5, "V"),
    (4, "IV"),
    (1, "I"),
]


def roman_to_int(s: str) -> int:
    s = s.upper()
    values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    total = 0
    prev = 0
    for ch in reversed(s):
        v = values[ch]
        if v < prev:
            total -= v
        else:
            total += v
            prev = v
    return total


BLANK_LINE_RE = re.compile(r"\n[ \t]*\n")


def is_paragraph_start(text: str, line_start_abs: int) -> bool:
    """True if the line beginning at ``line_start_abs`` opens a new paragraph,
    i.e. it is the first line of the document or the previous line is blank."""
    if line_start_abs == 0:
        return True
    return (
        line_start_abs >= 2
        and text[line_start_abs - 1] == "\n"
        and text[line_start_abs - 2] == "\n"
    )


def paragraph_end(text: str, start: int, limit: int | None = None) -> int:
    """Offset of the next blank-line boundary at/after ``start``, or ``limit``/len(text)."""
    hard_limit = len(text) if limit is None else min(limit, len(text))
    m = BLANK_LINE_RE.search(text, start, hard_limit)
    if m:
        return m.start()
    return hard_limit


def clean_heading(s: str) -> str:
    s = " ".join(s.split())
    s = s.strip(" \t-–—:")
    return s.strip()


def extract_heading_from_body(
    text: str, body_start: int, zone_end: int, split_on_period: bool
) -> str:
    """Heading = first line of body text, or (for the "Heading. Body" US style)
    the phrase up to the first sentence-ending period within 120 chars."""
    end = paragraph_end(text, body_start, zone_end)
    raw = text[body_start:end]
    collapsed = " ".join(raw.split())
    if split_on_period:
        m = re.search(r"\.(?=\s|$)", collapsed[:120])
        if m and m.start() > 0:
            return clean_heading(collapsed[: m.start()])[:120]
    return clean_heading(collapsed)[:120]


def line_start(text: str, pos: int) -> int:
    nl = text.rfind("\n", 0, pos)
    return nl + 1


# ---------------------------------------------------------------------------
# Clause dataclass-ish dict + finalisation
# ---------------------------------------------------------------------------


def make_clause(
    identifier: str,
    level: int,
    parent: str | None,
    char_start: int,
    heading: str,
    marker_start: int,
    marker_end: int,
) -> dict:
    return {
        "identifier": identifier,
        "level": level,
        "parent": parent,
        "char_start": char_start,
        "char_end": None,  # filled in finalize_clauses
        "is_top_level": level == 1,
        "heading": heading,
        "_marker_span": (marker_start, marker_end),
    }


def finalize_clauses(clauses: list[dict], text_len: int) -> list[dict]:
    clauses = sorted(clauses, key=lambda c: c["char_start"])
    for i, c in enumerate(clauses):
        if i + 1 < len(clauses):
            c["char_end"] = clauses[i + 1]["char_start"]
        else:
            c["char_end"] = text_len
    return clauses


# ---------------------------------------------------------------------------
# Lettered-item scanning shared by uk_decimal / us_article_section / us_section_decimal
# ---------------------------------------------------------------------------

LETTER_LINE_RE = re.compile(r"(?m)^([ \t]*)\(([a-z])\)[ \t]+")


def scan_letter_items(text: str, zone_start: int, zone_end: int):
    """Yield (letter, marker_start, marker_end, body_start) for lettered items that
    genuinely start a new list entry within [zone_start, zone_end): either the item is
    'a' preceded by a blank line, or it continues an already-started a,b,c,... chain."""
    expected_next = None
    for m in LETTER_LINE_RE.finditer(text, zone_start, zone_end):
        letter = m.group(2)
        abs_line_start = m.start()
        blank_before = is_paragraph_start(text, abs_line_start)
        if blank_before and letter == "a" or expected_next is not None and letter == expected_next:
            accept = True
        else:
            accept = False
        if accept:
            marker_start = m.start(2) - 1  # position of '('
            marker_end = m.end()
            yield letter, marker_start, marker_end, m.end()
            expected_next = chr(ord(letter) + 1)
        else:
            if not (blank_before and letter == "a"):
                expected_next = None


# ---------------------------------------------------------------------------
# EU regulation seeder
# ---------------------------------------------------------------------------

RECITAL_RE = re.compile(r"(?m)^\((\d+)\)[ \t]+")
CHAPTER_RE = re.compile(r"(?m)^CHAPTER[ \t]+([IVXLCDM]+)[ \t]*$")
ARTICLE_EU_RE = re.compile(r"(?m)^Article[ \t]+(\d+)[ \t]*$")
NUM_PARA_RE = re.compile(r"(?m)^(\d+)\.[ \t]+")


def seed_eu_regulation(text: str) -> list[dict]:
    clauses: list[dict] = []

    chapter_matches = list(CHAPTER_RE.finditer(text))
    first_chapter_start = chapter_matches[0].start() if chapter_matches else len(text)

    # Recitals: only before the first CHAPTER heading.
    for m in RECITAL_RE.finditer(text, 0, first_chapter_start):
        ls = line_start(text, m.start())
        if not is_paragraph_start(text, ls):
            continue
        n = m.group(1)
        marker_start, marker_end = m.start(), m.end()
        heading = extract_heading_from_body(text, marker_end, len(text), split_on_period=False)
        clauses.append(
            make_clause(f"recital_{n}", 1, None, marker_start, heading, marker_start, marker_end)
        )

    # Chapters
    chapter_ids = []  # (char_start, identifier)
    for m in chapter_matches:
        ls = line_start(text, m.start())
        if not is_paragraph_start(text, ls):
            continue
        n = roman_to_int(m.group(1))
        marker_start, marker_end = m.start(), m.end()
        # Heading is the next physical line.
        next_line_start = (
            marker_end + 1 if marker_end < len(text) and text[marker_end] == "\n" else marker_end
        )
        heading_end = text.find("\n", next_line_start)
        if heading_end == -1:
            heading_end = len(text)
        heading = clean_heading(text[next_line_start:heading_end])[:120]
        ident = f"chapter_{n}"
        clauses.append(make_clause(ident, 1, None, marker_start, heading, marker_start, marker_end))
        chapter_ids.append((marker_start, ident))

    # Articles (own line, arabic), and numbered paragraphs within each article's zone.
    article_matches = list(ARTICLE_EU_RE.finditer(text))
    for idx, m in enumerate(article_matches):
        ls = line_start(text, m.start())
        if not is_paragraph_start(text, ls):
            continue
        n = m.group(1)
        marker_start, marker_end = m.start(), m.end()
        next_line_start = (
            marker_end + 1 if marker_end < len(text) and text[marker_end] == "\n" else marker_end
        )
        heading_end = text.find("\n", next_line_start)
        if heading_end == -1:
            heading_end = len(text)
        heading = clean_heading(text[next_line_start:heading_end])[:120]
        ident = f"article_{n}"

        parent = None
        for cstart, cident in chapter_ids:
            if cstart < marker_start:
                parent = cident
            else:
                break
        clauses.append(
            make_clause(ident, 2, parent, marker_start, heading, marker_start, marker_end)
        )

        zone_end = article_matches[idx + 1].start() if idx + 1 < len(article_matches) else len(text)
        # Numbered paragraphs "1.", "2." inside this article.
        body_scan_start = heading_end
        for pm in NUM_PARA_RE.finditer(text, body_scan_start, zone_end):
            pls = line_start(text, pm.start())
            if not is_paragraph_start(text, pls):
                continue
            pn = pm.group(1)
            p_marker_start, p_marker_end = pm.start(), pm.end()
            p_heading = extract_heading_from_body(
                text, p_marker_end, zone_end, split_on_period=False
            )
            clauses.append(
                make_clause(
                    f"article_{n}_{pn}",
                    3,
                    ident,
                    p_marker_start,
                    p_heading,
                    p_marker_start,
                    p_marker_end,
                )
            )

    return clauses


# ---------------------------------------------------------------------------
# UK decimal seeder (uk_service_agreement, uk_terms_conditions)
# ---------------------------------------------------------------------------

TOP_CLAUSE_RE = re.compile(r"(?m)^(\d+)\.[ \t]+(?=\S)")
SUB_CLAUSE_RE = re.compile(r"(?m)^(\d+)\.(\d+)[ \t]+(?=\S)")
SCHEDULE_RE = re.compile(r"(?m)^SCHEDULE[ \t]+(\d+)\b[ \t]*(.*)$")


def seed_uk_decimal(text: str) -> list[dict]:
    clauses: list[dict] = []

    schedule_matches = list(SCHEDULE_RE.finditer(text))
    first_schedule_start = schedule_matches[0].start() if schedule_matches else len(text)

    # --- Body clauses (before the first SCHEDULE) ---
    sub_matches = [
        m
        for m in SUB_CLAUSE_RE.finditer(text, 0, first_schedule_start)
        if is_paragraph_start(text, line_start(text, m.start()))
    ]
    top_matches = [
        m
        for m in TOP_CLAUSE_RE.finditer(text, 0, first_schedule_start)
        if is_paragraph_start(text, line_start(text, m.start()))
        and not SUB_CLAUSE_RE.match(text, m.start())
    ]

    top_ids: list[tuple[int, str]] = []
    for m in top_matches:
        n = m.group(1)
        marker_start, marker_end = m.start(), m.end()
        heading = extract_heading_from_body(
            text, marker_end, first_schedule_start, split_on_period=False
        )
        clauses.append(make_clause(n, 1, None, marker_start, heading, marker_start, marker_end))
        top_ids.append((marker_start, n))

    sub_ids: list[tuple[int, str]] = []
    for m in sub_matches:
        parent = None
        for s, n in top_ids:
            if s < m.start():
                parent = n
            else:
                break
        ident = f"{m.group(1)}.{m.group(2)}"
        marker_start, marker_end = m.start(), m.end()
        heading = extract_heading_from_body(
            text, marker_end, first_schedule_start, split_on_period=False
        )
        clauses.append(
            make_clause(ident, 2, parent, marker_start, heading, marker_start, marker_end)
        )
        sub_ids.append((marker_start, ident))

    # Lettered items within each N.M zone.
    sub_starts_sorted = sorted(sub_ids, key=lambda t: t[0])
    for i, (sstart, sident) in enumerate(sub_starts_sorted):
        zone_end = (
            sub_starts_sorted[i + 1][0] if i + 1 < len(sub_starts_sorted) else first_schedule_start
        )
        # also bounded by the next top-level clause if that comes first
        for tstart, _tid in top_ids:
            if sstart < tstart < zone_end:
                zone_end = tstart
                break
        for letter, mstart, mend, body_start in scan_letter_items(text, sstart, zone_end):
            ident = f"{sident}({letter})"
            heading = extract_heading_from_body(text, body_start, zone_end, split_on_period=False)
            clauses.append(make_clause(ident, 3, sident, mstart, heading, mstart, mend))

    # --- Schedules ---
    for sidx, m in enumerate(schedule_matches):
        if not is_paragraph_start(text, line_start(text, m.start())):
            continue
        snum = m.group(1)
        marker_start, marker_end = m.start(), m.end()
        heading = clean_heading(m.group(2))[:120]
        sched_ident = f"schedule_{snum}"
        clauses.append(
            make_clause(sched_ident, 1, None, marker_start, heading, marker_start, marker_end)
        )

        sched_zone_end = (
            schedule_matches[sidx + 1].start() if sidx + 1 < len(schedule_matches) else len(text)
        )

        sched_sub_matches = [
            mm
            for mm in SUB_CLAUSE_RE.finditer(text, marker_end, sched_zone_end)
            if is_paragraph_start(text, line_start(text, mm.start()))
        ]
        sched_top_matches = [
            mm
            for mm in TOP_CLAUSE_RE.finditer(text, marker_end, sched_zone_end)
            if is_paragraph_start(text, line_start(text, mm.start()))
            and not SUB_CLAUSE_RE.match(text, mm.start())
        ]

        sched_top_ids: list[tuple[int, str]] = []
        for mm in sched_top_matches:
            n = mm.group(1)
            mstart, mend = mm.start(), mm.end()
            heading2 = extract_heading_from_body(text, mend, sched_zone_end, split_on_period=False)
            ident = f"{sched_ident}.{n}"
            clauses.append(make_clause(ident, 2, sched_ident, mstart, heading2, mstart, mend))
            sched_top_ids.append((mstart, ident))

        sched_sub_ids: list[tuple[int, str]] = []
        for mm in sched_sub_matches:
            parent = None
            for s, ident0 in sched_top_ids:
                if s < mm.start():
                    parent = ident0
                else:
                    break
            n1, n2 = mm.group(1), mm.group(2)
            mstart, mend = mm.start(), mm.end()
            heading2 = extract_heading_from_body(text, mend, sched_zone_end, split_on_period=False)
            ident = f"{sched_ident}.{n1}.{n2}"
            clauses.append(make_clause(ident, 3, parent, mstart, heading2, mstart, mend))
            sched_sub_ids.append((mstart, ident))
        # Lettered items under schedule sub-paragraphs are beyond the level-3 cap; folded
        # into their parent's span (not separately annotated).

    return clauses


# ---------------------------------------------------------------------------
# US article/section seeder (us_msa)
# ---------------------------------------------------------------------------

ARTICLE_US_RE = re.compile(r"(?m)^ARTICLE[ \t]+([IVXLCDM]+)[ \t]*$")
SECTION_NN_LETTER_RE = re.compile(r"(?m)^Section[ \t]+(\d+)\.(\d+)\(([a-z])\)[ \t]+(?=\S)")
SECTION_NN_RE = re.compile(r"(?m)^Section[ \t]+(\d+)\.(\d+)[ \t]+(?=\S)")
EXHIBIT_MARKER_RE = re.compile(r"(?m)^EXHIBIT[ \t]+[A-Z0-9]+\b")


def seed_us_article_section(text: str) -> list[dict]:
    clauses: list[dict] = []

    exhibit_m = EXHIBIT_MARKER_RE.search(text)
    doc_end_for_structure = exhibit_m.start() if exhibit_m else len(text)

    article_matches = [
        m
        for m in ARTICLE_US_RE.finditer(text, 0, doc_end_for_structure)
        if is_paragraph_start(text, line_start(text, m.start()))
    ]
    article_ids: list[tuple[int, str]] = []
    for m in article_matches:
        n = roman_to_int(m.group(1))
        marker_start, marker_end = m.start(), m.end()
        next_line_start = (
            marker_end + 1 if marker_end < len(text) and text[marker_end] == "\n" else marker_end
        )
        heading_end = text.find("\n", next_line_start)
        if heading_end == -1:
            heading_end = len(text)
        heading = clean_heading(text[next_line_start:heading_end])[:120]
        ident = f"article_{n}"
        clauses.append(make_clause(ident, 1, None, marker_start, heading, marker_start, marker_end))
        article_ids.append((marker_start, ident))

    # Section N.NN and explicit Section N.NN(x) headings.
    # (start, end) of "Section N.NN(x)" marker matches, so they are not counted twice
    letter_heading_spans = []
    section_matches = []
    for m in SECTION_NN_LETTER_RE.finditer(text, 0, doc_end_for_structure):
        section_matches.append(("letter", m))
        letter_heading_spans.append((m.start(), m.end()))
    for m in SECTION_NN_RE.finditer(text, 0, doc_end_for_structure):
        if any(s <= m.start() < e for s, e in letter_heading_spans):
            continue
        section_matches.append(("plain", m))
    section_matches.sort(key=lambda t: t[1].start())

    plain_section_positions: list[tuple[int, str]] = []
    for kind, m in section_matches:
        n, nn = m.group(1), m.group(2)
        marker_start, marker_end = m.start(), m.end()
        parent = None
        for s, aident in article_ids:
            if s < marker_start:
                parent = aident
            else:
                break
        if kind == "plain":
            ident = f"section_{n}.{nn}"
            heading = extract_heading_from_body(
                text, marker_end, doc_end_for_structure, split_on_period=True
            )
            clauses.append(
                make_clause(ident, 2, parent, marker_start, heading, marker_start, marker_end)
            )
            plain_section_positions.append((marker_start, ident))
        else:
            letter = m.group(3)
            parent_section = f"section_{n}.{nn}"
            ident = f"section_{n}.{nn}({letter})"
            heading = extract_heading_from_body(
                text, marker_end, doc_end_for_structure, split_on_period=True
            )
            clauses.append(
                make_clause(
                    ident, 3, parent_section, marker_start, heading, marker_start, marker_end
                )
            )

    # Lettered items (bare "(x)" paragraphs) within each plain Section's zone.
    plain_sorted = sorted(plain_section_positions, key=lambda t: t[0])
    all_section_marker_starts = sorted([m.start() for _, m in section_matches])
    for sstart, sident in plain_sorted:
        # zone ends at the next Section marker (of any kind) or the exhibit boundary.
        later = [p for p in all_section_marker_starts if p > sstart]
        zone_end = later[0] if later else doc_end_for_structure
        for letter, mstart, mend, body_start in scan_letter_items(text, sstart, zone_end):
            ident = f"{sident}({letter})"
            heading = extract_heading_from_body(text, body_start, zone_end, split_on_period=False)
            clauses.append(make_clause(ident, 3, sident, mstart, heading, mstart, mend))

    return clauses


# ---------------------------------------------------------------------------
# US section-decimal seeder (us_terms_of_service)
# ---------------------------------------------------------------------------

SECTION_TOP_RE = re.compile(r"(?m)^Section[ \t]+(\d+)\.[ \t]+(?=\S)")
SECTION_SUB_RE = re.compile(r"(?m)^Section[ \t]+(\d+)\.(\d+)[ \t]+(?=\S)")


def seed_us_section_decimal(text: str) -> list[dict]:
    clauses: list[dict] = []

    sub_matches = [
        m
        for m in SECTION_SUB_RE.finditer(text)
        if is_paragraph_start(text, line_start(text, m.start()))
    ]
    top_matches = [
        m
        for m in SECTION_TOP_RE.finditer(text)
        if is_paragraph_start(text, line_start(text, m.start()))
        and not SECTION_SUB_RE.match(text, m.start())
    ]

    top_ids: list[tuple[int, str]] = []
    for m in top_matches:
        n = m.group(1)
        marker_start, marker_end = m.start(), m.end()
        ident = f"section_{n}"
        heading = extract_heading_from_body(text, marker_end, len(text), split_on_period=False)
        clauses.append(make_clause(ident, 1, None, marker_start, heading, marker_start, marker_end))
        top_ids.append((marker_start, ident))

    sub_ids: list[tuple[int, str]] = []
    for m in sub_matches:
        n, nn = m.group(1), m.group(2)
        parent = None
        for s, ident0 in top_ids:
            if s < m.start():
                parent = ident0
            else:
                break
        marker_start, marker_end = m.start(), m.end()
        ident = f"section_{n}.{nn}"
        heading = extract_heading_from_body(text, marker_end, len(text), split_on_period=True)
        clauses.append(
            make_clause(ident, 2, parent, marker_start, heading, marker_start, marker_end)
        )
        sub_ids.append((marker_start, ident))

    all_marker_starts = sorted([m.start() for m in sub_matches] + [m.start() for m in top_matches])
    sub_sorted = sorted(sub_ids, key=lambda t: t[0])
    for sstart, sident in sub_sorted:
        later = [p for p in all_marker_starts if p > sstart]
        zone_end = later[0] if later else len(text)
        for letter, mstart, mend, body_start in scan_letter_items(text, sstart, zone_end):
            ident = f"{sident}({letter})"
            heading = extract_heading_from_body(text, body_start, zone_end, split_on_period=False)
            clauses.append(make_clause(ident, 3, sident, mstart, heading, mstart, mend))

    return clauses


SEEDERS = {
    "eu_regulation": seed_eu_regulation,
    "uk_decimal": seed_uk_decimal,
    "us_article_section": seed_us_article_section,
    "us_section_decimal": seed_us_section_decimal,
}


# ---------------------------------------------------------------------------
# Defined terms
# ---------------------------------------------------------------------------

DEFINITION_RE = re.compile(
    r"(?P<quote>[\"'])(?P<term>[^\"'\n]{1,120}?)(?P=quote)\s+(?:means|shall mean|refers to)\b"
)


def seed_defined_terms(text: str) -> list[dict]:
    sites = []  # (start, end_of_match, term)
    for m in DEFINITION_RE.finditer(text):
        sites.append((m.start(), m.group("term")))
    sites.sort(key=lambda t: t[0])

    terms = []
    for i, (start, term) in enumerate(sites):
        para_end = paragraph_end(text, start)
        next_start = sites[i + 1][0] if i + 1 < len(sites) else len(text)
        end = min(para_end, next_start) if next_start > start else para_end
        end = max(end, start + 1)
        terms.append({"term": term, "definition_span": [start, end]})
    return terms


# ---------------------------------------------------------------------------
# Cross references
# ---------------------------------------------------------------------------

EXTERNAL_PHRASES = [
    (re.compile(r"the\s+Data\s+Protection\s+Act\s+2018"), "external_statute"),
    (re.compile(r"Regulation\s*\(EU\)\s*2016/679"), "external_statute"),
    (
        re.compile(r"\b(?:the|an|any)\s+applicable\s+Order\s+Form|\b(?:the|an|any)\s+Order\s+Form"),
        "external_document",
    ),
]

_LIST_SEP = r"(?:\s*,)?(?:\s+(?:and|&|through))?\s*"
CLAUSE_LIST_RE = re.compile(
    r"\b[Cc]lauses?\b\s+((?:\d+(?:\.\d+)?(?:\([a-z]\))?" + _LIST_SEP + r")+)"
)
CLAUSE_TOKEN_RE = re.compile(r"\d+(?:\.\d+)?(?:\([a-z]\))?")
SCHEDULE_REF_RE = re.compile(r"\bSchedule\s+(\d+)\b")
PARA_OF_SCHEDULE_RE = re.compile(
    r"\bparagraphs?\s+(\d+(?:\.\d+)?)\s+of\s+Schedule\s+(\d+)\b", re.IGNORECASE
)
PARA_ABOVE_RE = re.compile(r"\bparagraphs?\s+(\d+(?:\.\d+)?)\s+(?:above|below)\b", re.IGNORECASE)

ARTICLE_ROMAN_REF_RE = re.compile(r"\bArticle\s+([IVXLCDM]+)\b")
SECTION_NN_LIST_RE = re.compile(r"\bSections?\s+((?:\d+\.\d+(?:\([a-z]\))?" + _LIST_SEP + r")+)")
SECTION_NN_TOKEN_RE = re.compile(r"\d+\.\d+(?:\([a-z]\))?")

SECTION_DEC_LIST_RE = re.compile(
    r"\bSections?\s+((?:\d+(?:\.\d+)?(?:\([a-z]\))?" + _LIST_SEP + r")+)"
)
SECTION_DEC_TOKEN_RE = re.compile(r"\d+(?:\.\d+)?(?:\([a-z]\))?")

ARTICLE_ARABIC_REF_RE = re.compile(r"\bArticle\s+(\d+)\b")


def overlaps_any(start: int, end: int, spans: list[tuple[int, int]]) -> bool:
    return any(start < e and s < end for s, e in spans)


def seed_cross_references(
    text: str,
    numbering_style: str,
    clause_ids: set,
    marker_spans: list[tuple[int, int]],
    schedule_positions: list[tuple[int, str]] | None = None,
) -> list[dict]:
    refs: list[dict] = []
    schedule_positions = schedule_positions or []

    def add(raw_text: str, start: int, target: str | None, kind: str):
        if overlaps_any(start, start + len(raw_text), marker_spans):
            return
        entry = {
            "raw_text": raw_text,
            "char_start": start,
            "target_identifier": target,
            "kind": kind,
        }
        if kind.startswith("internal_") and target is None:
            entry["_seed_unresolved"] = True
        refs.append(entry)

    # External phrases (checked for every document).
    for pat, kind in EXTERNAL_PHRASES:
        for m in pat.finditer(text):
            add(m.group(0), m.start(), None, kind)

    if numbering_style == "uk_decimal":
        for m in CLAUSE_LIST_RE.finditer(text):
            group_start = m.start(1)
            for tm in CLAUSE_TOKEN_RE.finditer(m.group(1)):
                abs_start = group_start + tm.start()
                raw = tm.group(0)
                target = raw if raw in clause_ids else None
                add(raw, abs_start, target, "internal_clause")

        for m in PARA_OF_SCHEDULE_RE.finditer(text):
            n, sched = m.group(1), m.group(2)
            candidate = f"schedule_{sched}.{n}"
            target = candidate if candidate in clause_ids else None
            add(m.group(0), m.start(), target, "internal_schedule")

        for m in PARA_ABOVE_RE.finditer(text):
            n = m.group(1)
            current_sched = None
            for pos, sid in schedule_positions:
                if pos < m.start():
                    current_sched = sid
                else:
                    break
            target = None
            if current_sched:
                candidate = f"{current_sched}.{n}"
                target = candidate if candidate in clause_ids else None
            add(m.group(0), m.start(), target, "internal_schedule")

        for m in SCHEDULE_REF_RE.finditer(text):
            candidate = f"schedule_{m.group(1)}"
            target = candidate if candidate in clause_ids else None
            add(m.group(0), m.start(), target, "internal_schedule")

    elif numbering_style == "eu_regulation":
        for m in ARTICLE_ARABIC_REF_RE.finditer(text):
            candidate = f"article_{m.group(1)}"
            target = candidate if candidate in clause_ids else None
            add(m.group(0), m.start(), target, "internal_clause")

    elif numbering_style == "us_article_section":
        for m in ARTICLE_ROMAN_REF_RE.finditer(text):
            candidate = f"article_{roman_to_int(m.group(1))}"
            target = candidate if candidate in clause_ids else None
            add(m.group(0), m.start(), target, "internal_clause")

        for m in SECTION_NN_LIST_RE.finditer(text):
            group_start = m.start(1)
            for tm in SECTION_NN_TOKEN_RE.finditer(m.group(1)):
                abs_start = group_start + tm.start()
                raw = tm.group(0)
                candidate = f"section_{raw}"
                target = candidate if candidate in clause_ids else None
                add(raw, abs_start, target, "internal_clause")

    elif numbering_style == "us_section_decimal":
        for m in SECTION_DEC_LIST_RE.finditer(text):
            group_start = m.start(1)
            for tm in SECTION_DEC_TOKEN_RE.finditer(m.group(1)):
                abs_start = group_start + tm.start()
                raw = tm.group(0)
                candidate = f"section_{raw}"
                target = candidate if candidate in clause_ids else None
                add(raw, abs_start, target, "internal_clause")

    # De-duplicate identical (char_start, raw_text) pairs that might arise from
    # overlapping patterns (e.g. a phrase matched by two rules).
    seen = set()
    deduped = []
    for r in sorted(refs, key=lambda x: x["char_start"]):
        key = (r["char_start"], r["raw_text"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)
    return deduped


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def build_gold_for_doc(doc_id: str) -> dict:
    numbering_style = DOCS[doc_id]
    source_file_rel = f"src/legal_rag_eval/fixtures/documents/{doc_id}.txt"
    raw_bytes = (FIXTURES_DIR / f"{doc_id}.txt").read_bytes()
    raw_text = raw_bytes.decode("utf-8")
    text = sanitize(raw_text)

    clauses = SEEDERS[numbering_style](text)
    clauses = finalize_clauses(clauses, len(text))

    marker_spans = [c["_marker_span"] for c in clauses]
    clause_ids = {c["identifier"] for c in clauses}

    schedule_positions = None
    if numbering_style == "uk_decimal":
        schedule_positions = sorted(
            [
                (c["char_start"], c["identifier"])
                for c in clauses
                if re.fullmatch(r"schedule_\d+", c["identifier"])
            ]
        )

    defined_terms = seed_defined_terms(text)
    cross_refs = seed_cross_references(
        text, numbering_style, clause_ids, marker_spans, schedule_positions
    )

    clean_clauses = []
    for c in clauses:
        clean_clauses.append(
            {
                "identifier": c["identifier"],
                "level": c["level"],
                "parent": c["parent"],
                "char_start": c["char_start"],
                "char_end": c["char_end"],
                "is_top_level": c["is_top_level"],
                "heading": c["heading"],
            }
        )
    clean_clauses.sort(key=lambda c: c["char_start"])
    defined_terms.sort(key=lambda d: d["definition_span"][0])
    cross_refs.sort(key=lambda r: r["char_start"])

    return {
        "document_id": doc_id,
        "source_file": source_file_rel,
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "n_chars": len(text),
        "numbering_style": numbering_style,
        "clauses": clean_clauses,
        "defined_terms": defined_terms,
        "cross_references": cross_refs,
    }


def dump_json(data: dict) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


def write_gold(doc_id: str, *, force: bool = False) -> Path | None:
    """Write the seeded annotation for one document.

    Refuses to overwrite an existing file that differs from a fresh seed. That difference
    is the hand-correction pass — the step that makes these files usable as ground truth —
    and re-seeding would silently discard it. Pass ``force=True`` to overwrite deliberately,
    for instance after a fixture genuinely changed and the corrections logged in
    ``gold/CHANGES.md`` have been reconciled by hand.
    """
    data = build_gold_for_doc(doc_id)
    text_out = dump_json(data)
    out_path = GOLD_DIR / f"{doc_id}.json"

    if out_path.exists() and not force:
        existing = out_path.read_text(encoding="utf-8")
        if existing != text_out:
            print(
                f"REFUSED: {out_path.name} differs from a fresh seed, so it carries hand "
                f"corrections (see gold/CHANGES.md). Re-seeding would discard them.\n"
                f"         Pass --force to overwrite deliberately.",
                file=sys.stderr,
            )
            return None

    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text_out)
    return out_path


def check_drift(doc_ids: list[str]) -> bool:
    drift = False
    for doc_id in doc_ids:
        data = build_gold_for_doc(doc_id)
        seeded = dump_json(data)
        existing_path = GOLD_DIR / f"{doc_id}.json"
        if not existing_path.exists():
            print(f"DRIFT: {doc_id}.json does not exist in gold/")
            drift = True
            continue
        existing = existing_path.read_text(encoding="utf-8")
        if existing != seeded:
            print(
                f"DRIFT: {doc_id}.json differs from a fresh seed "
                f"(hand corrections and/or fixture changes)"
            )
            drift = True
    return drift


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc", choices=sorted(DOCS), help="Only process this document")
    parser.add_argument(
        "--check", action="store_true", help="Diff seeded output against gold/ and exit 1 on drift"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite a hand-corrected gold file (discards the corrections in gold/CHANGES.md)",
    )
    args = parser.parse_args()

    doc_ids = [args.doc] if args.doc else sorted(DOCS)

    if args.check:
        drift = check_drift(doc_ids)
        return 1 if drift else 0

    refused = 0
    for doc_id in doc_ids:
        out_path = write_gold(doc_id, force=args.force)
        if out_path is None:
            refused += 1
        else:
            print(f"wrote {out_path}")
    return 1 if refused else 0


if __name__ == "__main__":
    sys.exit(main())
