# Gold annotation corrections

This file is the audit trail for the ground truth. Every gold file in this directory was
first produced mechanically by `scripts/build_gold.py`, a stdlib-only regex seeder that
reads each document's own numbering and never consults LexiChunk. Each file was then read
against its document and corrected by hand. This is the human-in-the-loop step, and it is
what makes these annotations usable as ground truth: a regex over legal prose gets clause
boundaries close, and gets defined terms and cross-references badly wrong.

The corrections below are recorded per document, in the annotator's own words. Where an
annotator checked a category and changed nothing, that is stated too — a silent category
would be indistinguishable from an unchecked one.

Because the committed files carry these corrections, `python scripts/build_gold.py --check`
reports drift by design. The seeder is a reproducible starting point, not the source of
truth. `tests/test_gold_annotations.py` is what actually validates the committed files
against the fixtures, and it runs in CI.

Final counts after correction:

| Document | Characters | Clauses (L1 / L2 / L3) | Leaf clauses | Defined terms | Cross-references |
|---|---:|---:|---:|---:|---:|
| `eu_gdpr_excerpt` | 5,141 | 21 (5 / 6 / 10) | 14 | 6 | 2 |
| `uk_service_agreement` | 25,187 | 103 (10 / 49 / 44) | 77 | 18 | 31 |
| `uk_terms_conditions` | 16,948 | 70 (10 / 40 / 20) | 55 | 16 | 40 |
| `us_msa` | 25,346 | 57 (9 / 38 / 10) | 47 | 31 | 43 |
| `us_terms_of_service` | 25,743 | 114 (11 / 60 / 43) | 92 | 19 | 55 |
| **Total** | **98,365** | **365** | **285** | **90** | **171** |

---

## eu_gdpr_excerpt

Correction pass performed by reading `src/legal_rag_eval/fixtures/documents/eu_gdpr_excerpt.txt`
(sanitised via `legal_rag_eval.gold.sanitize`) against the seeded `gold/eu_gdpr_excerpt.json`,
line by line, using a throwaway inspection script (printed every clause span's boundary
context, every defined-term span, and a regex sweep for candidate cross-references).

- **Clause spans**: 21 checked, no corrections needed. Verified every `char_start` begins
  exactly at the first character of the clause's own number/heading (e.g. `recital_1` at 0
  starts on `"(1) The protection..."`; `article_5` at 3580 starts on `"Article 5\n"`), and every
  `char_end` lands exactly on the first character of the next clause at any level (e.g.
  `article_1_3` ends at 1094, precisely where `article_2`'s heading `"Article 2\nMaterial
  scope\n\n"` begins). No gaps, no overlaps, spans sum correctly to `n_chars` = 5141.
- **Clause hierarchy**: 21 checked, no corrections needed. Confirmed recitals 1-3 are level 1
  with `parent: null`; `CHAPTER I` (GENERAL PROVISIONS) and `CHAPTER II` (PRINCIPLES) are level
  1; Articles 1-4 are level 2 under `chapter_1` (all appear in the document between "CHAPTER I"
  and "CHAPTER II" headings) and Articles 5-6 are level 2 under `chapter_2` (both appear after
  "CHAPTER II"); numbered paragraphs (`1.`, `2.`, `3.`) inside each article are level 3 under
  that article. Lettered sub-items `(a)`-`(f)` within paragraphs (e.g. Article 2(2), Article
  5(1), Article 6(1)) are correctly folded into their parent paragraph's span per the level-3
  cap — the seeder did not over-split on these.
- **Missing/invented clauses**: none found. Walked the document top to bottom; every numbered
  unit present (3 recitals, 2 chapters, 6 articles, 9 numbered paragraphs) has an entry, and no
  entry corresponds to a number that isn't a real heading (e.g. the monetary/list numbers
  inside lettered sub-items were correctly *not* split out as separate clauses).
- **Defined terms**: 6 checked, no corrections needed. All six terms in Article 4 ('Personal
  Data', 'Processing', 'Controller', 'Processor', 'Recipient', 'Consent') have
  `definition_span` starting exactly at the opening quote mark of the term and ending exactly
  before the next term's opening quote (or, for 'Consent', at the end of its sentence before
  the blank line preceding "CHAPTER II"). Every span contains its term and its `means ...`
  definition text. No terms are used-but-undefined in this excerpt that would need exclusion.
- **Cross-references**: seed had `[]` (confirmed correct that the regex seeder found none —
  this document uses no `Article N(m)`-style body cross-references). Corrected by adding the
  2 explicit references the seeder missed:
  - **Added external reference**: `"TEU"` at char_start 1632, inside Article 2(2)(b)'s text
    `"...activities which fall within the scope of Chapter 2 of Title V of the TEU;"`. This
    names an external treaty (Treaty on European Union), so `kind: "external_statute"`,
    `target_identifier: null`. Important to flag as *external* despite containing the literal
    substring "Chapter 2" — that phrase refers to a chapter of the TEU, not to this file's own
    `chapter_2` (PRINCIPLES), so it must never be resolved internally.
  - **Added internal reference**: `"paragraph 1"` at char_start 4338, inside Article 5(2)'s
    text `"The controller shall be responsible for, and be able to demonstrate compliance
    with, paragraph 1."` This is an explicit numbered back-reference to Article 5's own
    paragraph 1, i.e. clause `article_5_1` (not anaphoric — it names the paragraph number
    explicitly, unlike "the foregoing"/"this Article"), so `kind: "internal_clause"`,
    `target_identifier: "article_5_1"`.
  - Verified for both: `text[char_start:char_start+len(raw_text)] == raw_text` holds exactly
    (`text[1632:1635] == "TEU"`, `text[4338:4349] == "paragraph 1"`).
  - No other candidate references exist in this excerpt: no "Article N" body references (only
    heading self-references, correctly excluded), no "Directive 95/46/EC" or "Regulation (EU)
    2016/679" mentions, and the many uses of "this Regulation" are self-referential (the
    document referring to itself), not cross-references to a numbered target.
- **`_seed_unresolved` markers**: none present in the seeded file; nothing to remove.
- **Housekeeping**: `text_sha256` (`06d5053c...376ad9e`) and `n_chars` (5141) were already
  correct against the sanitised fixture text — verified by recomputing both; left unchanged.

### Final counts
- Clauses: 21 total — 5 at level 1 (3 recitals + 2 chapters), 6 at level 2 (articles), 10 at
  level 3 (numbered paragraphs).
- Defined terms: 6.
- Cross-references: 2 — 1 `internal_clause`, 1 `external_statute`.

### Uncertain / flagged for awareness
- The external reference's `raw_text` was set to the minimal unambiguous token `"TEU"` rather
  than the longer phrase `"Chapter 2 of Title V of the TEU"`; I judged the acronym alone is the
  cleanest, uniquely-locatable anchor for the named external instrument, consistent with the
  schema's own example (`"Directive 95/46/EC"`, a short proper-name token). If the harness
  expects the fuller phrase instead, this would need revisiting.
- I classified "TEU" as `external_statute` (a treaty is primary EU law, statute-like) rather
  than `external_document`; the schema doesn't give a treaty example specifically, so this is a
  judgment call rather than a certainty.

Verification: `pytest tests/test_gold_annotations.py -q --no-cov -k eu_gdpr` — 13 passed,
0 failed (including `test_no_seed_markers_remain`).

---

## uk_service_agreement

Correction pass against `src/legal_rag_eval/fixtures/documents/uk_service_agreement.txt`
(sanitised, 25187 chars, sha256 `3676cfa0ce34b08675a537847b4c955266c0aebbbe0ed4ab5a573422a64a9ef9`,
unchanged by this pass). Method: `legal_rag_eval.gold.sanitize()` + a throwaway inspection
script that printed every clause/term/reference with surrounding context, read line by
line against the source text.

### Clause spans
- **Clause spans**: 103 checked (10 level-1, 49 level-2, 44 level-3), no corrections
  needed. Every span begins at the first character of its number/letter, ends exactly
  where the next clause (at any level) begins, and the parties block / BACKGROUND
  recitals (offsets 0–852) are correctly excluded as front matter. Hierarchy is correct
  throughout, including the schedule numbering scheme: `schedule_K` (level 1) →
  `schedule_K.N` for each "N. Heading" section (level 2) → `schedule_K.N.M` for each
  "N.M ..." paragraph (level 3), with lettered `(a)/(b)/(c)` sub-items correctly folded
  into their level-3 parent's span per the level-3 cap in SCHEMA.md. No spurious clauses
  from the parties' `(1)/(2)` numbering, the `(A)/(B)` recitals, or monetary amounts.

### Defined terms
- **Added "Disclosing Party" and "Receiving Party"**: seed had 12 terms and stopped at
  the outer `"X" means ...` matches in clause 1.1; missed the two role-labels nested
  inside the `"Confidential Information"` definition — "...disclosed by one party (the
  "Disclosing Party") to the other party (the "Receiving Party")..." — which are genuine
  quoted-and-defined terms (same nesting pattern the seed already handled correctly for
  `"control"` inside `"Affiliate"`) and are reused throughout clause 4 (4.1, 4.2(b),
  4.3(b)-(d)). Both given the span `[1810, 1921]` covering the sentence that establishes
  both roles.
- **Added "UK GDPR"**: nested inside the `"Data Protection Legislation"` definition —
  "...as amended from time to time (the "UK GDPR")..." — quoted-and-defined via
  apposition, same pattern as above. Span `[2352, 2601]` covers "the UK General Data
  Protection Regulation ... (the "UK GDPR")".
- **Added "Payment Due Date"** (clause 3.3), **"Initial Term"** (clause 6.1), and
  **"Renewal Term"** (clause 6.2): each is quoted-and-defined via the standard UK-drafting
  apposition "(the "X")" / "(each a "X")" outside the main definitions clause, and each
  is reused later in the document (Payment Due Date in 3.4/3.6(a); Initial Term and
  Renewal Term against each other in 6.2/6.3(b)). SCHEMA.md's definition of a defined
  term ("quoted-and-defined") is not restricted to clause 1.1, and these are unambiguous
  instances of the same drafting device, so they were added with spans covering their
  full defining sentence.
- **Not added** (judgment call, flagging for review): "Agreement", "Supplier", "Client" —
  all quoted-and-defined in the front-matter/parties block ("This Services Agreement (the
  "Agreement")...", "...(the "Supplier")...", "...(the "Client")..."). I treated these as
  structural boilerplate identity labels rather than substantive definitions and left them
  out, consistent with the instruction that the parties block is front matter, but this is
  a genuine judgment call — they do technically satisfy "quoted-and-defined" and a
  reviewer could reasonably want them included.
- **Not added**: "Statement of Work" — capitalised consistently throughout (clauses 2.1,
  2.7, 8.1, Schedule 1 §1.1(a)) but never actually introduced with a quoted `"X" means`
  or `(the "X")` construction anywhere in this document (unlike `us_msa.txt`, which does
  define `"Statement of Work"`/`"SOW"`). Per SCHEMA.md, "terms used but never defined in
  the document are not annotated," so it was left out of `defined_terms`. Also not
  annotated as an `external_document` cross-reference, since (unlike "the Order Form" in
  the sibling fixtures) it is never used as a fixed, singular, already-existing
  instrument — it is introduced as an optional future mechanism ("the parties may agree
  additional services by way of a Statement of Work") whose *form* is itself an internal
  schedule reference (Schedule 4, see below).
- Final count: **18 defined terms** (was 12).

### Cross-references
- **Resolved all 4 `_seed_unresolved` markers by deletion**, not reclassification:
  `"Schedule 3"` (×3, at the "Service Levels" definition in 1.1, and twice in Schedule 1
  itself) and `"Schedule 4"` (×1, in clause 2.7). This fixture's document text ends after
  Schedule 2 — Schedules 3 and 4 are named ("Schedule 3 (Service Levels)", "Schedule 4")
  but their content is not part of this fixture, so there is no `schedule_3`/`schedule_4`
  clause to resolve `target_identifier` to, and SCHEMA.md requires
  `internal_*` kinds to resolve to a real identifier in this file. Relabelling them
  `external_document` was rejected as inaccurate — they are structurally identical to
  the (correctly resolved) `"Schedule 1"`/`"Schedule 2"` internal references, not
  references to a separate instrument like "the Order Form"; mislabeling them external
  would misrepresent them for anyone using the `kind` field. Deleting was the only option
  that didn't require either fabricating clauses for content absent from the fixture or
  making a false claim about the reference's nature.
- **Added 4 missed external-statute references**: `"the UK General Data Protection
  Regulation"` (2352) and `"the European Union (Withdrawal) Act 2018"` (2510), both named
  within the "Data Protection Legislation" definition alongside the already-annotated
  "the Data Protection Act 2018"; `"the Insolvency Act 1986"` (15997, clause 6.3(b)) and
  `"the Corporation Tax Act 2010"` (16321, clause 6.3(c)) — both explicitly named statutes
  the seed's regex missed entirely (it only picked up the one Act name already inside the
  definitions clause).
- Checked every `clause`/`Clause`/`Schedule`/`paragraph` mention in the document via a
  regex sweep and confirmed all resolvable internal references were already present and
  correctly targeted (including the multi-target ones — "clauses 5.1 and 5.2" and
  "clauses 4, 5, 7 and 8" — which the seed had already correctly split into one
  cross-reference per number). No additional internal references were missing.
- Verified programmatically for every entry (old and new) that
  `text[char_start:char_start+len(raw_text)] == raw_text`.
- Final count: **31 cross-references** — 21 `internal_clause`, 5 `internal_schedule`,
  5 `external_statute`, 0 `external_document`. Before this pass it was also 31 total, but
  split as 21 `internal_clause` + 5 resolved `internal_schedule` + 4 unresolved
  `internal_schedule` (the deleted ones) + 1 `external_statute`; the count is unchanged
  only because 4 deletions were offset by 4 additions.

### Final counts
- Clauses: 103 total — 10 level 1, 49 level 2, 44 level 3 (unchanged).
- Defined terms: 18 (was 12; +6).
- Cross-references: 31 (was 31; -4 unresolved, +4 new, net 0) — 21 internal_clause,
  5 internal_schedule, 5 external_statute.

### Genuinely unsure about
- Whether "Agreement"/"Supplier"/"Client" (front-matter quoted-and-defined party/document
  labels) should count as `defined_terms`. Left them out as boilerplate identity labels,
  but a reviewer applying SCHEMA.md's "quoted-and-defined" test literally could disagree.
- Whether "Statement of Work" mentions should be `external_document` cross-references
  despite never being formally defined in this document. Left them out because the term
  is never used to point at one specific, already-existing instrument the way "the Order
  Form" is used in the sibling fixtures — it names an optional future mechanism whose
  template is an (absent) internal schedule — but this reading rests on the fixture's own
  internal inconsistency (capitalised throughout, yet never defined), and it is possible
  the fixture intended it to be treated as effectively defined by convention.

### Verification
`pytest tests/test_gold_annotations.py -q --no-cov -k uk_service` — 13 passed, including
`test_no_seed_markers_remain` and `test_internal_cross_reference_targets_resolve`.
Full `tests/test_gold_annotations.py` (all 5 fixtures) — 66 passed.

---

## uk_terms_conditions

Correction pass performed by reading `src/legal_rag_eval/fixtures/documents/uk_terms_conditions.txt`
(sanitised via `legal_rag_eval.gold.sanitize`) against the seeded `gold/uk_terms_conditions.json`,
using a throwaway inspection script that printed every clause span's boundary context, every
defined-term span in full, and every cross-reference's 60-character context, plus targeted
regex sweeps (`^\d+\.\s`, `^\d+\.\d+\s`, `^\s*\([a-z]\)\s`, quoted-phrase scan, `Article/section/
Act/GDPR/Order Form/Documentation/Renewal Term` sweeps) to look for clauses, defined terms and
cross-references the seeder might have invented or missed.

- **Clause spans**: 70 checked (10 level 1, 40 level 2, 20 level 3), no corrections needed.
  Every top-level (`N.` heading), sub-clause (`N.M`), and lettered (`N.M(x)`) span starts exactly
  at the first character of its own number/letter and ends exactly at the first character of the
  next clause at any level — verified both by reading the printed boundary context for all 70
  clauses and by an independent regex sweep over the raw line-start patterns, which found exactly
  10 `^\d+\.\s` headings, 40 `^\d+\.\d+\s` sub-clauses, and 20 `^\s*\([a-z]\)\s` lettered items,
  matching the seeded set position-for-position. No gaps, no overlaps.
- **Clause hierarchy**: confirmed correct throughout — clauses 1-10 are level 1 with
  `parent: null`; each `N.M` has `parent: "N"`; each `N.M(x)` has `parent: "N.M"`.
- **Missing/invented clauses**: none found. In particular checked the specific traps called out
  in the brief: the lettered items `(a)`/`(b)`/`(c)` inside the "Data Protection Legislation"
  definition (clause 1.1) are correctly *not* split into clauses (they are prose inside a defined
  term, not numbered clauses); "Version 3.2" in the document header and the "These Terms and
  Conditions..." / "By clicking 'I Accept'..." preamble paragraphs are correctly left unannotated
  as front matter (first clause starts at char 947); no monetary amounts, dates, or percentages
  at the start of a line were mistaken for clause numbers.
- **Defined terms**: seed had 10 (Account, Authorised Users, Customer Data, Data Protection
  Legislation, Documentation, Fees, Intellectual Property Rights, Order Form, Permitted Purpose,
  Subscription Term) — all 10 verified correct (span starts at the opening quote, ends at the end
  of that definition's paragraph, contains the term). A full quoted-phrase sweep of the sanitised
  text found 22 quoted phrases total; corrected by **adding 6 the seeder missed**:
  - **"Terms"** `[100, 142]` — defined in the preamble: `These Terms and Conditions (these
    "Terms")`. Used as the document's own name throughout every clause.
  - **"Platform"** `[175, 256]` — defined in the same preamble sentence: `the CloudVault Analytics
    Platform and related services (together, the "Platform")`.
  - **"CloudVault"** `[269, 463]` — defined in the preamble: `CloudVault Limited, a company
    incorporated in England and Wales with registered number 11736492 and whose registered
    office is at 50 Broadway, London, SW1H 0BL ("CloudVault", "we", "us", "our")`.
  - **"Customer"** `[649, 741]` — defined in the second preamble paragraph: `the entity or
    individual identified in the applicable Order Form ("Customer", "you", "your")`.
  - **"Personal Data"** `[3004, 3063]` — defined in clause 1.1: `"Personal Data" has the meaning
    given to it in the UK GDPR.` (a valid definition even though it defers to an external
    source for the substantive meaning — the document itself marks the term as defined here).
  - **"UK GDPR"** `[1647, 1833]` — defined parenthetically inside the "Data Protection
    Legislation" definition (clause 1.1): `the UK General Data Protection Regulation (as it forms
    part of domestic law in England and Wales by virtue of section 3 of the European Union
    (Withdrawal) Act 2018) ("UK GDPR")`. This span deliberately overlaps the "Data Protection
    Legislation" span (nothing in `SCHEMA.md` or `tests/test_gold_annotations.py` requires
    defined-term spans to be disjoint — only clause spans must be) and is heavily relied on
    throughout clauses 4.2-4.5.
  - **Judgement call — excluded**: `"I Accept"` (preamble) is a quoted UI action, not a
    definition, and the party-shorthand pronouns `"we"`, `"us"`, `"our"`, `"you"`, `"your"`
    (preamble) are excluded per the brief's explicit guidance — they are redundant alternate
    pronouns for the already-recorded "CloudVault" and "Customer" definitions, not
    independently useful defined terms. This leaves 16 defined terms total (within the 15-25
    range the brief flagged as typical for a definitions clause of this kind).
- **Cross-references**: seed had 31 (not the ~30 estimated in the brief), all 31 verified
  correct — `text[char_start:char_start+len(raw_text)] == raw_text` holds for every one, every
  `internal_clause` target resolves to a real clause identifier in this file, and every
  `external_document`/`external_statute` classification (Order Form, Data Protection Act 2018)
  matches the document text. No `kind` or `target_identifier` corrections were needed on the
  existing 31. Corrected by **adding 9 explicit references the seeder missed**, found via the
  `Article/section/Act/GDPR/Documentation` regex sweeps:
  - **"the UK GDPR"** at char_start 3051 (clause 1.1, inside the "Personal Data" definition),
    7489 (clause 4.3), 8046 (clause 4.5, "Personal Data breach (as defined in the UK GDPR)"),
    and 8219 (clause 4.5, "under the UK GDPR and the Data Protection Act 2018") — 4 references,
    all `kind: "external_statute"`, `target_identifier: null`. (A 5th occurrence at 7290 is
    subsumed by the "Article 28 of the UK GDPR" reference below rather than double-counted.)
  - **"Article 28 of the UK GDPR"** at char_start 7276 (clause 4.2) — `external_statute`,
    `target_identifier: null`.
  - **"section 3 of\n     the European Union (Withdrawal) Act 2018"** at char_start 1762
    (clause 1.1, inside the "Data Protection Legislation"/"UK GDPR" definition) — a distinct
    named external statute (the EU (Withdrawal) Act 2018) cited to explain the legal basis of
    the UK GDPR, analogous to how the seed already treats "the Data Protection Act 2018" inside
    the same definition as a legitimate cross-reference rather than pure definition text.
    `external_statute`, `target_identifier: null`.
  - **"the Late Payment of Commercial Debts (Interest) Act 1998"** at char_start 9203
    (clause 5.3) — a named external statute the seeder missed entirely. `external_statute`,
    `target_identifier: null`.
  - **"the Documentation"** at char_start 4108 (clause 2.1, "in accordance with the
    Documentation") and 9875 (clause 6.1, "the Platform, the Documentation, and all software...")
    — 2 references to the "Documentation" defined term, which (like "Order Form", already
    exhaustively annotated by the seed at all 11 of its occurrences) refers to materials outside
    this Terms document. `external_document`, `target_identifier: null`, matching the brief's own
    example of `"the Documentation"` as an external-document reference.
  - Considered and **excluded**: "its security policy as published at
    https://cloudvault.io/security" (clause 4.4) — a generic descriptive phrase, not a named
    document title in the style of "Order Form"/"Documentation"; judged too low-confidence to
    annotate as a distinct external document (see "Uncertain" below).
- **`_seed_unresolved` markers**: none present in the seeded file; nothing to remove.
- **Housekeeping**: `text_sha256` and `n_chars` (16948) were already correct against the
  sanitised fixture text — the fixture itself was not touched, so both were left unchanged.
  JSON key order, 2-space indent, LF line endings, and trailing newline preserved.

### Final counts
- Clauses: 70 total — 10 at level 1, 40 at level 2, 20 at level 3. No changes from the seed.
- Defined terms: 16 total (10 from the seed + 6 added: Terms, Platform, CloudVault, Customer,
  Personal Data, UK GDPR).
- Cross-references: 40 total (31 from the seed + 9 added) — 22 `internal_clause`,
  9 `external_document`, 9 `external_statute`.

### Uncertain / flagged for awareness
- Including "CloudVault" and "Customer" as defined terms (rather than treating them like the
  excluded pronoun shorthands) is a judgement call: they are the actual party names given a
  quoted shorthand definition in the preamble, used as the operative subject/object of nearly
  every clause, so I judged them genuinely useful defined terms distinct from the purely
  pronominal "we"/"us"/"our"/"you"/"your". If the harness intends *only* clause-1.1-style `"X"
  means ...` definitions to count, these two (plus "Terms" and "Platform") should be removed.
- The "UK GDPR" definition span overlaps the "Data Protection Legislation" span by design (both
  are defined in the same sentence, one nested inside the other). If a future validator adds a
  no-overlap constraint for `defined_terms`, this pair would need to be reconciled.
- Did not annotate "its security policy as published at https://cloudvault.io/security" (clause
  4.4) as an external_document reference — it reads as generic descriptive prose rather than a
  named document title like "Order Form" or "Documentation". Flagging in case the harness wants
  URL-style references captured too.

Verification: `.venv/Scripts/python.exe -m pytest tests/test_gold_annotations.py -q --no-cov -k
uk_terms` — 13 passed, 0 failed (including `test_no_seed_markers_remain`).

---

## us_msa

Correction pass performed by reading `src/legal_rag_eval/fixtures/documents/us_msa.txt`
(sanitised via `legal_rag_eval.gold.sanitize`) against the seeded `gold/us_msa.json`, clause by
clause, using a throwaway inspection script (printed every clause span's boundary context,
every defined-term span, and a regex sweep for candidate cross-references), and by reading
`scripts/build_gold.py`'s `seed_us_article_section` / `seed_defined_terms` /
`seed_cross_references` to understand exactly what patterns the regex seeder does and does
not catch, so that "missing" candidates could be told apart from patterns the seeder
deliberately (and correctly) does not split.

- **Roman-numeral conversion (Article I-IX)**: 9 checked, no corrections needed. Every
  `ARTICLE <roman>` heading converts to the correct `article_N` (I->1 ... IX->9) and each
  is the 9th consecutive article with no gaps or duplicates.
- **Section-to-article mapping**: 46 Section-level entries checked (1.01-1.02, 2.01-2.05,
  3.01-3.05, 4.01-4.03, 5.01-5.05, 6.01-6.02, 7.01-7.03, 8.01-8.04, 9.01-9.08), no corrections
  needed — every section's `parent` matches the article it physically falls under, verified
  at every article boundary.
- **Clause hierarchy bug fixed — `section_6.01(b)`**: seed had `level: 3`,
  `parent: "section_6.01"` (the seeder's `SECTION_NN_LETTER_RE` treats any
  `Section N.NN(x)` heading as an automatic level-3 child of `section_N.NN`). Reading the
  text shows this is wrong: `"Section 6.01(b)  Cap on Liability. SUBJECT TO SECTION 6.02..."`
  is typeset exactly like every other top-of-paragraph `"Section N.NN  Heading."` block (own
  paragraph after a blank line, full ALL-CAPS operative text, no `(a)` sibling anywhere inside
  `Section 6.01`'s own text for a `(b)` to attach to). It is a mislabeled *Section*
  (the drafter evidently meant to write "Section 6.02" and bumped the real "Section 6.02"
  down), not a lettered sub-paragraph of Section 6.01. Corrected to `level: 2`,
  `parent: "article_6"`, keeping the identifier `section_6.01(b)` as printed in the document
  (this is a case where "record the document's own numbering" and "record its structural
  role" pull in different directions — the label is a typo but the structure is a plain
  Section, and the structure is what `level`/`parent` encode). Its own span
  `[14300, 14836)` was already correct and unaffected by this change.
- **Clause span bug fixed — `section_9.08`**: seed had `char_end: 25346` (== `n_chars`,
  i.e. the very end of the sanitised text). Because `section_9.08` (Counterparts) is the
  last Section in the last Article, the seeder's "end = start of next clause" rule had
  nothing to stop at and let the span run through the signature block
  (`"[SIGNATURE PAGE FOLLOWS]" ... "Date: ___"`) and the *entire* Exhibit A (including its
  own numbered items, dollar amounts, and milestone dates) as if all of that were part of
  Section 9.08's own text. Corrected `char_end` to `22957`, right after
  `"...Electronic signatures shall be deemed valid and binding."` — the true end of Section
  9.08's own content — verified with `text[22949:22957] == "binding."` and
  `text[22957:22960] == "\n\n["` (start of `[SIGNATURE PAGE FOLLOWS]`). This is the single
  highest-impact fix in this pass: left uncorrected, every metric that looks at "the text of
  Section 9.08" would have been scored against ~2,650 characters of unrelated signature-block
  and Exhibit-A content.
- **Missing/invented top-level or Section-level clauses**: none found. All 9 Articles and 46
  Sections present in the document are annotated once each, in order, with no invented
  entries from dates, dollar amounts, or the parties/RECITALS front matter (confirmed the
  parties block and RECITALS remain unannotated, as required).
- **Lettered `(a)`/`(b)`/... paragraphs — deliberately NOT split further**: the seed's 10
  `section_1.01(a)`-`(j)` entries (the definitions) were kept as-is (spans and headings
  verified correct). I looked hard for additional lettered paragraphs per the general
  guidance that this document type typically has more than the seeder finds, and confirmed
  there is a real, principled reason the seeder found only these 10 (plus the `6.01(b)`
  anomaly handled above): `Section 1.01`'s items are typeset as **block-style** paragraphs —
  each `(a)`/`(b)`/... starts its own physical line (5-space indent), separated by blank
  lines, exactly like a top-level Section. Every other lettered enumeration in this document
  (`Section 1.02(a)-(d)`, `2.04(a)-(d)`, `4.01(a)-(c)`, `4.02(a)-(b)`, `6.02(a)-(e)`,
  `7.01(a)-(b)`, `7.02(a)-(d)`, `7.03(a)-(c)`, `8.03(a)-(b)`, `8.04(a)-(d)`, `9.04(a)-(d)`,
  `9.05(a)-(b)` — 39 letters in total) is **run-in**: the letters appear mid-sentence
  (e.g. `"Customer shall: (a) provide Provider with timely access...; (b) designate..."`),
  never at the start of a physical line. I verified this by running the seeder's own
  line-start letter-scanner (`LETTER_LINE_RE` / `scan_letter_items`, which requires `^\(`)
  across the *entire* sanitised text (not just the zone the seeder restricts itself to) and
  it finds exactly the same 10 items in Section 1.01, plus the 3 block-style items in
  Exhibit A's item 1 (out of scope — see below), and nothing else. I also confirmed no
  cross-reference anywhere in the document ever cites a specific letter (e.g. no
  `"Section 7.02(a)"`) — every internal reference in this Agreement names a whole Article or
  Section, never a sub-letter — which is further evidence these run-in letters function as
  prose enumeration within one sentence, not as independently addressable document units.
  Splitting them would also produce nonsensical micro-"clauses" for the shortest ones (e.g.
  `9.04(a)` would be the 16-character fragment `"(a) personally;"`). I left all 39 out;
  this is the one guidance item I am flagging as a judgment call rather than a certainty
  (see "Uncertain" below).
- **Exhibit A excluded from clause annotation (unchanged from seed, decision documented)**:
  the seeder's own `EXHIBIT_MARKER_RE` stops all structural scanning at `"EXHIBIT A"`, so no
  clause of any kind is created for it. I confirmed this is the right call for this
  `numbering_style` (`us_article_section` has no schedule/exhibit handling at all, unlike
  `uk_decimal`, which explicitly creates `schedule_N` clauses) and is consistent with how the
  Agreement's own definitions treat Exhibit A/an SOW/an Order Form as *external* instruments
  incorporated by reference, not part of the Agreement's own Article/Section numbering.
  Exhibit A (including its item `1.`/`2.`/`3.` numbering, its own `(a)-(c)` block-style list,
  and its `(i)-(iv)` payment schedule) remains entirely unannotated, the same as the parties
  block and RECITALS are unannotated front matter. A cross-reference from inside Exhibit A
  back into the Agreement body (`"Section 3.01"` at 25266) is still correctly kept, since
  cross-references don't need to originate inside an annotated clause.
- **Defined terms — expanded from 11 to 31**. The seeder's `DEFINITION_RE` only matches the
  forward pattern `"X" means ...`, so it found exactly the 11 terms defined that way in
  Section 1.01 (`Affiliate`, `control`, `Authorized User`, `Confidential Information`,
  `Documentation`, `Intellectual Property Rights`, `Order Form`, `Provider Technology`,
  `Services`, `SOW`, `Term`) — all verified correct and left unchanged. It completely misses
  the reverse "quoted-and-defined" pattern used everywhere else in this document,
  `<description> (the "X")` / `<description> ("X")`, which SCHEMA.md explicitly allows
  ("quoted-and-defined, or listed in a definitions clause"). Added 20 terms of this kind,
  each verified to contain its term and its defining clause:
  - **`Statement of Work`** (missed entirely — only the paired synonym `SOW` was seeded from
    the same sentence `"Statement of Work" or "SOW" means an Order Form..."`). Given the
    same span as `SOW`'s definition since both names are defined by the identical `means`
    clause (`[4822, 5052]`, overlapping `SOW`'s `[4845, 5052]` — SCHEMA does not require
    defined-term spans to be mutually non-overlapping, only clause spans).
  - **`Disclosing Party`** and **`Receiving Party`**, nested inside the `Confidential
    Information` definition (`"...disclosed by one Party (the "Disclosing Party") to the
    other Party (the "Receiving Party")..."`) — the same pattern already used for `control`
    nested inside `Affiliate`, just not carried through to this clause by the seeder's
    forward-only regex.
  - **`Fees`** (Section 3.01), **`Payment Due Date`** (3.02), **`Taxes`** (3.04),
    **`Customer Data`** (5.03), **`Feedback`** (5.05), **`Customer Indemnitees`** and
    **`Losses`** (7.01), **`Provider Indemnitees`** (7.02), **`Indemnified Party`** and
    **`Indemnifying Party`** (7.03), **`Force Majeure Event`** (9.07) — each a
    substantive, actively cross-referenced defined term (e.g. `Fees` recurs in Sections
    3.02-3.05, 6.01(b), 8.04; `Losses` recurs in 7.02) established via the same
    `(the "X")`/`("X")` convention.
  - **`Agreement`**, **`Effective Date`**, **`Provider`**, **`Customer`**, **`Party`**,
    **`Parties`** — the task's own example, `(the "Effective Date")` in the preamble, led me
    to check the whole parties/preamble block, which turns out to establish 6 defined terms
    the same way (`(this "Agreement")`, `("Provider")`, `("Customer")`, `a "Party"`, `the
    "Parties."`). These are front matter (not annotated as a clause), but SCHEMA does not
    require a defined term to sit inside a clause span, and all six are used pervasively
    throughout the operative text, so I included them (flagged under "Uncertain" below).
  - **Explicitly excluded**: `"include," "includes," and "including"` in Section 1.02 — this
    is quoted, but it is an interpretive convention about word usage
    (`"...shall be deemed to be followed by the words 'without limitation'"`), not the
    creation of a capitalized term of art, so it does not fit "the document itself marks as
    defined" in the substantive sense the other 31 terms do.
  - **Explicitly excluded**: `"SOW Effective Date"` (Exhibit A) — consistent with treating
    Exhibit A as out of scope for all annotation categories, not just clauses.
- **Cross-references — expanded from 35 to 43, `_seed_unresolved` markers**: none were
  present in the seed; nothing to remove. Verified `text[char_start:char_start+len(raw_text)]
  == raw_text` for all 43 entries programmatically — all pass. All 8 additions:
  - **`SECTION 6.02`** at 14346, inside `Section 6.01(b)`'s ALL-CAPS text
    (`"SUBJECT TO SECTION 6.02, EACH PARTY'S TOTAL CUMULATIVE LIABILITY..."`). This is a
    genuine seeder bug, not a scope judgment call: `SECTION_NN_LIST_RE` in
    `build_gold.py` matches `\bSections?\b` case-sensitively, so it cannot match the
    ALL-CAPS `"SECTION"` used throughout this liability-cap paragraph. `kind:
    internal_clause`, `target_identifier: "section_6.02"`.
  - **`Exhibit A`** at 4588, inside the `Services` definition (Section 1.01(h)):
    `"...as described in each Order Form or Exhibit A attached hereto..."`. The seeder has no
    pattern for `Exhibit` references at all. Since Exhibit A is not itself annotated as a
    clause in this file (see above), `kind: external_document`, `target_identifier: null`.
  - **`a Statement of Work`** at 12885 (Section 5.04) and 5 bare-acronym siblings — **`a
    SOW`** at 6336, 6574, 6676, **`the SOW`** at 6411, and **`a\nSOW`** at 12990 — added by
    direct analogy to the seeder's own `EXTERNAL_PHRASES` pattern for Order Form
    (`\b(?:the|an|any)\s+(?:applicable\s+)?Order\s+Form`), which the seeder already uses to
    flag 8 "Order Form" mentions as `external_document`. `Statement of Work`/`SOW` is the
    exact same kind of reference to an external, unnumbered instrument, so I applied the
    same "the/an/any + singular" filter (excluding the one generic `"Each SOW..."` mention
    at 6233, by analogy with the seeder's own exclusion of the generic `"Each Order Form..."`
    mention inside Section 1.01(f)'s own definition). This is a consistency extension rather
    than a hard rule from SCHEMA.md, so I am flagging it as a judgment call.
  - Verified there are no missed `Article <roman>` or `Section N.NN` body references: reran
    the seeder's exact `ARTICLE_ROMAN_REF_RE` / `SECTION_NN_LIST_RE` patterns plus a
    case-insensitive sweep across the whole document and cross-checked every hit against the
    existing entries; the only true gap was the ALL-CAPS `SECTION 6.02` above.
  - No `external_statute` references exist in this document (it never cites a specific
    statute or regulation by name/number, only "applicable law" generically) — 0 is correct,
    not an omission.
- **Housekeeping**: `text_sha256` and `n_chars` were already correct against the sanitised
  fixture text (fixture untouched) — left unchanged. JSON re-serialized with 2-space indent,
  LF newlines, trailing newline, matching the original file's formatting exactly.

### Final counts
- Clauses: 57 total — 9 level 1 (Articles I-IX), 38 level 2 (37 original Sections +
  `section_6.01(b)` reclassified from level 3), 10 level 3 (`section_1.01(a)`-`(j)`).
- Defined terms: 31 (11 original, unchanged, + 20 added).
- Cross-references: 43 — 28 `internal_clause`, 15 `external_document`, 0 `external_statute`,
  0 `internal_schedule`, 0 `internal_definition` (this document never cross-references a
  definition by name from elsewhere in a way that would warrant `internal_definition`; all
  defined-term usages are plain prose references to the term, not numbered citations).

### Uncertain / flagged for awareness
- **Run-in lettered paragraphs (39 instances) left unsplit** — see above. I'm confident in
  the block-style-vs-run-in distinction as the correct read of *this* document, but if the
  harness's intent for `us_article_section` documents is closer to "every lettered
  enumeration is a clause regardless of typesetting," this is the biggest lever to revisit.
- **`Agreement`, `Party`, `Parties` as defined terms** — these are correctly
  quoted-and-defined by the letter of SCHEMA.md, but they are also the three most generic,
  purely self-referential/structural terms in the document (versus e.g. `Customer Data` or
  `Force Majeure Event`, which are clearly substantive). I included them for mechanical
  consistency but would not be surprised if the intended gold standard omits them as
  "too generic to be worth tracking."
- **`a SOW` / `the SOW` bare-acronym cross-references (5 of the 8 additions)** — a reasoned
  extension of the seeder's own Order-Form pattern rather than something SCHEMA.md states
  directly; flagged in case a narrower reading (only the one explicit `"a Statement of
  Work"` mention) was intended.
- **`section_6.01(b)`'s identifier** was left as printed (`section_6.01(b)`) even after
  promoting it to level 2, rather than renaming it to something that reads as level-2 (e.g.
  `section_6.01b`), since the document's own printed label is literally `"Section 6.01(b)"`
  and SCHEMA.md's `parent`/`level` fields — not the identifier's lexical shape — are what
  encode the hierarchy. Worth a second look if downstream tooling assumes identifiers
  matching `section_N.NN(x)` are always level 3.

Verification: `.venv/Scripts/python.exe -m pytest tests/test_gold_annotations.py -q --no-cov -k us_msa`
— 13 passed, 0 failed (including the no-seed-markers and raw-text-match checks). Full suite
(`tests/test_gold_annotations.py`, no `-k` filter) also passes: 66 passed, 0 failed.

---

## us_terms_of_service

Correction pass against `src/legal_rag_eval/fixtures/documents/us_terms_of_service.txt`
(sanitised, 25743 chars, sha256
`ef6b36ea2bf8d647a946b141ca6c3d690aad566e0dc2067c9951e03503c1df62`, unchanged by this
pass). Method: `legal_rag_eval.gold.sanitize()` + a throwaway inspection script that printed
every clause/term/reference with surrounding context, read line by line against the
source text, plus targeted `text.find()` probes to pin exact offsets before writing them.

### Clause spans
- **Front matter correctly excluded already**: offsets 0–762 (title block, "Last
  Updated"/"Effective Date" lines, the ALL-CAPS notice paragraph, and the "These Terms are
  entered into between..." parties recital) carry no clause — correct, no change needed.
  `section_1` begins exactly at the first character of `Section 1.` with no leading
  blank-line drift.
- **Added 36 missing level-3 lettered clauses.** The seed's `scan_letter_items()` regex
  (`^([ \t]*)\(([a-z])\)[ \t]+`, anchored to the *start of a physical line*) only matches
  a lettered list formatted as its own indented paragraph per letter — the style used
  exactly once in this document, in Section 2.2 (`(a)` through `(g)`, already seeded
  correctly). Every other lettered enumeration in the document is written inline inside a
  running sentence (`"Customer shall: (a) ...; (b) ..."`), so the line-start regex never
  matched them and the seeder silently produced zero clauses for ten separate lettered
  lists. Checked with a `\([a-zA-Z]{1,4}\)` sweep of the whole document and confirmed all
  of the following are genuine enumerated sub-items (own concept, `;` or `or`-separated,
  parallel structure) rather than stray parentheticals, so each now has its own
  `level: 3` clause with `char_start` at the `(` and `char_end` at the next item (or the
  next Section marker for the last item in a list, per SCHEMA.md's "up to the next clause
  at any level" rule):
  - `section_2.3(a)`–`(d)` (Account Security: password/MFA/notify/no-sharing duties)
  - `section_3.1(a)`–`(e)` (Prohibited Content categories)
  - `section_5.3(a)`–`(b)` (Late Payments: suspend / charge interest)
  - `section_7.3(a)`–`(c)` (Termination for Cause grounds)
  - `section_7.5(a)`–`(d)` (Effect of Termination items, including the surviving-sections
    list folded into `(d)`)
  - `section_9.2(a)`–`(b)` (Cap on Liability alternatives — document uses `(A)`/`(B)` in
    ALL CAPS; identifiers normalised to lowercase `(a)`/`(b)` for consistency with every
    other lettered identifier in the file, per SCHEMA.md's own `"1.1(a)"` example; the
    `heading` field keeps the document's original ALL-CAPS text, matching how `section_8.1`
    etc. already do)
  - `section_9.3(a)`–`(e)` (Exceptions to the liability cap)
  - `section_10.1(a)`–`(c)` (Strata's indemnification-cure options)
  - `section_10.2(a)`–`(e)` (grounds for Customer's indemnification of Strata)
  - `section_10.3(a)`–`(c)` (Indemnification Procedure duties)

  Each affected level-2 parent's own span was shrunk to end at the first lettered item's
  `char_start` (it previously ran through the whole list, which violates "a clause span
  excludes its descendants"), e.g. `section_2.3` end moved from 5919 to 5433.
- **Not split further: the roman-numeral exceptions inside `section_10.1`.** After its
  `(a)`/`(b)`/(c)` list, the same section adds a second, sibling enumeration — "Strata
  shall have no obligation ... if the alleged infringement arises from: (i) Customer
  Content; (ii) ...; (iii) ...; or (iv) ...". These are romanettes, and SCHEMA.md is
  explicit that "deeper lettered romanettes are folded into their parent's span" at the
  level-3 cap, so `section_10.1(c)`'s span was left running through to the next Section
  marker (20305) rather than creating a fourth nesting level; the `(i)`–`(iv)` text is
  content of `section_10.1(c)`.
- **Trimmed the trailing document footer out of `section_11.10`.** The seed's last clause
  ran all the way to `n_chars` (25743), swallowing `"[END OF TERMS OF SERVICE]"` and the
  address/contact footer block. That block is document furniture, not clause content —
  the same category SCHEMA.md excludes at the front of the document — so
  `section_11.10.char_end` was moved back to 25582, immediately after "...original ink
  signatures." (the end of its last substantive sentence), leaving the closing marker and
  footer as an unannotated trailing gap (legal per SCHEMA.md: "gaps between spans are
  legal").
- Verified section_1.9/1.10/1.11/1.12 and section_11.1/.../11.10 are ordered by
  `char_start`, not lexically — `section_1.10` correctly sorts after `section_1.9` (not
  after `section_1.1`), and `section_11.10` correctly sorts last (not immediately after
  `section_11.1`). No misordering found.
- Final count: **114 clauses** — 11 level-1, 60 level-2, 43 level-3 (was 78: 11/60/7).

### Defined terms
- **Added 7 quoted-and-defined terms the seed missed.** The seed's `DEFINITION_RE` only
  matches `"X" means|shall mean|refers to`, the pattern used by the twelve terms in
  Section 1. This document also defines terms the standard drafting way, via apposition —
  `<description> ("X")` — inside its operative sections, and the seed's regex never looked
  for that pattern at all. Each of these is reused later in the document (not a one-off
  parenthetical), which is exactly SCHEMA.md's "quoted-and-defined" test:
  - **SOW** `[6238, 6365]` — defined in 2.5 ("a mutually executed Statement of Work
    ("SOW")..."), reused in the same section ("Each SOW is subject...", "the SOW expressly
    identifies...") and again in 11.2 ("any executed DPA or SOW").
  - **DPA** `[9104, 9387]` — defined in 4.3 ("a Data Processing Agreement ("DPA")..."),
    reused twice more in 4.3 ("between the DPA and these Terms", "the DPA shall control")
    and again in 11.2.
  - **Feedback** `[12709, 12864]` — defined and reused within 6.2.
  - **Usage Data** `[13149, 13347]` — defined and reused within 6.3.
  - **Customer Indemnitees** `[19086, 19220]` — defined in 10.1's indemnification grant.
    Note: the document's own line-wrap breaks the phrase as `"Customer\nIndemnitees"`; the
    stored `term` value keeps that literal embedded newline (matching the existing
    seed convention for wrapped `raw_text` in cross-references, e.g. `"the Order\nForm"`)
    because `test_defined_term_spans_contain_their_term` requires an exact
    case-insensitive substring match against the sanitised text, which is not
    whitespace-normalised.
  - **Claims** `[19295, 19400]` — defined in 10.1, reused in 10.2 ("any Claims arising
    from") and 10.3 ("any Claim").
  - **Strata Indemnitees** `[20347, 20479]` — defined in 10.2's indemnification grant.
- **Decision: party shorthands not added.** The front-matter recital quoted-and-defines
  four pronoun aliases for each party — `("Strata," "we," "us," or "our")` and
  `("Customer," "you," or "your")`. These technically satisfy "quoted-and-defined," but I
  left them out of `defined_terms`: they are pronoun/party-identity shorthands declared in
  the same un-numbered front-matter recital that SCHEMA.md already excludes from the
  clause partition, not substantive legal/technical concepts like the twelve Section 1
  definitions or the seven added above — a retrieval system being scored against this
  fixture is far more likely to be asked "what is Customer Content" than "who is 'our'".
  Flagging this as a judgment call a reviewer could reasonably overturn.
- Final count: **19 defined terms** (was 12; +7).

### Cross-references
- **No `_seed_unresolved` markers were present** in this file to begin with (confirmed by
  grep) — nothing to resolve/delete in that category.
- **Verified every existing entry programmatically**: `text[char_start:char_start +
  len(raw_text)] == raw_text` held for all 41 seeded entries, and all `internal_*` targets
  already resolved to a real identifier in this file. No corrections needed to the
  pre-existing 41.
- **Added 14 references the seed missed:**
  - 3 `external_statute`: `"the California Consumer Privacy Act"` (2364) and `"the
    California Privacy\nRights Act"` (2415), both named inline in the `Personal
    Information` definition (1.7) alongside the already-defined `"CCPA/CPRA"` shorthand;
    `"the Export Administration Regulations"` (7906) in 3.3's export-compliance clause.
    None were caught because the seed's `EXTERNAL_PHRASES` list only contains
    `Order Form`, the GDPR regulation, and the UK Data Protection Act — it has no US
    statute patterns at all for this fixture.
  - 11 `external_document`: further mentions of the other-instrument terms named in
    SCHEMA.md's own examples that the seed's `EXTERNAL_PHRASES` regex (Order-Form-only)
    never looked for — `"the Documentation"` (3931, in 2.1; 4206, in 2.2(a)), `"the
    Privacy Policy"` (8659, in 4.1; 21928, in 11.2, as `"the Privacy\nPolicy"`), `"the
    Acceptable Use Policy"` (21948, in 11.2), `"the DPA"` (9425 and 9505, both in 4.3),
    `"DPA"` and `"SOW"` (21992 and 21999, in 11.2's "any executed DPA or SOW"), and `"SOW"`
    / `"the SOW"` (6371 and 6463, in 2.5).
  - Each new entry checked programmatically against `text[char_start:char_start+len
    (raw_text)] == raw_text` before being added.
  - **Not tagged**: bare uses of "Documentation" without an article that name it alongside
    "Platform" in a list — e.g. 2.2(e) "...marks on the\nPlatform or Documentation;" and
    6.1 "...the Platform, Documentation, and all improvements..." — where the visible
    article "the" grammatically attaches only to "Platform". Treated these as ordinary
    reuse of the defined term rather than a referential pointer, by analogy with how
    "Platform", "Customer Content", etc. are never cross-referenced on reuse. This mirrors
    exactly why the seed itself never tags a bare "Order Form" without a preceding
    article — only the "the/an/any Order Form" article+noun pattern is treated as a
    pointer to the external instrument.
  - **Not tagged**: the defining/introducing occurrence of each newly-added external-
    document term (e.g. `("SOW")`, `("DPA")`, `"Strata's Privacy Policy at
    strataplatform.io/privacy"`, `"Strata's Acceptable Use Policy available at
    strataplatform.io/aup"`) — these establish what the term names rather than pointing
    back to an already-introduced instrument, the same distinction that already explains
    why the seed never tags "Order Form" inside its own Section 1.6 definition sentence.
- Final count: **55 cross-references** (was 41; +14) — 28 `internal_clause`,
  24 `external_document`, 3 `external_statute`.

### Final counts
- Clauses: 114 total — 11 level 1, 60 level 2, 43 level 3 (was 78: 11/60/7; +36).
- Defined terms: 19 (was 12; +7).
- Cross-references: 55 (was 41; +14) — 28 internal_clause, 24 external_document,
  3 external_statute.

### Genuinely unsure about
- Whether the four party-pronoun shorthands (`Strata`/`we`/`us`/`our`,
  `Customer`/`you`/`your`) should be `defined_terms`. Left out as front-matter identity
  labels rather than substantive concepts (see above), but SCHEMA.md's literal
  "quoted-and-defined" test does not itself exclude them.
- Whether bare "Documentation" (no article) alongside "Platform" in a list should also be
  tagged `external_document` — I judged it as ordinary defined-term reuse, not a pointer,
  but the line between "the Documentation" (tagged) and "Documentation" (not tagged) is a
  drafting-style artifact rather than a difference in what is being referred to, and a
  reviewer could reasonably want both tagged consistently.
- The `"Customer\nIndemnitees"` defined-term value keeps a literal embedded newline to
  satisfy the exact-substring test — functionally correct but visually unusual next to
  every other `term` value in the file; flagging in case a normalised alternative (e.g.
  changing the test, or excluding this term) is preferred instead.

### Verification
`pytest tests/test_gold_annotations.py -q --no-cov -k us_terms` — 13 passed, including
`test_no_seed_markers_remain`, `test_defined_term_spans_contain_their_term`, and
`test_internal_cross_reference_targets_resolve`.
Full `tests/test_gold_annotations.py` (all 5 fixtures) — 66 passed.

---

