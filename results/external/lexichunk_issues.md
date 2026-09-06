# LexiChunk issues found on real CUAD contracts

Findings from running LexiChunk `0.8.0b1` over 150 real commercial contracts
sampled from CUAD (`theatticusproject/cuad-qa`, seed 0). These are SEC exhibit
filings — real OCR-era text with page furniture, exhibit headers, ragged
indentation and mixed numbering conventions.

Ranked by impact on someone using LexiChunk for retrieval. Each entry has a
self-contained reproducer that runs against `lexichunk` alone (no dataset
download); the probe scripts that surfaced them are under `.cache/recon/`.

**LexiChunk was not modified.** Every count below is from an unmodified
install.

Headline: **no exceptions were raised on any of the 150 contracts.** LexiChunk
does not crash on real filings. Every issue below is wrong or surprising
*output*, not a stack trace.

| # | Issue | Severity | Frequency (of 150 contracts) |
|---|-------|----------|------------------------------|
| 1 | `us` jurisdiction profile misses structure that `uk` finds, on US filings | High | 34 (22.7%) |
| 2 | `chunk.content` is not `text[char_start:char_end]` | High | 65 (43.3%) |
| 3 | `max_chunk_size` is exceeded, by up to 2.4x | Medium | 3 (2.0%) |
| 4 | Worst-case latency ~500x the median (20.6 s on one contract) | Medium | tail of the distribution |
| 5 | Single characters are dropped between consecutive chunks | Low | 21 (14.0%) |
| 6 | Every chunk classified `unknown` | Low | 2 (1.3%) |

---

## 1. The `us` jurisdiction profile misses structure that `uk` finds — on US filings

**Severity: High.** This is the single biggest result in the CUAD evaluation.
On 100 sampled contracts, `jurisdiction="us"` recovered five or more top-level
clauses in only a small minority of contracts; switching the same contracts to
`jurisdiction="uk"` recovers far more. Since CUAD is entirely US SEC filings,
the profile named for the jurisdiction is the *worse* choice for it.

The cause is the heading pattern. The `us` profile appears to require an
explicit `Section N` or `ARTICLE N` marker. The bare decimal style that
dominates US commercial drafting — `1. Definitions.` / `1.1 ...` — is
recognised by the `uk` profile and missed by `us`.

In the 150-contract probe, 34 contracts had `uk` finding >= 5 top-level nodes
while `us` found fewer than 3. Examples: `top_us=1, top_uk=10` on a 29k-char
co-branding agreement; `top_us=1, top_uk=10` on a 19k-char hosting agreement.

At corpus scale, on the 100-contract evaluation sample: `us` recovers >= 5
top-level clauses in **14.0%** of contracts (mean 2.8 top-level nodes); `uk`
manages **31.0%** (mean 4.8) on the *same US filings*.

Anyone who follows the obvious advice ("it's a US contract, pass `us`") gets a
chunker that has silently degraded to fixed-size splitting.

```python
from lexichunk import LegalChunker

text = """DISTRIBUTION AGREEMENT

1. Definitions.
1.1 "Affiliate" means any entity under common control with a party.

2. Appointment.
2.1 Supplier appoints Distributor as its exclusive distributor.

3. Confidentiality.
3.1 Each party shall keep confidential all Confidential Information.

4. Indemnification.
4.1 Distributor shall indemnify and hold harmless Supplier.

5. Governing Law.
5.1 This Agreement is governed by the laws of the State of Delaware.
"""

for jurisdiction in ("us", "uk"):
    chunker = LegalChunker(jurisdiction=jurisdiction, max_chunk_size=512)
    nodes = chunker.parse_structure(text)
    top = [n for n in nodes if n.level == 0]
    print(jurisdiction, "top-level nodes:", len(top), "chunks:", len(chunker.chunk(text)))

# us top-level nodes: 0 chunks: 1      <- the whole contract collapses to one chunk
# uk top-level nodes: 5 chunks: 6      <- correct
```

Prefixing every heading with the literal word `Section` makes the `us` profile
work, which confirms the pattern is the trigger.

---

## 2. `chunk.content` is not `text[chunk.char_start:chunk.char_end]`

**Severity: High.** For 43% of contracts (and 51% of individual chunks
measured over 60 contracts: 1,244 of 2,422), `chunk.content` is the slice with
the **parent heading prepended**, while `char_start` is *not* moved back to
cover that heading.

Anything that trusts the offsets — highlighting a retrieved passage in the
source document, mapping an answer span back to a page, deduplicating against
the original — silently mis-renders. Anything that trusts `content` instead
gets text that does not exist contiguously in the source.

`include_context_header=False` does **not** turn this off. That flag controls
the separate `chunk.context_header` field (`[Section: ...] [Type: ...]`); the
heading prefix on `content` is added regardless.

```python
from lexichunk import LegalChunker

filler = " ".join(["The parties acknowledge and agree that this applies."] * 6)
text = (
    "Exhibit 99.1\n\n"
    "COOPERATION AGREEMENT\n\n"
    "This Agreement is made as of January 1, 2020. " + filler + "\n\n"
    "(a) The Board shall appoint the designee promptly. " + filler + "\n\n"
    "(b) The designee shall comply with all policies. " + filler + "\n"
)

chunker = LegalChunker(jurisdiction="us", max_chunk_size=512,
                       include_context_header=False)
for chunk in chunker.chunk(text):
    slice_ = text[chunk.char_start:chunk.char_end]
    if slice_ != chunk.content:
        extra = chunk.content[: len(chunk.content) - len(slice_)]
        print(f"{chunk.hierarchy_path!r}: content has extra prefix {extra!r}")

# 'COOPERATION AGREEMENT > (a)': content has extra prefix 'COOPERATION AGREEMENT\n'
# 'COOPERATION AGREEMENT > (b)': content has extra prefix 'COOPERATION AGREEMENT\n'
```

If the prefix is deliberate context enrichment, the fix is documentation plus a
separate field (or extending `char_start` to include the heading). Either way
the current contract is a trap, and it is not stated anywhere in the API docs.

*(This does not invalidate the CUAD containment numbers: the offsets themselves
are monotonic, in-bounds and index the correct region of the source. Only
`content` carries the extra text. Containment is computed from offsets.)*

---

## 3. `max_chunk_size` is exceeded, by up to 2.4x

**Severity: Medium.** `max_chunk_size` is documented as a cap. On three
contracts it was breached — worst case a chunk of **1,220 tokens against a
configured limit of 512** (4,881 characters), and seven chunks over the limit
in that one contract.

The mechanism is that the splitter has no fallback below the whitespace level:
when a stretch of text offers no split point inside the budget, the whole
stretch is emitted as one oversized chunk. The synthetic case below isolates
that cleanly (a run with no whitespace at all). The real-world instance is
milder in form but the same in kind — the offending chunk was 4,881 characters
containing only 14 newlines, with a single line of 2,485 characters.

This matters because callers size `max_chunk_size` to an embedding model's
context window. A 2.4x overrun is silent truncation at embedding time.

```python
from lexichunk import LegalChunker

# A run with no whitespace to split on - e.g. an OCR'd table, a base64 blob,
# or a long identifier string, all of which occur in SEC exhibit filings.
text = "Section 1. Definitions.\n\n" + ("A" * 6000) + "\n"

chunker = LegalChunker(jurisdiction="us", max_chunk_size=512)
print("configured max:", 512,
      "worst observed:", max(c.token_count for c in chunker.chunk(text)))
# configured max: 512 worst observed: 1500
```

Real example: `EtonPharmaceuticalsInc_20191114_10-Q_EX-10.1_..._Development
Agreement` — 7 of its 39 chunks exceed the cap, worst at 1,220 tokens. Note
that ordinary long prose does *not* trigger this: a 9k-character paragraph with
spaces but no sentence punctuation caps correctly at 512.

---

## 4. Worst-case latency is ~500x the median, and is content-dependent

**Severity: Medium.** Median chunking cost across the 150 contracts was
**0.042 s**, but the mean was **0.70 s** and the worst case **20.6 s** for one
291,873-character contract. Two other contracts took 8.1 s (168k chars) and
7.8 s (290k chars).

It is *not* simply document length. Synthetic text scales linearly and cheaply
— 722,000 characters of repeated well-formed clauses chunk in 0.48 s, more than
twice the size of the worst real contract at 2% of the cost:

```python
import time
from lexichunk import LegalChunker

chunker = LegalChunker(jurisdiction="us", max_chunk_size=512)
body = "Section 1. Terms.\n\n" + ("The parties agree to the foregoing. " * 200)
for multiplier in (1, 10, 50, 100):
    text = body * multiplier
    start = time.perf_counter()
    chunker.chunk(text)
    print(f"{len(text):>9,} chars -> {time.perf_counter() - start:6.2f}s")

#     7,219 chars ->   0.01s
#    72,190 chars ->   0.05s
#   360,950 chars ->   0.22s
#   721,900 chars ->   0.48s
```

So something in the *content* of certain real filings — most likely the
definition extraction or cross-reference resolution pass, which the synthetic
text gives nothing to chew on — is the cost driver. Isolating it needs
profiling against the real document rather than a synthetic one; the reproducer
is: load CUAD contract `PhasebioPharmaceuticalsInc_20200330_10-K_EX-10.21_12086810_EX-10.21_Development Agreement`
and call `chunk()` on it.

For comparison, `RecursiveCharacterTextSplitter` processes the same corpus at
roughly 0.006 s per contract — two orders of magnitude faster. That trade is
defensible for a structure-aware chunker, but an unadvertised 20-second
worst case on a single document will time out an ingestion pipeline that sizes
its budget from the median.

---

## 5. Single characters are dropped between consecutive chunks

**Severity: Low.** Consecutive chunk spans are not contiguous: 21 contracts had
at least one gap. Every observed gap was exactly one character and every lost
character was whitespace, so no substantive text is lost — but the invariant
"the chunks tile the document" does not hold, and a gold span that straddles
such a boundary can fail a containment check for a cosmetic reason.

```python
from lexichunk import LegalChunker

body = " ".join(["The parties acknowledge and agree that the foregoing applies."] * 40)
text = "DISTRIBUTION AGREEMENT\n\n" + body + "\n"

chunker = LegalChunker(jurisdiction="us", max_chunk_size=64)
spans = [(c.char_start, c.char_end) for c in chunker.chunk(text)]
gaps = [(a_end, b_start) for (_, a_end), (b_start, _) in zip(spans, spans[1:])
        if b_start != a_end]
print("gaps:", gaps[:5])
# gaps: [(257, 258), (491, 492), (725, 726), (959, 960), (1193, 1194)]
```

---

## 6. Every chunk classified `unknown`

**Severity: Low.** Two contracts produced chunks where *every* chunk came back
`ClauseType.UNKNOWN`. Both are short and unusually formatted (one is a
Chinese-English photovoltaic cooperation agreement, one a bank outsourcing
agreement that reduced to a single chunk). This is the classifier declining
rather than misfiring, which is the right failure direction, but it means those
documents carry no clause metadata at all.

Affected: `SPIENERGYCO,LTD_07_10_2014-EX-10-Cooperation Agreement of 50MWp
Photovoltaic...` (2 chunks) and `UNITEDNATIONALBANCORP_03_03_1999-EX-99-Outsourcing
Agreement with the BISYS Group, Inc.` (1 chunk).

---

## Related: classifier signals that are too broad to be useful

Not a crash, but the LEDGAR evaluation traced a large share of the classifier's
errors to two over-general keyword signals. Recorded here because the fix is in
`CLAUSE_SIGNALS`, not in the evaluation:

* `ClauseType.COVENANTS` includes the signal **`"shall not"`**, which appears in
  a large fraction of *every* clause type in commercial drafting. Measured
  precision for `covenants` on LEDGAR was in the single digits.
* `ClauseType.GOVERNING_LAW` includes **`"jurisdiction"`** and
  **`"applicable law"`**, so arbitration and forum-selection clauses are pulled
  into `governing_law`. `governing_law` over-fires: high recall, roughly
  40% precision, and it is the single most common wrong prediction.

See [`docs/external_evals.md`](../../docs/external_evals.md) for the full
per-class numbers.
