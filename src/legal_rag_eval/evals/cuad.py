"""CUAD evaluation: do chunk boundaries preserve gold answer spans?

CUAD (the Atticus Project's Contract Understanding Atticus Dataset) is 510 real
commercial contracts with expert-annotated answer spans for 41 categories. A
chunker that cuts through the middle of a gold span destroys that answer for
retrieval, so **span containment** — does at least one chunk wholly contain the
span? — is a direct, retrieval-agnostic measure of boundary quality.

Containment on its own is a trap: a splitter that emits the whole document as
one chunk scores 100%. Two corrections are therefore reported alongside it:

* the **mean chunk length** for each strategy, and
* the **length-matched grid expectation** — the containment a naive fixed-size
  splitter with the *same* mean chunk length would achieve on the same spans,
  computed analytically as ``mean(max(0, 1 - span_len / mean_chunk_len))`` for
  a uniformly-placed span against a boundary grid of that period. Subtracting
  it gives a **lift** that credits a strategy only for boundaries that beat
  cutting at a fixed stride.
"""

from __future__ import annotations

import time
import traceback
from collections import Counter
from contextlib import suppress
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

from legal_rag_eval.evals.chunkers import Span, lexichunk_spans, rcts_spans, sentence_window_spans
from legal_rag_eval.evals.common import (
    DEFAULT_BOOTSTRAP,
    Interval,
    bootstrap_ci,
    cache_dir,
    deterministic_sample,
    markdown_table,
    mean,
)

#: Dataset sources tried in order, as ``(dataset_id, revision)``.
#:
#: ``theatticusproject/cuad-qa`` on the Hub ships only a legacy loading script,
#: which ``datasets`` >= 3 refuses to execute, so the pinned first entry targets
#: the Hub's automatic Parquet conversion branch. Loading the repo *without* an
#: explicit revision can silently succeed from a stale local cache and then
#: misreport its provenance, so the revision is always passed explicitly.
CUAD_SOURCES: tuple[tuple[str, str | None], ...] = (
    ("theatticusproject/cuad-qa", "refs/convert/parquet"),
    ("cuad", "refs/convert/parquet"),
    ("theatticusproject/cuad-qa", None),
)

#: Both CUAD splits are used: nothing here is trained, and the split boundary
#: only shrinks the pool of real contracts available to sample from.
DEFAULT_SPLIT = "train+test"

#: Default minimum number of top-level structural nodes for a contract to count
#: as "parsed" rather than "fell back to flat text".
DEFAULT_MIN_CLAUSES = 5


@dataclass(frozen=True, slots=True)
class Contract:
    """One CUAD contract with its gold answer spans."""

    contract_id: str
    text: str
    spans: tuple[Span, ...]
    categories: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class StrategySpec:
    """A named chunking configuration to evaluate."""

    name: str
    build: Callable[[str], list[Span]]
    note: str = ""


@dataclass(slots=True)
class StrategyResult:
    """Containment and cost measurements for one chunking strategy."""

    name: str
    note: str
    containment: Interval
    per_contract_containment: Interval
    mean_chunk_chars: float
    median_chunk_chars: float
    mean_chunks_per_contract: float
    grid_expected_containment: float
    lift: float
    chunks_per_span: dict[str, int]
    mean_chunks_per_span: float
    seconds_per_contract: float
    spans_evaluated: int
    failures: int

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly mapping."""
        return {
            "name": self.name,
            "note": self.note,
            "containment": self.containment.as_dict(),
            "per_contract_containment": self.per_contract_containment.as_dict(),
            "mean_chunk_chars": self.mean_chunk_chars,
            "median_chunk_chars": self.median_chunk_chars,
            "mean_chunks_per_contract": self.mean_chunks_per_contract,
            "grid_expected_containment": self.grid_expected_containment,
            "lift_over_length_matched_grid": self.lift,
            "chunks_per_span_distribution": self.chunks_per_span,
            "mean_chunks_per_span": self.mean_chunks_per_span,
            "seconds_per_contract": self.seconds_per_contract,
            "spans_evaluated": self.spans_evaluated,
            "failures": self.failures,
        }


@dataclass(slots=True)
class ParseFailure:
    """A LexiChunk exception (or degenerate output) on a real contract."""

    contract_id: str
    strategy: str
    kind: str
    message: str
    contract_chars: int
    excerpt: str

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly mapping."""
        return {
            "contract_id": self.contract_id,
            "strategy": self.strategy,
            "kind": self.kind,
            "message": self.message,
            "contract_chars": self.contract_chars,
            "excerpt": self.excerpt,
        }


@dataclass(slots=True)
class StructureResult:
    """How often LexiChunk's parser recovered real clause structure."""

    min_clauses: int
    contracts: int
    parsed: int
    parse_rate: float
    fell_back: int
    fallback_rate: float
    parsed_any_level: int
    parse_rate_any_level: float
    mean_top_level_nodes: float
    median_top_level_nodes: float
    mean_total_nodes: float
    median_total_nodes: float
    single_chunk_contracts: int
    exceptions: int
    jurisdiction_comparison: dict[str, float] | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly mapping."""
        return {
            "min_clauses": self.min_clauses,
            "contracts": self.contracts,
            "parsed_top_level": self.parsed,
            "parse_rate_top_level": self.parse_rate,
            "fell_back": self.fell_back,
            "fallback_rate": self.fallback_rate,
            "parsed_any_level": self.parsed_any_level,
            "parse_rate_any_level": self.parse_rate_any_level,
            "mean_top_level_nodes": self.mean_top_level_nodes,
            "median_top_level_nodes": self.median_top_level_nodes,
            "mean_total_nodes": self.mean_total_nodes,
            "median_total_nodes": self.median_total_nodes,
            "single_chunk_contracts": self.single_chunk_contracts,
            "exceptions": self.exceptions,
            "jurisdiction_comparison": self.jurisdiction_comparison,
        }


@dataclass(slots=True)
class CuadResult:
    """Everything the CUAD evaluation produces."""

    dataset_id: str
    dataset_revision: str | None
    split: str
    seed: int
    requested_contracts: int
    contracts: int
    total_contracts_available: int
    spans: int
    mean_span_chars: float
    median_span_chars: float
    mean_contract_chars: float
    strategies: list[StrategyResult] = field(default_factory=list)
    structure: StructureResult | None = None
    failures: list[ParseFailure] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly mapping."""
        return {
            "evaluation": "cuad",
            "dataset": {
                "id": self.dataset_id,
                "revision": self.dataset_revision,
                "split": self.split,
                "contracts_available": self.total_contracts_available,
            },
            "sampling": {
                "seed": self.seed,
                "requested_contracts": self.requested_contracts,
                "contracts": self.contracts,
                "spans": self.spans,
            },
            "corpus": {
                "mean_span_chars": self.mean_span_chars,
                "median_span_chars": self.median_span_chars,
                "mean_contract_chars": self.mean_contract_chars,
            },
            "strategies": [s.as_dict() for s in self.strategies],
            "structure": self.structure.as_dict() if self.structure else None,
            "failures": [f.as_dict() for f in self.failures],
        }


def load_cuad_contracts(
    *,
    split: str = DEFAULT_SPLIT,
    sources: Sequence[tuple[str, str | None]] = CUAD_SOURCES,
) -> tuple[str, str | None, list[Contract]]:
    """Load CUAD and group its SQuAD-style QA rows into contracts.

    Rows sharing a ``title`` are one contract; their ``context`` is identical,
    and each row contributes zero or more gold answer spans. Spans whose
    recorded offset does not reproduce the answer text are dropped (they would
    make containment meaningless), and duplicates are collapsed.

    Returns:
        ``(dataset_id, revision, contracts)`` sorted by contract id for
        determinism.

    Raises:
        RuntimeError: If no source could be loaded.
    """
    from datasets import load_dataset

    errors: list[str] = []
    data = None
    used = ""
    used_revision: str | None = None
    for dataset_id, revision in sources:
        try:
            data = load_dataset(
                dataset_id, split=split, revision=revision, cache_dir=str(cache_dir())
            )
            used, used_revision = dataset_id, revision
            break
        except Exception as exc:  # noqa: BLE001 - fall through to the next source
            errors.append(f"{dataset_id}@{revision}[{split}]: {type(exc).__name__}: {exc}")
    if data is None:
        joined = "\n  ".join(errors)
        msg = f"could not load CUAD from any source:\n  {joined}"
        raise RuntimeError(msg)

    texts: dict[str, str] = {}
    spans: dict[str, set[tuple[int, int]]] = {}
    categories: dict[str, set[str]] = {}
    for row in data:
        title = str(row["title"])
        context = str(row["context"])
        texts.setdefault(title, context)
        bucket = spans.setdefault(title, set())
        answers = row["answers"]
        question = str(row.get("question", ""))
        for start, answer in zip(answers["answer_start"], answers["text"], strict=True):
            answer_text = str(answer)
            if not answer_text:
                continue
            start = int(start)
            end = start + len(answer_text)
            if context[start:end] != answer_text:
                continue
            bucket.add((start, end))
            categories.setdefault(title, set()).add(question)

    contracts = [
        Contract(
            contract_id=title,
            text=texts[title],
            spans=tuple(Span(s, e) for s, e in sorted(spans.get(title, set()))),
            categories=tuple(sorted(categories.get(title, set()))),
        )
        for title in sorted(texts)
    ]
    return used, used_revision, contracts


def default_strategies() -> list[StrategySpec]:
    """The chunking configurations compared on CUAD.

    ``lexichunk-128tok`` exists purely to defuse the length confound: at four
    characters per token it targets the same ~512-character chunks as
    ``rcts-512``, so the two are compared on equal terms.
    """
    return [
        StrategySpec(
            "lexichunk-512tok",
            lambda t: lexichunk_spans(t, jurisdiction="us", max_chunk_size=512),
            "LexiChunk, max 512 tokens (~2048 chars)",
        ),
        StrategySpec(
            "lexichunk-1024tok",
            lambda t: lexichunk_spans(t, jurisdiction="us", max_chunk_size=1024),
            "LexiChunk, max 1024 tokens (~4096 chars)",
        ),
        StrategySpec(
            "lexichunk-128tok",
            lambda t: lexichunk_spans(t, jurisdiction="us", max_chunk_size=128),
            "LexiChunk length-matched to rcts-512 (~512 chars)",
        ),
        StrategySpec("rcts-512", lambda t: rcts_spans(t, 512), "RecursiveCharacterTextSplitter"),
        StrategySpec("rcts-1024", lambda t: rcts_spans(t, 1024), "RecursiveCharacterTextSplitter"),
        StrategySpec(
            "sentence-window-3",
            lambda t: sentence_window_spans(t, window=3, stride=2),
            "3-sentence windows, stride 2 (overlapping)",
        ),
    ]


def evaluate_strategy(
    spec: StrategySpec,
    contracts: Sequence[Contract],
    *,
    seed: int = 0,
    n_boot: int = DEFAULT_BOOTSTRAP,
    failures: list[ParseFailure] | None = None,
) -> StrategyResult:
    """Measure containment, chunk geometry and cost for one strategy."""
    span_hits: list[bool] = []
    per_contract_rates: list[float] = []
    chunk_lengths: list[int] = []
    chunks_per_contract: list[int] = []
    spanned_counts: Counter[int] = Counter()
    span_lengths: list[int] = []
    elapsed = 0.0
    failed = 0

    # Warm up so the first contract is not charged for importing LangChain or
    # LexiChunk; otherwise "seconds per contract" measures import cost. A crash
    # here is ignored because it is recorded per contract in the loop below.
    with suppress(Exception):
        spec.build("1. Warm up. This sentence exists only to import the splitter.")

    for contract in contracts:
        start = time.perf_counter()
        try:
            spans = spec.build(contract.text)
        except Exception as exc:  # noqa: BLE001 - a crash is a finding, not a stop
            failed += 1
            if failures is not None:
                failures.append(_capture_failure(contract, spec.name, exc))
            continue
        finally:
            elapsed += time.perf_counter() - start

        chunks_per_contract.append(len(spans))
        chunk_lengths.extend(s.length for s in spans)

        hits: list[bool] = []
        for gold in contract.spans:
            span_lengths.append(gold.length)
            contained = any(chunk.contains(gold) for chunk in spans)
            overlapping = sum(1 for chunk in spans if chunk.overlaps(gold))
            spanned_counts[overlapping] += 1
            hits.append(contained)
            span_hits.append(contained)
        if hits:
            per_contract_rates.append(mean([1.0 if h else 0.0 for h in hits]))

    mean_len = mean([float(x) for x in chunk_lengths])
    grid = _grid_expectation(span_lengths, mean_len)
    containment = bootstrap_ci(
        span_hits, lambda xs: mean([1.0 if x else 0.0 for x in xs]), n_boot=n_boot, seed=seed
    )
    by_contract = bootstrap_ci(per_contract_rates, mean, n_boot=n_boot, seed=seed + 1)

    total_spans = sum(spanned_counts.values())
    return StrategyResult(
        name=spec.name,
        note=spec.note,
        containment=containment,
        per_contract_containment=by_contract,
        mean_chunk_chars=mean_len,
        median_chunk_chars=_median([float(x) for x in chunk_lengths]),
        mean_chunks_per_contract=mean([float(x) for x in chunks_per_contract]),
        grid_expected_containment=grid,
        lift=containment.point - grid,
        chunks_per_span={
            _bucket_label(k): v for k, v in sorted(_bucketise(spanned_counts).items())
        },
        mean_chunks_per_span=(
            sum(k * v for k, v in spanned_counts.items()) / total_spans if total_spans else 0.0
        ),
        seconds_per_contract=elapsed / len(contracts) if contracts else 0.0,
        spans_evaluated=len(span_hits),
        failures=failed,
    )


def evaluate_structure(
    contracts: Sequence[Contract],
    *,
    min_clauses: int = DEFAULT_MIN_CLAUSES,
    jurisdiction: str = "us",
    failures: list[ParseFailure] | None = None,
) -> StructureResult:
    """Measure how often LexiChunk recovers real clause structure on CUAD.

    Two thresholds are reported, because the strict one alone would overstate
    the failure:

    * *top level* — at least ``min_clauses`` level-0 nodes, i.e. the parser
      identified the contract's numbered sections as sections;
    * *any level* — at least ``min_clauses`` nodes at any depth, i.e. it found
      *some* hierarchy even if it never anchored a top level.

    A contract below the "any level" bar has effectively been treated as flat
    text, which is the failure mode that matters for a structure-aware chunker.

    ``jurisdiction_comparison`` re-runs the parser under the alternative
    jurisdiction profile, because a large gap between the two means the
    numbering styles in the corpus are recognised by one profile and not the
    other rather than being genuinely unstructured.
    """
    from lexichunk import LegalChunker

    chunker = LegalChunker(jurisdiction=jurisdiction, max_chunk_size=512)
    alternative = "uk" if jurisdiction == "us" else "us"
    other = LegalChunker(jurisdiction=alternative, max_chunk_size=512)

    top_counts: list[float] = []
    total_counts: list[float] = []
    other_top_counts: list[float] = []
    parsed = 0
    parsed_any = 0
    other_parsed = 0
    single_chunk = 0
    exceptions = 0

    for contract in contracts:
        try:
            nodes = chunker.parse_structure(contract.text)
            chunks = chunker.chunk(contract.text)
        except Exception as exc:  # noqa: BLE001 - a crash is a finding
            exceptions += 1
            if failures is not None:
                failures.append(_capture_failure(contract, "parse_structure", exc))
            continue
        top = sum(1 for n in nodes if getattr(n, "level", -1) == 0)
        top_counts.append(float(top))
        total_counts.append(float(len(nodes)))
        if top >= min_clauses:
            parsed += 1
        if len(nodes) >= min_clauses:
            parsed_any += 1
        if len(chunks) <= 1:
            single_chunk += 1

        with suppress(Exception):
            other_nodes = other.parse_structure(contract.text)
            other_top = sum(1 for n in other_nodes if getattr(n, "level", -1) == 0)
            other_top_counts.append(float(other_top))
            if other_top >= min_clauses:
                other_parsed += 1

    scored = len(top_counts)
    comparison = None
    if other_top_counts:
        comparison = {
            f"{jurisdiction}_parse_rate_top_level": parsed / scored if scored else 0.0,
            f"{alternative}_parse_rate_top_level": other_parsed / len(other_top_counts),
            f"{jurisdiction}_mean_top_level_nodes": mean(top_counts),
            f"{alternative}_mean_top_level_nodes": mean(other_top_counts),
        }

    return StructureResult(
        min_clauses=min_clauses,
        contracts=len(contracts),
        parsed=parsed,
        parse_rate=parsed / scored if scored else 0.0,
        fell_back=scored - parsed_any,
        fallback_rate=(scored - parsed_any) / scored if scored else 0.0,
        parsed_any_level=parsed_any,
        parse_rate_any_level=parsed_any / scored if scored else 0.0,
        mean_top_level_nodes=mean(top_counts),
        median_top_level_nodes=_median(top_counts),
        mean_total_nodes=mean(total_counts),
        median_total_nodes=_median(total_counts),
        single_chunk_contracts=single_chunk,
        exceptions=exceptions,
        jurisdiction_comparison=comparison,
    )


def run_cuad(
    *,
    contracts: int = 100,
    seed: int = 0,
    split: str = DEFAULT_SPLIT,
    n_boot: int = DEFAULT_BOOTSTRAP,
    min_clauses: int = DEFAULT_MIN_CLAUSES,
    strategies: Sequence[StrategySpec] | None = None,
    provider: Callable[[], tuple[str, str | None, list[Contract]]] | None = None,
) -> CuadResult:
    """Run the full CUAD evaluation.

    Args:
        contracts: Number of contracts to sample (0 = all of them).
        seed: Seed for the contract sample and the bootstrap.
        split: CUAD split to draw from.
        n_boot: Bootstrap resamples per interval.
        min_clauses: Threshold for "structure was parsed".
        strategies: Override the compared configurations.
        provider: Override for corpus loading. Used by ``--smoke`` and by the
            tests to run the whole pipeline offline.

    Returns:
        A :class:`CuadResult`.
    """
    if provider is not None:
        dataset_id, revision, all_contracts = provider()
    else:
        dataset_id, revision, all_contracts = load_cuad_contracts(split=split)
    with_spans = [c for c in all_contracts if c.spans]
    sample = deterministic_sample(with_spans, contracts, seed)

    span_lengths = [float(s.length) for c in sample for s in c.spans]
    failures: list[ParseFailure] = []
    specs = list(strategies) if strategies is not None else default_strategies()

    results = [
        evaluate_strategy(spec, sample, seed=seed, n_boot=n_boot, failures=failures)
        for spec in specs
    ]
    structure = evaluate_structure(sample, min_clauses=min_clauses, failures=failures)

    return CuadResult(
        dataset_id=dataset_id,
        dataset_revision=revision,
        split=split,
        seed=seed,
        requested_contracts=contracts,
        contracts=len(sample),
        total_contracts_available=len(all_contracts),
        spans=sum(len(c.spans) for c in sample),
        mean_span_chars=mean(span_lengths),
        median_span_chars=_median(span_lengths),
        mean_contract_chars=mean([float(len(c.text)) for c in sample]),
        strategies=results,
        structure=structure,
        failures=failures,
    )


def render_markdown(result: CuadResult) -> str:
    """Render the CUAD result as markdown tables."""
    lines = ["## CUAD: gold-span containment under different chunkers", ""]
    lines.append(
        f"Dataset `{result.dataset_id}`"
        + (f" at revision `{result.dataset_revision}`" if result.dataset_revision else "")
        + f", split `{result.split}`; {result.contracts} of "
        f"{result.total_contracts_available} contracts sampled with seed {result.seed}, "
        f"carrying {result.spans:,} verified gold answer spans "
        f"(mean {result.mean_span_chars:.0f} chars, median "
        f"{result.median_span_chars:.0f}). Mean contract length "
        f"{result.mean_contract_chars:,.0f} chars."
    )
    lines += ["", "### Containment", ""]
    lines.append(
        markdown_table(
            [
                "Strategy",
                "Containment (95% CI)",
                "Mean chunk chars",
                "Chunks/contract",
                "Length-matched grid",
                "Lift",
                "s/contract",
            ],
            [
                [
                    s.name,
                    s.containment.format(),
                    f"{s.mean_chunk_chars:,.0f}",
                    f"{s.mean_chunks_per_contract:.1f}",
                    f"{s.grid_expected_containment * 100:.1f}%",
                    f"{s.lift * 100:+.1f}pp",
                    f"{s.seconds_per_contract:.3f}",
                ]
                for s in result.strategies
            ],
        )
    )
    lines += [
        "",
        "`Length-matched grid` is the containment a fixed-stride splitter with the same "
        "mean chunk length would get on these spans; `Lift` is the strategy's containment "
        "minus that. Lift, not raw containment, is the number that reflects boundary quality.",
        "",
        "### How many chunks a gold span is split across",
        "",
    ]
    buckets = sorted({k for s in result.strategies for k in s.chunks_per_span})
    lines.append(
        markdown_table(
            ["Strategy", *buckets, "Mean"],
            [
                [
                    s.name,
                    *[_pct(s.chunks_per_span.get(b, 0), s.spans_evaluated) for b in buckets],
                    f"{s.mean_chunks_per_span:.2f}",
                ]
                for s in result.strategies
            ],
        )
    )

    if result.structure is not None:
        st = result.structure
        lines += ["", "### LexiChunk structure recall on CUAD", ""]
        lines.append(
            markdown_table(
                ["Metric", "Value"],
                [
                    [
                        f"Contracts with >= {st.min_clauses} TOP-LEVEL clauses",
                        f"{st.parsed} ({st.parse_rate * 100:.1f}%)",
                    ],
                    [
                        f"Contracts with >= {st.min_clauses} nodes at ANY level",
                        f"{st.parsed_any_level} ({st.parse_rate_any_level * 100:.1f}%)",
                    ],
                    ["Fell back to flat text", f"{st.fell_back} ({st.fallback_rate * 100:.1f}%)"],
                    [
                        "Mean / median top-level nodes",
                        f"{st.mean_top_level_nodes:.1f} / {st.median_top_level_nodes:.0f}",
                    ],
                    [
                        "Mean / median total nodes",
                        f"{st.mean_total_nodes:.1f} / {st.median_total_nodes:.0f}",
                    ],
                    ["Contracts yielding a single chunk", f"{st.single_chunk_contracts}"],
                    ["Exceptions during parsing", f"{st.exceptions}"],
                ],
            )
        )
        if st.jurisdiction_comparison:
            lines += [
                "",
                "Same contracts, both jurisdiction profiles (a gap means the corpus's "
                "numbering style is recognised by one profile and not the other):",
                "",
            ]
            lines.append(
                markdown_table(
                    ["Measure", "Value"],
                    [
                        [k, f"{v * 100:.1f}%" if "rate" in k else f"{v:.1f}"]
                        for k, v in st.jurisdiction_comparison.items()
                    ],
                )
            )

    lines += ["", "### Failures", ""]
    if result.failures:
        counts = Counter((f.strategy, f.kind) for f in result.failures)
        lines.append(
            markdown_table(
                ["Strategy", "Exception", "Count"],
                [[s, k, str(n)] for (s, k), n in counts.most_common()],
            )
        )
    else:
        lines.append("No exceptions were raised on any sampled contract.")
    return "\n".join(lines) + "\n"


def _capture_failure(contract: Contract, strategy: str, exc: BaseException) -> ParseFailure:
    """Record enough about a crash to reproduce it later."""
    tb = traceback.format_exception_only(type(exc), exc)
    return ParseFailure(
        contract_id=contract.contract_id,
        strategy=strategy,
        kind=type(exc).__name__,
        message="".join(tb).strip(),
        contract_chars=len(contract.text),
        excerpt=contract.text[:400],
    )


def _grid_expectation(span_lengths: Sequence[float], mean_chunk_chars: float) -> float:
    """Containment expected from a fixed-stride grid of period ``mean_chunk_chars``.

    A span of length ``s`` placed uniformly at random against boundaries every
    ``L`` characters survives intact with probability ``max(0, 1 - s / L)``.
    """
    if not span_lengths or mean_chunk_chars <= 0:
        return 0.0
    return mean([max(0.0, 1.0 - s / mean_chunk_chars) for s in span_lengths])


def _bucketise(counts: Counter[int]) -> dict[int, int]:
    """Collapse the tail of the chunks-per-span distribution into a 4+ bucket."""
    out: dict[int, int] = {}
    for k, v in counts.items():
        key = k if k <= 3 else 4
        out[key] = out.get(key, 0) + v
    return out


def _bucket_label(key: int) -> str:
    """Human label for a chunks-per-span bucket."""
    return "4+" if key >= 4 else str(key)


def _pct(count: int, total: int) -> str:
    """Format ``count`` as a percentage of ``total``."""
    return f"{count / total * 100:.1f}%" if total else "-"


def _median(values: Sequence[float]) -> float:
    """Median of ``values``; 0.0 when empty."""
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0
