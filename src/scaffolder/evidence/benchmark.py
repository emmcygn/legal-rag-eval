"""Deterministic boundary-only anchored-evidence benchmark."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import yaml

from scaffolder import __version__
from scaffolder.evidence.lexical import RankedCandidate, SpanCandidate, rank_lexical
from scaffolder.evidence.schema import (
    DatasetError,
    DocumentRecord,
    EvidenceDataset,
    load_dataset,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from scaffolder.evidence.schema import EvidenceSpan

REPORT_VERSION = "evidence_benchmark_report_v1"
EVIDENCE_METRIC_VERSION = "source_span_coverage_v1"
LEXICAL_METRIC_VERSION = "cosine_term_frequency_v1"
SUPPORTED_STRATEGIES = frozenset({"lexichunk", "token_window", "rcts"})
_TOKEN_SPAN_PATTERN = re.compile(r"\S+")
DEFAULT_DATASET_PATH = Path(__file__).parents[1] / "data" / "synthetic_contracts_v1.json"
DEFAULT_OUTPUT_PATH = Path("results/evidence-benchmark.json")


@dataclass(frozen=True, slots=True)
class EvidenceBenchmarkConfig:
    dataset_path: Path = DEFAULT_DATASET_PATH
    output_path: Path = DEFAULT_OUTPUT_PATH
    strategies: tuple[str, ...] = ("lexichunk", "token_window", "rcts")
    retrieval_depth: int = 5
    context_budget_tokens: int = 128
    token_window_tokens: int = 64
    token_window_overlap: int = 8
    rcts_chunk_chars: int = 512
    rcts_chunk_overlap: int = 50
    lexichunk_max_tokens: int = 512

    @classmethod
    def load(
        cls,
        config_path: Path | None = None,
        *,
        dataset_path: Path | None = None,
        output_path: Path | None = None,
    ) -> EvidenceBenchmarkConfig:
        """Load resolved benchmark settings from YAML plus explicit CLI overrides."""
        values: dict[str, object] = {}
        base_directory = Path.cwd()
        if config_path is not None:
            resolved_config = config_path.resolve()
            try:
                loaded = yaml.safe_load(resolved_config.read_text(encoding="utf-8"))
            except FileNotFoundError as error:
                raise ValueError(f"benchmark config not found: {config_path}") from error
            if loaded is None:
                loaded = {}
            if not isinstance(loaded, dict):
                raise ValueError("benchmark config root must be an object")
            allowed = set(cls.__dataclass_fields__)
            unknown = set(loaded) - allowed
            if unknown:
                raise ValueError(f"unknown evidence benchmark config keys: {sorted(unknown)}")
            values.update(loaded)
            base_directory = resolved_config.parent

        if "strategies" in values:
            strategies = values["strategies"]
            if not isinstance(strategies, list) or not all(
                isinstance(item, str) for item in strategies
            ):
                raise ValueError("strategies must be a list of strings")
            values["strategies"] = tuple(strategies)
        for key in ("dataset_path", "output_path"):
            if key in values:
                raw_path = values[key]
                if not isinstance(raw_path, str):
                    raise ValueError(f"{key} must be a string path")
                candidate = Path(raw_path)
                values[key] = candidate if candidate.is_absolute() else base_directory / candidate
        if dataset_path is not None:
            values["dataset_path"] = dataset_path
        if output_path is not None:
            values["output_path"] = output_path
        defaults = cls()
        config = cls(
            dataset_path=_config_path(values, "dataset_path", defaults.dataset_path),
            output_path=_config_path(values, "output_path", defaults.output_path),
            strategies=cast("tuple[str, ...]", values.get("strategies", defaults.strategies)),
            retrieval_depth=_config_int(values, "retrieval_depth", defaults.retrieval_depth),
            context_budget_tokens=_config_int(
                values,
                "context_budget_tokens",
                defaults.context_budget_tokens,
            ),
            token_window_tokens=_config_int(
                values,
                "token_window_tokens",
                defaults.token_window_tokens,
            ),
            token_window_overlap=_config_int(
                values,
                "token_window_overlap",
                defaults.token_window_overlap,
            ),
            rcts_chunk_chars=_config_int(
                values,
                "rcts_chunk_chars",
                defaults.rcts_chunk_chars,
            ),
            rcts_chunk_overlap=_config_int(
                values,
                "rcts_chunk_overlap",
                defaults.rcts_chunk_overlap,
            ),
            lexichunk_max_tokens=_config_int(
                values,
                "lexichunk_max_tokens",
                defaults.lexichunk_max_tokens,
            ),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not isinstance(self.dataset_path, Path) or not isinstance(self.output_path, Path):
            raise ValueError("dataset_path and output_path must be Path values")
        if not isinstance(self.strategies, tuple) or not all(
            isinstance(strategy, str) for strategy in self.strategies
        ):
            raise ValueError("strategies must be a tuple of strings")
        unknown = set(self.strategies) - SUPPORTED_STRATEGIES
        if unknown:
            raise ValueError(f"unknown evidence benchmark strategies: {sorted(unknown)}")
        if not self.strategies or len(self.strategies) != len(set(self.strategies)):
            raise ValueError("strategies must be non-empty and unique")
        _require_positive_int("retrieval_depth", self.retrieval_depth)
        _require_positive_int("context_budget_tokens", self.context_budget_tokens)
        _require_positive_int("token_window_tokens", self.token_window_tokens)
        _require_nonnegative_int("token_window_overlap", self.token_window_overlap)
        if not 0 <= self.token_window_overlap < self.token_window_tokens:
            raise ValueError("token_window_overlap must be smaller than token_window_tokens")
        _require_positive_int("rcts_chunk_chars", self.rcts_chunk_chars)
        _require_nonnegative_int("rcts_chunk_overlap", self.rcts_chunk_overlap)
        if not 0 <= self.rcts_chunk_overlap < self.rcts_chunk_chars:
            raise ValueError("rcts_chunk_overlap must be smaller than rcts_chunk_chars")
        _require_positive_int("lexichunk_max_tokens", self.lexichunk_max_tokens)


@dataclass(frozen=True, slots=True)
class SelectedSpan:
    document_id: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class EvidenceScore:
    evidence_recall: float | None
    evidence_precision: float | None
    abstention_correct: bool
    selected_context_chars: int
    covered_evidence_chars: int
    total_evidence_chars: int


def run_benchmark(config: EvidenceBenchmarkConfig) -> dict[str, object]:
    """Run the deterministic benchmark and write one self-identifying JSON report."""
    config.validate()
    dataset = load_dataset(config.dataset_path)
    evidence_by_id = {span.id: span for span in dataset.evidence}
    results: dict[str, list[dict[str, object]]] = {}
    aggregates: dict[str, dict[str, object]] = {}

    for strategy in config.strategies:
        candidates = _strategy_candidates(strategy, dataset.documents, config)
        strategy_results: list[dict[str, object]] = []
        for query in dataset.queries:
            ranked = rank_lexical(query.text, candidates, config.retrieval_depth)
            selected = _select_ranked(ranked, config.context_budget_tokens)
            evidence = tuple(evidence_by_id[evidence_id] for evidence_id in query.evidence_ids)
            score = score_selected_spans(selected, evidence, answerable=query.answerable)
            record: dict[str, object] = {
                "query_id": query.id,
                "answerable": query.answerable,
                "selected_context_chars": score.selected_context_chars,
                "selected_context_tokens": sum(
                    _span_token_count(span, dataset) for span in selected
                ),
                "abstention_correct": score.abstention_correct,
                "retrieved": [
                    {
                        "candidate_id": item.candidate.id,
                        "document_id": item.candidate.document_id,
                        "source_start": item.candidate.start,
                        "source_end": item.candidate.end,
                        "rank": item.rank,
                        "score": item.score,
                    }
                    for item in ranked
                ],
                "selected_source_spans": [asdict(span) for span in selected],
            }
            if query.answerable:
                record.update(
                    evidence_recall=score.evidence_recall,
                    evidence_precision=score.evidence_precision,
                    covered_evidence_chars=score.covered_evidence_chars,
                    total_evidence_chars=score.total_evidence_chars,
                )
            strategy_results.append(record)
        results[strategy] = strategy_results
        aggregates[strategy] = _aggregate(strategy_results)

    report: dict[str, object] = {
        "report_version": REPORT_VERSION,
        "metric_versions": {
            "evidence_coverage": EVIDENCE_METRIC_VERSION,
            "lexical_ranker": LEXICAL_METRIC_VERSION,
        },
        "scope": {
            "retrieval": "deterministic lexical cosine term-frequency ranking",
            "lexichunk_mode": "boundary_only_source_spans",
            "generated_context_headers": "not_evaluated",
            "definition_expansion": "not_evaluated",
            "cross_reference_expansion": "not_evaluated",
        },
        "evaluator": {
            "version": __version__,
            "package_tree_sha256": _package_tree_sha256("scaffolder"),
        },
        "lexichunk": {
            "version": _distribution_version("lexichunk"),
            "package_tree_sha256": _package_tree_sha256("lexichunk"),
        },
        "corpus": _corpus_manifest(dataset),
        "config": _public_config(config, dataset),
        "results": results,
        "aggregates": aggregates,
        "comparisons": _comparisons(aggregates),
    }
    report["payload_sha256"] = _canonical_hash(report)
    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def score_selected_spans(
    selected: Sequence[SelectedSpan],
    evidence: Sequence[EvidenceSpan],
    *,
    answerable: bool,
) -> EvidenceScore:
    """Score exact source overlap while charging all selected context characters."""
    _validate_selected_spans(selected)
    selected_context_chars = sum(span.end - span.start for span in selected)
    if not answerable:
        correct = selected_context_chars == 0
        return EvidenceScore(None, None, correct, selected_context_chars, 0, 0)

    evidence_intervals = _merge_intervals(
        (span.document_id, span.start, span.end) for span in evidence
    )
    total_evidence_chars = sum(end - start for _, start, end in evidence_intervals)
    covered = _covered_evidence_chars(selected, evidence_intervals)
    recall = covered / total_evidence_chars if total_evidence_chars else 0.0
    precision = covered / selected_context_chars if selected_context_chars else 0.0
    return EvidenceScore(
        evidence_recall=recall,
        evidence_precision=precision,
        abstention_correct=selected_context_chars > 0,
        selected_context_chars=selected_context_chars,
        covered_evidence_chars=covered,
        total_evidence_chars=total_evidence_chars,
    )


def _strategy_candidates(
    strategy: str,
    documents: Sequence[DocumentRecord],
    config: EvidenceBenchmarkConfig,
) -> tuple[SpanCandidate, ...]:
    if strategy == "lexichunk":
        return _lexichunk_candidates(documents, config.lexichunk_max_tokens)
    if strategy == "token_window":
        return _token_window_candidates(
            documents,
            config.token_window_tokens,
            config.token_window_overlap,
        )
    if strategy == "rcts":
        return _rcts_candidates(documents, config.rcts_chunk_chars, config.rcts_chunk_overlap)
    raise ValueError(f"unsupported evidence benchmark strategy: {strategy}")


def _lexichunk_candidates(
    documents: Sequence[DocumentRecord],
    max_chunk_tokens: int,
) -> tuple[SpanCandidate, ...]:
    lexichunk = importlib.import_module("lexichunk")
    chunker_class = lexichunk.LegalChunker
    candidates: list[SpanCandidate] = []
    for document in documents:
        if chunker_class.sanitize(document.text) != document.text:
            raise DatasetError(
                f"document {document.id} must already match LexiChunk sanitized source text"
            )
        chunker = chunker_class(
            jurisdiction=document.jurisdiction,
            max_chunk_size=max_chunk_tokens,
            min_chunk_size=1,
            include_context_header=False,
        )
        for index, chunk in enumerate(chunker.chunk(document.text, document_id=document.id)):
            start = getattr(chunk, "char_start", None)
            end = getattr(chunk, "char_end", None)
            if (
                not isinstance(start, int)
                or isinstance(start, bool)
                or not isinstance(end, int)
                or isinstance(end, bool)
                or start < 0
                or end <= start
                or end > len(document.text)
            ):
                raise DatasetError(f"LexiChunk returned invalid source offsets for {document.id}")
            source_text = document.text[start:end]
            candidates.append(
                SpanCandidate(
                    id=f"lexichunk:{document.id}:{index}",
                    document_id=document.id,
                    start=start,
                    end=end,
                    text=source_text,
                )
            )
    return tuple(candidates)


def _token_window_candidates(
    documents: Sequence[DocumentRecord],
    window_tokens: int,
    overlap_tokens: int,
) -> tuple[SpanCandidate, ...]:
    step = window_tokens - overlap_tokens
    candidates: list[SpanCandidate] = []
    for document in documents:
        token_spans = [
            (match.start(), match.end()) for match in _TOKEN_SPAN_PATTERN.finditer(document.text)
        ]
        for index, token_start in enumerate(range(0, len(token_spans), step)):
            window = token_spans[token_start : token_start + window_tokens]
            if not window:
                continue
            start = window[0][0]
            end = window[-1][1]
            candidates.append(
                SpanCandidate(
                    id=f"token_window:{document.id}:{index}",
                    document_id=document.id,
                    start=start,
                    end=end,
                    text=document.text[start:end],
                )
            )
            if token_start + window_tokens >= len(token_spans):
                break
    return tuple(candidates)


def _rcts_candidates(
    documents: Sequence[DocumentRecord],
    chunk_chars: int,
    overlap_chars: int,
) -> tuple[SpanCandidate, ...]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_chars,
        chunk_overlap=overlap_chars,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )
    candidates: list[SpanCandidate] = []
    for document in documents:
        for index, split in enumerate(splitter.create_documents([document.text])):
            start = split.metadata.get("start_index")
            if not isinstance(start, int) or isinstance(start, bool) or start < 0:
                raise DatasetError(f"RCTS did not return a valid source offset for {document.id}")
            text = split.page_content
            end = start + len(text)
            if document.text[start:end] != text:
                raise DatasetError(f"RCTS source span mismatch for {document.id}")
            candidates.append(
                SpanCandidate(
                    id=f"rcts:{document.id}:{index}",
                    document_id=document.id,
                    start=start,
                    end=end,
                    text=text,
                )
            )
    return tuple(candidates)


def _select_ranked(
    ranked: Sequence[RankedCandidate],
    context_budget_tokens: int,
) -> tuple[SelectedSpan, ...]:
    remaining = context_budget_tokens
    selected: list[SelectedSpan] = []
    for item in ranked:
        token_spans = list(_TOKEN_SPAN_PATTERN.finditer(item.candidate.text))
        if not token_spans or remaining <= 0:
            break
        selected_tokens = min(remaining, len(token_spans))
        end_offset = token_spans[selected_tokens - 1].end()
        selected.append(
            SelectedSpan(
                document_id=item.candidate.document_id,
                start=item.candidate.start,
                end=item.candidate.start + end_offset,
            )
        )
        remaining -= selected_tokens
    return tuple(selected)


def _span_token_count(span: SelectedSpan, dataset: EvidenceDataset) -> int:
    document = next(item for item in dataset.documents if item.id == span.document_id)
    return len(_TOKEN_SPAN_PATTERN.findall(document.text[span.start : span.end]))


def _aggregate(records: Sequence[dict[str, object]]) -> dict[str, object]:
    answerable = [record for record in records if record["answerable"] is True]
    unanswerable = [record for record in records if record["answerable"] is False]
    return {
        "answerable_query_count": len(answerable),
        "mean_evidence_recall": _mean(
            cast("float", record["evidence_recall"]) for record in answerable
        ),
        "mean_evidence_precision": _mean(
            cast("float", record["evidence_precision"]) for record in answerable
        ),
        "unanswerable_query_count": len(unanswerable),
        "abstention_accuracy": _mean(
            1.0 if record["abstention_correct"] is True else 0.0 for record in unanswerable
        ),
    }


def _mean(values: Iterable[float]) -> float | None:
    numeric = list(values)
    return sum(numeric) / len(numeric) if numeric else None


def _comparisons(
    aggregates: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    lexichunk = aggregates.get("lexichunk")
    if lexichunk is None:
        return []
    comparisons: list[dict[str, object]] = []
    for strategy, aggregate in aggregates.items():
        if strategy == "lexichunk":
            continue
        recall_delta = _delta(lexichunk["mean_evidence_recall"], aggregate["mean_evidence_recall"])
        precision_delta = _delta(
            lexichunk["mean_evidence_precision"], aggregate["mean_evidence_precision"]
        )
        comparisons.append(
            {
                "baseline": strategy,
                "lexichunk_recall_delta": recall_delta,
                "lexichunk_precision_delta": precision_delta,
                "outcome": _outcome(recall_delta, precision_delta),
            }
        )
    return comparisons


def _delta(first: object, second: object) -> float | None:
    if not isinstance(first, (int, float)) or not isinstance(second, (int, float)):
        return None
    return float(first) - float(second)


def _outcome(recall_delta: float | None, precision_delta: float | None) -> str:
    if recall_delta is None or precision_delta is None:
        return "not_comparable"
    if recall_delta == 0 and precision_delta == 0:
        return "tie"
    if recall_delta >= 0 and precision_delta >= 0:
        return "lexichunk_win"
    if recall_delta <= 0 and precision_delta <= 0:
        return "lexichunk_loss"
    return "mixed"


def _corpus_manifest(dataset: EvidenceDataset) -> dict[str, object]:
    return {
        "id": dataset.corpus.id,
        "title": dataset.corpus.title,
        "description": dataset.corpus.description,
        "authorship": dataset.corpus.authorship,
        "legal_validation": dataset.corpus.legal_validation,
        "held_out": dataset.corpus.held_out,
        "customer_proof": dataset.corpus.customer_proof,
        "manifest_sha256": dataset.manifest_sha256,
        "documents": [
            {
                "id": document.id,
                "jurisdiction": document.jurisdiction,
                "path": document.path,
                "sha256": document.sha256,
                "provenance": {
                    "kind": document.provenance_kind,
                    "source_url": document.source_url,
                },
                "license_status": document.license_status,
                "license_identifier": document.license_identifier,
                "review_status": document.review_status,
            }
            for document in dataset.documents
        ],
    }


def _public_config(
    config: EvidenceBenchmarkConfig,
    dataset: EvidenceDataset,
) -> dict[str, object]:
    return {
        "dataset_id": dataset.corpus.id,
        "strategies": list(config.strategies),
        "retrieval_depth": config.retrieval_depth,
        "context_budget_tokens": config.context_budget_tokens,
        "token_budget_unit": "whitespace_tokens_regex_\\S+",
        "token_window_tokens": config.token_window_tokens,
        "token_window_overlap": config.token_window_overlap,
        "rcts_chunk_chars": config.rcts_chunk_chars,
        "rcts_chunk_overlap": config.rcts_chunk_overlap,
        "lexichunk_max_tokens": config.lexichunk_max_tokens,
        "output_filename": Path(config.output_path).name,
    }


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _package_tree_sha256(package_name: str) -> str:
    package = importlib.import_module(package_name)
    package_file = getattr(package, "__file__", None)
    if package_file is None:
        return "unavailable"
    package_root = Path(package_file).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(package_root.rglob("*.py")):
        relative = path.relative_to(package_root).as_posix().encode()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        content = path.read_bytes()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _validate_selected_spans(selected: Sequence[SelectedSpan]) -> None:
    for span in selected:
        if not isinstance(span.document_id, str) or not span.document_id.strip():
            raise ValueError("selected span document_id must be a non-empty string")
        if not isinstance(span.start, int) or isinstance(span.start, bool) or span.start < 0:
            raise ValueError("selected span start must be a nonnegative integer")
        if not isinstance(span.end, int) or isinstance(span.end, bool) or span.end <= span.start:
            raise ValueError("selected span end must be an integer greater than start")


def _config_path(values: dict[str, object], key: str, default: Path) -> Path:
    value = values.get(key, default)
    if not isinstance(value, Path):
        raise ValueError(f"{key} must be a path")
    return value


def _config_int(values: dict[str, object], key: str, default: int) -> int:
    value = values.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key} must be an integer")
    return value


def _require_positive_int(name: str, value: object) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _require_nonnegative_int(name: str, value: object) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _covered_evidence_chars(
    selected: Sequence[SelectedSpan],
    evidence: tuple[tuple[str, int, int], ...],
) -> int:
    intersections: list[tuple[str, int, int]] = []
    for selected_span in selected:
        for document_id, start, end in evidence:
            if selected_span.document_id != document_id:
                continue
            overlap_start = max(selected_span.start, start)
            overlap_end = min(selected_span.end, end)
            if overlap_end > overlap_start:
                intersections.append((document_id, overlap_start, overlap_end))
    return sum(end - start for _, start, end in _merge_intervals(intersections))


def _merge_intervals(
    intervals: Iterable[tuple[str, int, int]],
) -> tuple[tuple[str, int, int], ...]:
    ordered = sorted(intervals)
    merged: list[tuple[str, int, int]] = []
    for document_id, start, end in ordered:
        if not merged or merged[-1][0] != document_id or start > merged[-1][2]:
            merged.append((document_id, start, end))
            continue
        previous_document, previous_start, previous_end = merged[-1]
        merged[-1] = (previous_document, previous_start, max(previous_end, end))
    return tuple(merged)
