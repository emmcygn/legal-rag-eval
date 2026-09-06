from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import lexichunk

from scaffolder.evidence.benchmark import (
    EvidenceBenchmarkConfig,
    SelectedSpan,
    _lexichunk_candidates,
    _select_ranked,
    run_benchmark,
    score_selected_spans,
)
from scaffolder.evidence.lexical import RankedCandidate, SpanCandidate
from scaffolder.evidence.schema import DocumentRecord, EvidenceSpan

if TYPE_CHECKING:
    import pytest


def test_score_selected_spans_perfect_multi_evidence() -> None:
    evidence = (
        EvidenceSpan("first", "doc", 0, 5, "alpha", 3),
        EvidenceSpan("second", "doc", 10, 15, "bravo", 3),
    )
    selected = (SelectedSpan("doc", 0, 5), SelectedSpan("doc", 10, 15))

    score = score_selected_spans(selected, evidence, answerable=True)

    assert score.evidence_recall == 1.0
    assert score.evidence_precision == 1.0
    assert score.abstention_correct is True


def test_score_selected_spans_penalizes_extra_context_without_double_counting_overlap() -> None:
    evidence = (EvidenceSpan("gold", "doc", 5, 10, "12345", 3),)
    selected = (SelectedSpan("doc", 0, 10), SelectedSpan("doc", 5, 15))

    score = score_selected_spans(selected, evidence, answerable=True)

    assert score.evidence_recall == 1.0
    assert score.evidence_precision == 0.25


def test_score_selected_spans_reports_omitted_evidence() -> None:
    evidence = (EvidenceSpan("gold", "doc", 0, 10, "0123456789", 3),)

    score = score_selected_spans((SelectedSpan("doc", 0, 5),), evidence, answerable=True)

    assert score.evidence_recall == 0.5
    assert score.evidence_precision == 1.0


def test_score_selected_spans_unanswerable_reports_abstention_separately() -> None:
    abstained = score_selected_spans((), (), answerable=False)
    false_positive = score_selected_spans((SelectedSpan("doc", 0, 5),), (), answerable=False)

    assert (abstained.evidence_recall, abstained.evidence_precision) == (None, None)
    assert abstained.abstention_correct is True
    assert (false_positive.evidence_recall, false_positive.evidence_precision) == (None, None)
    assert false_positive.abstention_correct is False


def test_score_selected_spans_rejects_wrong_document_credit() -> None:
    evidence = (EvidenceSpan("gold", "right", 0, 5, "alpha", 3),)

    score = score_selected_spans(
        (SelectedSpan("wrong", 0, 5),),
        evidence,
        answerable=True,
    )

    assert score.evidence_recall == 0.0
    assert score.evidence_precision == 0.0


def test_score_selected_spans_merges_duplicate_and_overlapping_gold() -> None:
    evidence = (
        EvidenceSpan("first", "doc", 0, 10, "0123456789", 3),
        EvidenceSpan("duplicate", "doc", 0, 10, "0123456789", 2),
        EvidenceSpan("overlap", "doc", 5, 15, "56789abcde", 1),
    )

    score = score_selected_spans(
        (SelectedSpan("doc", 0, 15),),
        evidence,
        answerable=True,
    )

    assert score.total_evidence_chars == 15
    assert score.covered_evidence_chars == 15
    assert score.evidence_recall == 1.0
    assert score.evidence_precision == 1.0


def test_select_ranked_clips_to_context_token_budget() -> None:
    candidate = SpanCandidate("chunk", "doc", 10, 26, "alpha beta gamma")
    ranked = (RankedCandidate(candidate, 1.0, 1),)

    selected = _select_ranked(ranked, context_budget_tokens=2)

    assert selected == (SelectedSpan("doc", 10, 20),)


def test_lexichunk_candidates_use_sdk_source_offsets_not_generated_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = "1 HEADING\nOwn clause text."

    class FakeChunker:
        @staticmethod
        def sanitize(text: str) -> str:
            return text

        def __init__(self, **_kwargs: object) -> None:
            pass

        def chunk(self, text: str, document_id: str) -> list[object]:
            start = text.index("Own")
            return [
                SimpleNamespace(
                    char_start=start,
                    char_end=len(text),
                    content="1 HEADING\nOwn clause text.",
                )
            ]

    monkeypatch.setattr(lexichunk, "LegalChunker", FakeChunker)
    document = DocumentRecord(
        id="doc",
        jurisdiction="uk",
        path="documents/doc.txt",
        sha256=hashlib.sha256(source.encode()).hexdigest(),
        provenance_kind="synthetic",
        source_url=None,
        license_status="project_license",
        license_identifier="MIT",
        review_status="not_human_reviewed",
        text=source,
    )

    candidates = _lexichunk_candidates((document,), max_chunk_tokens=40)

    assert candidates == (
        SpanCandidate("lexichunk:doc:0", "doc", 10, len(source), "Own clause text."),
    )


def _write_benchmark_dataset(tmp_path: Path) -> Path:
    text = "PAYMENT\nPayment is due in ten days.\n\nNOTICE\nNotice must be written."
    document_path = tmp_path / "documents" / "contract.txt"
    document_path.parent.mkdir()
    document_path.write_text(text, encoding="utf-8", newline="")
    payment_start = text.index("Payment is")
    payment_text = "Payment is due in ten days."
    payload = {
        "schema_version": "anchored_evidence_v1",
        "corpus": {
            "id": "runner-test",
            "title": "Runner test",
            "description": "Synthetic runner fixture.",
            "authorship": "ai_authored",
            "legal_validation": "not_human_reviewed",
            "held_out": False,
            "customer_proof": False,
        },
        "documents": [
            {
                "id": "contract",
                "jurisdiction": "uk",
                "path": "documents/contract.txt",
                "sha256": hashlib.sha256(text.encode()).hexdigest(),
                "provenance": {"kind": "synthetic", "source_url": None},
                "license_status": "project_license",
                "license_identifier": "MIT",
                "review_status": "not_human_reviewed",
            }
        ],
        "evidence": [
            {
                "id": "payment",
                "document_id": "contract",
                "start": payment_start,
                "end": payment_start + len(payment_text),
                "text": payment_text,
                "grade": 3,
            }
        ],
        "queries": [
            {
                "id": "payment",
                "text": "When is payment due?",
                "answerable": True,
                "answer_spans": [
                    {
                        "document_id": "contract",
                        "start": payment_start,
                        "end": payment_start + len(payment_text),
                        "text": payment_text,
                    }
                ],
                "evidence_ids": ["payment"],
            },
            {
                "id": "court",
                "text": "exclusive tribunal venue",
                "answerable": False,
                "answer_spans": [],
                "evidence_ids": [],
            },
        ],
    }
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(json.dumps(payload), encoding="utf-8")
    return dataset_path


def test_run_benchmark_emits_deterministic_hashed_report(tmp_path: Path) -> None:
    dataset_path = _write_benchmark_dataset(tmp_path)
    config = EvidenceBenchmarkConfig(
        dataset_path=dataset_path,
        output_path=tmp_path / "report.json",
        strategies=("token_window", "rcts"),
        retrieval_depth=3,
        context_budget_tokens=20,
        token_window_tokens=10,
        token_window_overlap=2,
        rcts_chunk_chars=80,
        rcts_chunk_overlap=10,
        lexichunk_max_tokens=40,
    )

    first = run_benchmark(config)
    second = run_benchmark(config)

    assert first == second
    assert first["report_version"] == "evidence_benchmark_report_v1"
    assert first["metric_versions"]["evidence_coverage"] == "source_span_coverage_v1"
    assert first["config"]["context_budget_tokens"] == 20
    assert first["config"]["token_budget_unit"] == "whitespace_tokens_regex_\\S+"
    assert len(first["corpus"]["manifest_sha256"]) == 64
    assert len(first["payload_sha256"]) == 64
    assert first["aggregates"]["token_window"]["answerable_query_count"] == 1
    assert first["aggregates"]["token_window"]["unanswerable_query_count"] == 1
    assert "evidence_recall" not in first["results"]["token_window"][1]
    assert Path(config.output_path).read_text(encoding="utf-8").endswith("\n")


def test_config_loads_yaml_and_rejects_unknown_keys(tmp_path: Path) -> None:
    config_path = tmp_path / "benchmark.yaml"
    config_path.write_text(
        "strategies: [token_window, rcts]\ncontext_budget_tokens: 40\n",
        encoding="utf-8",
    )

    config = EvidenceBenchmarkConfig.load(config_path)

    assert config.strategies == ("token_window", "rcts")
    assert config.context_budget_tokens == 40

    config_path.write_text("unknown_setting: true\n", encoding="utf-8")
    try:
        EvidenceBenchmarkConfig.load(config_path)
    except ValueError as error:
        assert "unknown" in str(error)
    else:
        raise AssertionError("unknown config key was accepted")


def test_config_rejects_falsey_nonmapping_yaml_roots(tmp_path: Path) -> None:
    config_path = tmp_path / "benchmark.yaml"

    for content in ("false\n", "0\n", "[]\n"):
        config_path.write_text(content, encoding="utf-8")
        try:
            EvidenceBenchmarkConfig.load(config_path)
        except ValueError as error:
            assert "root must be an object" in str(error)
        else:
            raise AssertionError(f"falsey nonmapping config root accepted: {content!r}")


def test_score_selected_spans_rejects_malformed_source_spans() -> None:
    malformed = (
        SelectedSpan("", 0, 1),
        SelectedSpan("doc", True, 1),
        SelectedSpan("doc", 2, 2),
    )

    for selected in malformed:
        try:
            score_selected_spans((selected,), (), answerable=False)
        except ValueError:
            pass
        else:
            raise AssertionError(f"malformed selected span accepted: {selected}")


def test_config_rejects_boolean_integer_settings(tmp_path: Path) -> None:
    config_path = tmp_path / "benchmark.yaml"
    config_path.write_text("context_budget_tokens: true\n", encoding="utf-8")

    try:
        EvidenceBenchmarkConfig.load(config_path)
    except ValueError as error:
        assert "integer" in str(error)
    else:
        raise AssertionError("boolean integer setting was accepted")
