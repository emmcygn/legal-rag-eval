"""CLI argument parsing and the offline ``--smoke`` path.

Nothing here touches the network: the smoke path swaps in the synthetic corpora
from ``scaffolder.evals.smoke``, so the whole pipeline (sampling, mapping,
scoring, bootstrap, export, rendering) is exercised end to end offline.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scaffolder.evals.__main__ import build_parser, run
from scaffolder.evals.cuad import (
    StrategySpec,
    evaluate_strategy,
    evaluate_structure,
    run_cuad,
)
from scaffolder.evals.cuad import (
    render_markdown as render_cuad,
)
from scaffolder.evals.label_map import DEFAULT_MAP_PATH
from scaffolder.evals.ledgar import (
    map_items,
    predict_lexichunk,
    run_ledgar,
)
from scaffolder.evals.ledgar import (
    render_markdown as render_ledgar,
)
from scaffolder.evals.smoke import smoke_cuad_contracts, smoke_ledgar_split


class TestArgumentParsing:
    def test_ledgar_defaults(self) -> None:
        args = build_parser().parse_args(["ledgar"])
        assert args.command == "ledgar"
        assert args.sample == 5000
        assert args.seed == 0
        assert args.split == "test"
        assert not args.smoke

    def test_cuad_defaults(self) -> None:
        args = build_parser().parse_args(["cuad"])
        assert args.command == "cuad"
        assert args.contracts == 100
        assert args.seed == 0

    def test_documented_invocations_parse(self) -> None:
        parser = build_parser()
        ledgar = parser.parse_args(["ledgar", "--sample", "5000", "--seed", "0"])
        assert (ledgar.sample, ledgar.seed) == (5000, 0)
        cuad = parser.parse_args(["cuad", "--contracts", "100", "--seed", "0"])
        assert (cuad.contracts, cuad.seed) == (100, 0)

    def test_output_dir_is_a_path(self) -> None:
        args = build_parser().parse_args(["cuad", "--output-dir", "somewhere"])
        assert isinstance(args.output_dir, Path)

    def test_flags(self) -> None:
        args = build_parser().parse_args(["ledgar", "--smoke", "--skip-tfidf", "--quiet"])
        assert args.smoke and args.skip_tfidf and args.quiet

    def test_numeric_options(self) -> None:
        args = build_parser().parse_args(
            ["ledgar", "--bootstrap", "10", "--relative-position", "0.9", "--train-sample", "5"]
        )
        assert args.bootstrap == 10
        assert args.relative_position == pytest.approx(0.9)
        assert args.train_sample == 5

    def test_min_clauses_option(self) -> None:
        assert build_parser().parse_args(["cuad", "--min-clauses", "3"]).min_clauses == 3

    def test_a_command_is_required(self) -> None:
        with pytest.raises(SystemExit):
            build_parser().parse_args([])

    def test_unknown_command_rejected(self) -> None:
        with pytest.raises(SystemExit):
            build_parser().parse_args(["nonsense"])

    def test_non_integer_sample_rejected(self) -> None:
        with pytest.raises(SystemExit):
            build_parser().parse_args(["ledgar", "--sample", "lots"])


class TestCuadSmoke:
    def test_cli_writes_json_and_markdown(self, tmp_path: Path) -> None:
        assert run(["cuad", "--smoke", "--quiet", "--output-dir", str(tmp_path)]) == 0
        payload = json.loads((tmp_path / "cuad_smoke.json").read_text(encoding="utf-8"))
        assert payload["smoke"] is True
        assert payload["evaluation"] == "cuad"
        assert payload["strategies"]
        assert (tmp_path / "cuad_smoke.md").read_text(encoding="utf-8").startswith("> **Smoke")

    def test_every_strategy_is_reported(self) -> None:
        result = run_cuad(contracts=0, n_boot=20, provider=smoke_cuad_contracts)
        names = {s.name for s in result.strategies}
        assert {"lexichunk-512tok", "rcts-512", "rcts-1024", "sentence-window-3"} <= names

    def test_results_are_deterministic(self) -> None:
        first = run_cuad(contracts=0, seed=3, n_boot=20, provider=smoke_cuad_contracts)
        second = run_cuad(contracts=0, seed=3, n_boot=20, provider=smoke_cuad_contracts)
        assert [s.containment.point for s in first.strategies] == [
            s.containment.point for s in second.strategies
        ]

    def test_markdown_renders(self) -> None:
        markdown = render_cuad(run_cuad(contracts=0, n_boot=20, provider=smoke_cuad_contracts))
        assert "Containment" in markdown
        assert "structure recall" in markdown

    def test_containment_is_a_proportion(self) -> None:
        for strategy in run_cuad(contracts=0, n_boot=20, provider=smoke_cuad_contracts).strategies:
            assert 0.0 <= strategy.containment.point <= 1.0
            assert 0.0 <= strategy.grid_expected_containment <= 1.0


class TestCuadMechanics:
    def test_a_single_whole_document_chunk_contains_everything(self) -> None:
        _, _, contracts = smoke_cuad_contracts()
        whole = StrategySpec("whole-doc", lambda t: [__import__(
            "scaffolder.evals.chunkers", fromlist=["Span"]
        ).Span(0, len(t))])
        result = evaluate_strategy(whole, contracts, n_boot=20)
        # Containment is trivially perfect, and the length-matched grid says so:
        # the lift over an equally long fixed-stride splitter is tiny.
        assert result.containment.point == 1.0
        assert result.lift < 0.15

    def test_a_crashing_strategy_is_recorded_not_raised(self) -> None:
        _, _, contracts = smoke_cuad_contracts()

        def explode(_text: str) -> list:  # type: ignore[type-arg]
            msg = "synthetic chunker failure"
            raise RuntimeError(msg)

        failures: list = []
        result = evaluate_strategy(
            StrategySpec("broken", explode), contracts, n_boot=10, failures=failures
        )
        assert result.failures == len(contracts)
        assert failures and failures[0].kind == "RuntimeError"
        assert "synthetic chunker failure" in failures[0].message

    def test_structure_recall_separates_structured_from_flat(self) -> None:
        _, _, contracts = smoke_cuad_contracts()
        structured = evaluate_structure(contracts, min_clauses=5)
        assert structured.contracts == 2
        # SMOKE_A has five numbered sections, SMOKE_B has none.
        assert structured.parsed == 1
        assert structured.fell_back == 1
        assert structured.exceptions == 0


class TestLedgarSmoke:
    def test_cli_writes_json_and_markdown(self, tmp_path: Path) -> None:
        if not DEFAULT_MAP_PATH.is_file():
            pytest.skip(f"{DEFAULT_MAP_PATH} not present")
        assert run(["ledgar", "--smoke", "--quiet", "--output-dir", str(tmp_path)]) == 0
        payload = json.loads((tmp_path / "ledgar_smoke.json").read_text(encoding="utf-8"))
        assert payload["smoke"] is True
        assert payload["evaluation"] == "ledgar"
        assert payload["systems"]

    def test_pipeline_runs_offline(self) -> None:
        if not DEFAULT_MAP_PATH.is_file():
            pytest.skip(f"{DEFAULT_MAP_PATH} not present")
        result = run_ledgar(
            sample=0, n_boot=20, skip_tfidf=True, provider=smoke_ledgar_split
        )
        assert result.dataset_id == "synthetic-smoke"
        assert result.evaluated > 0
        assert 0.0 <= result.systems[0].accuracy.point <= 1.0
        assert result.systems[0].calibration is not None

    def test_out_of_scope_rows_are_excluded_and_counted(self) -> None:
        if not DEFAULT_MAP_PATH.is_file():
            pytest.skip(f"{DEFAULT_MAP_PATH} not present")
        from scaffolder.evals.label_map import load_label_map

        split = smoke_ledgar_split("test")
        items, dropped = map_items(split.texts, split.labels, load_label_map(DEFAULT_MAP_PATH))
        assert len(items) + dropped == len(split.texts)
        assert all(item.gold != "unknown" for item in items)

    def test_markdown_renders(self) -> None:
        if not DEFAULT_MAP_PATH.is_file():
            pytest.skip(f"{DEFAULT_MAP_PATH} not present")
        markdown = render_ledgar(
            run_ledgar(sample=0, n_boot=20, skip_tfidf=True, provider=smoke_ledgar_split)
        )
        assert "Accuracy (95% CI)" in markdown
        assert "calibration" in markdown.lower()

    def test_deterministic(self) -> None:
        if not DEFAULT_MAP_PATH.is_file():
            pytest.skip(f"{DEFAULT_MAP_PATH} not present")
        runs = [
            run_ledgar(sample=0, seed=5, n_boot=20, skip_tfidf=True, provider=smoke_ledgar_split)
            for _ in range(2)
        ]
        # Wall-clock timings are the only field allowed to differ between runs.
        payloads = []
        for result in runs:
            payload = result.as_dict()
            for system in payload["systems"]:
                system.pop("seconds")
            payloads.append(payload)
        assert payloads[0] == payloads[1]


class TestClassifierAdapter:
    def test_predictions_and_confidences_align(self) -> None:
        from scaffolder.evals.ledgar import LedgarItem

        items = [
            LedgarItem(
                "This Agreement is governed by the laws of Delaware.",
                "Governing Laws",
                "governing_law",
            ),
            LedgarItem(
                "Each party shall keep all information confidential.",
                "Confidentiality",
                "confidentiality",
            ),
        ]
        preds, confs, seconds = predict_lexichunk(items)
        assert len(preds) == len(confs) == 2
        assert all(0.0 <= c <= 1.0 for c in confs)
        assert seconds >= 0.0
