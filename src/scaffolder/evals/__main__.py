"""CLI for the external evaluations.

python -m scaffolder.evals ledgar --sample 5000 --seed 0
python -m scaffolder.evals cuad --contracts 100 --seed 0
python -m scaffolder.evals ledgar --smoke      # no download, synthetic data
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from scaffolder.evals import cuad as cuad_eval
from scaffolder.evals import ledgar as ledgar_eval
from scaffolder.evals.common import DEFAULT_BOOTSTRAP, DEFAULT_RESULTS_DIR, write_json
from scaffolder.evals.label_map import DEFAULT_MAP_PATH


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for ``python -m scaffolder.evals``."""
    parser = argparse.ArgumentParser(
        prog="python -m scaffolder.evals",
        description="External evaluations of LexiChunk on public legal datasets.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name in ("ledgar", "cuad"):
        sub = subparsers.add_parser(name)
        sub.add_argument("--seed", type=int, default=0, help="Sampling and bootstrap seed.")
        sub.add_argument(
            "--output-dir",
            type=Path,
            default=DEFAULT_RESULTS_DIR,
            help=f"Directory for JSON and markdown output (default: {DEFAULT_RESULTS_DIR}).",
        )
        sub.add_argument(
            "--bootstrap",
            type=int,
            default=DEFAULT_BOOTSTRAP,
            help="Bootstrap resamples per confidence interval.",
        )
        sub.add_argument(
            "--smoke",
            action="store_true",
            help="Run on tiny synthetic inputs with no download. Numbers are meaningless.",
        )
        sub.add_argument("--quiet", action="store_true", help="Do not print the markdown report.")

    ledgar = subparsers.choices["ledgar"]
    ledgar.add_argument(
        "--sample", type=int, default=5000, help="Test provisions to score (0 = whole split)."
    )
    ledgar.add_argument("--split", default="test", help="LEDGAR split to score.")
    ledgar.add_argument(
        "--label-map", type=Path, default=DEFAULT_MAP_PATH, help="LEDGAR -> LexiChunk mapping."
    )
    ledgar.add_argument(
        "--train-sample",
        type=int,
        default=20000,
        help="Cap on train rows used to fit the TF-IDF baseline.",
    )
    ledgar.add_argument(
        "--skip-tfidf", action="store_true", help="Skip the supervised TF-IDF baseline."
    )
    ledgar.add_argument(
        "--relative-position",
        type=float,
        default=ledgar_eval.NEUTRAL_POSITION,
        help="Document position hint passed to the classifier (0.0-1.0).",
    )

    cuad = subparsers.choices["cuad"]
    cuad.add_argument(
        "--contracts", type=int, default=100, help="Contracts to sample (0 = all of them)."
    )
    cuad.add_argument("--split", default=cuad_eval.DEFAULT_SPLIT, help="CUAD split(s) to draw on.")
    cuad.add_argument(
        "--min-clauses",
        type=int,
        default=cuad_eval.DEFAULT_MIN_CLAUSES,
        help="Top-level clauses required for a contract to count as parsed.",
    )
    return parser


def run(argv: list[str] | None = None) -> int:
    """Parse ``argv``, run the requested evaluation and write its artefacts."""
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    suffix = "_smoke" if args.smoke else ""

    payload: dict[str, Any]
    if args.command == "ledgar":
        from scaffolder.evals.smoke import smoke_ledgar_split

        result = ledgar_eval.run_ledgar(
            sample=0 if args.smoke else args.sample,
            seed=args.seed,
            split=args.split,
            label_map_path=args.label_map,
            n_boot=50 if args.smoke else args.bootstrap,
            relative_position=args.relative_position,
            train_sample=args.train_sample,
            skip_tfidf=args.skip_tfidf or args.smoke,
            provider=smoke_ledgar_split if args.smoke else None,
        )
        payload = result.as_dict()
        markdown = ledgar_eval.render_markdown(result)
    else:
        from scaffolder.evals.smoke import smoke_cuad_contracts

        result_cuad = cuad_eval.run_cuad(
            contracts=0 if args.smoke else args.contracts,
            seed=args.seed,
            split=args.split,
            n_boot=50 if args.smoke else args.bootstrap,
            min_clauses=args.min_clauses,
            provider=smoke_cuad_contracts if args.smoke else None,
        )
        payload = result_cuad.as_dict()
        markdown = cuad_eval.render_markdown(result_cuad)

    if args.smoke:
        payload["smoke"] = True
        markdown = (
            "> **Smoke run on synthetic data — these numbers are meaningless.**\n\n" + markdown
        )

    json_path = write_json(payload, output_dir / f"{args.command}{suffix}.json")
    md_path = output_dir / f"{args.command}{suffix}.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(markdown, encoding="utf-8")

    if not args.quiet:
        _write_stdout(markdown)
    _write_stdout(f"\nWrote {json_path} and {md_path}\n")
    return 0


def _write_stdout(text: str) -> None:
    """Write to stdout without exploding on a non-UTF-8 Windows console."""
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    sys.stdout.write(text.encode(encoding, errors="replace").decode(encoding))


def main() -> None:
    """Console entry point."""
    raise SystemExit(run())


if __name__ == "__main__":
    main()
