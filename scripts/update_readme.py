"""Regenerate the README's results section from an exported benchmark run.

The README's tables used to be typed by hand and drifted away from the code they described
— the documented baseline parameters were wrong, and the headline numbers could not be
traced to a run. This script is the only supported way to change them.

Usage:
    python scripts/update_readme.py                      # results/full_benchmark.json
    python scripts/update_readme.py --results path.json
    python scripts/update_readme.py --check              # exit 1 if the README is stale
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from legal_rag_eval.reporting.json_export import load_json, reconstruct_benchmark_result
from legal_rag_eval.reporting.markdown import (
    END_MARKER,
    START_MARKER,
    render_results_markdown,
    update_readme,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    """Render the results section and write it into the README."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=REPO_ROOT / "results" / "full_benchmark.json",
        help="Benchmark results JSON to render (default: results/full_benchmark.json)",
    )
    parser.add_argument(
        "--readme",
        type=Path,
        default=REPO_ROOT / "README.md",
        help="README to update",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Do not write; exit 1 if the README does not already match the results",
    )
    args = parser.parse_args()

    if not args.results.exists():
        print(
            f"No results at {args.results}. Run `make benchmark-embed` first.",
            file=sys.stderr,
        )
        return 1

    result = reconstruct_benchmark_result(load_json(args.results))
    rendered = render_results_markdown(result)

    if args.check:
        current = args.readme.read_text(encoding="utf-8")
        if START_MARKER not in current or END_MARKER not in current:
            print(f"README is missing {START_MARKER} / {END_MARKER}.", file=sys.stderr)
            return 1
        start = current.index(START_MARKER) + len(START_MARKER)
        end = current.index(END_MARKER)
        if current[start:end].strip() != rendered.strip():
            print(
                "README results section is stale. Run `make readme`.",
                file=sys.stderr,
            )
            return 1
        print("README results section is up to date.")
        return 0

    update_readme(args.readme, result)
    print(f"Updated {args.readme} from {args.results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
