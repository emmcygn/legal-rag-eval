"""Compare one strategy across two benchmark runs produced by different LexiChunk builds.

The rest of the harness compares strategies *within* a run. That cannot answer the
question a chunker's maintainer actually has — "did this parser change help?" — because
the baselines are byte-identical across runs and the only thing that moved is LexiChunk
itself. This script puts the same strategy's numbers from two result files side by side,
so a parser change shows up as a number rather than as an impression.

Usage:
    python scripts/compare_builds.py \\
        --before results/lexichunk_baseline/full_benchmark.json \\
        --after  results/lexichunk_fixed/full_benchmark.json

It prints a Markdown table on stdout. Nothing is written to the README; paste the output
if you want it published, so a table in the README is always something a human chose to
put there.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Structural metrics worth showing, with the direction that counts as better.
_STRUCTURAL_ROWS: list[tuple[str, str]] = [
    ("localization_rate", "Chunks located in the source text"),
    ("clause_fragmentation_rate", "Leaf-clause fragmentation (lower is better)"),
    ("top_level_over_merge_rate", "Top-level over-merge (lower is better)"),
    ("heading_recall", "Heading attachment recall"),
    ("definition_attachment_recall", "Definition attachment recall"),
    ("xref_target_recall", "Cross-reference target recall"),
    ("avg_chunk_chars", "Mean chunk length (chars)"),
]

_RETRIEVAL_ROWS: list[tuple[str, str]] = [
    ("precision_at_1", "P@1"),
    ("precision_at_5", "P@5"),
    ("recall_at_5", "R@5"),
    ("mrr", "MRR"),
    ("ndcg_at_10", "NDCG@10"),
]


def _mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def _structural(data: dict[str, Any], strategy: str, key: str) -> float | None:
    return _mean(
        [
            row[key]
            for row in data.get("structural_metrics", [])
            if row.get("strategy") == strategy and row.get(key) is not None
        ]
    )


def _retrieval(data: dict[str, Any], strategy: str, model: str, key: str) -> float | None:
    return _mean(
        [
            row[key]
            for row in data.get("retrieval_metrics", [])
            if row.get("strategy") == strategy
            and row.get("embedding_model") == model
            and row.get(key) is not None
        ]
    )


def _fmt(value: float | None, *, as_chars: bool = False) -> str:
    if value is None:
        return "n/a"
    return f"{value:.0f}" if as_chars else f"{value:.3f}"


def _label(data: dict[str, Any], fallback: str) -> str:
    version = data.get("lexichunk_version") or "?"
    commit = data.get("lexichunk_commit")
    suffix = f" `{commit}`" if commit else ""
    return f"{fallback} — LexiChunk `{version}`{suffix}"


def _models(before: dict[str, Any], after: dict[str, Any]) -> list[str]:
    shared = {m for m in before.get("models", [])} & {m for m in after.get("models", [])}
    return sorted(shared)


def render(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    strategy: str,
    control: str,
    before_name: str,
    after_name: str,
) -> str:
    """Render the side-by-side Markdown table."""
    n_queries = len({row.get("query_id") for row in after.get("retrieval_metrics", [])})
    n_documents = len(after.get("documents", []))

    lines = [
        f"| Measure (strategy `{strategy}`) | {_label(before, before_name)} "
        f"| {_label(after, after_name)} |",
        "|---|---:|---:|",
    ]
    for key, label in _STRUCTURAL_ROWS:
        as_chars = key == "avg_chunk_chars"
        lines.append(
            f"| {label} (n = {n_documents} documents) "
            f"| {_fmt(_structural(before, strategy, key), as_chars=as_chars)} "
            f"| {_fmt(_structural(after, strategy, key), as_chars=as_chars)} |"
        )
    for model in _models(before, after):
        for key, label in _RETRIEVAL_ROWS:
            lines.append(
                f"| {label}, {model} (n = {n_queries} queries) "
                f"| {_fmt(_retrieval(before, strategy, model, key))} "
                f"| {_fmt(_retrieval(after, strategy, model, key))} |"
            )
    # The control row is the point of the table: if a baseline that never saw the
    # dependency change also moved, the difference above is not about the dependency.
    for model in _models(before, after):
        lines.append(
            f"| _control_: `{control}` MRR, {model} "
            f"| {_fmt(_retrieval(before, control, model, 'mrr'))} "
            f"| {_fmt(_retrieval(after, control, model, 'mrr'))} |"
        )
    return "\n".join(lines)


def main() -> int:
    """Print the two-build comparison table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True, help="Earlier run's results JSON")
    parser.add_argument("--after", type=Path, required=True, help="Later run's results JSON")
    parser.add_argument("--before-name", default="before", help="Column label for --before")
    parser.add_argument("--after-name", default="after", help="Column label for --after")
    parser.add_argument("--strategy", default="lexichunk", help="Strategy to compare")
    parser.add_argument(
        "--control",
        default="rcts_1024",
        help="Strategy that does not depend on LexiChunk, shown as a control row",
    )
    args = parser.parse_args()

    for path in (args.before, args.after):
        if not path.exists():
            print(f"No results at {path}.", file=sys.stderr)
            return 1

    before = json.loads(args.before.read_text(encoding="utf-8"))
    after = json.loads(args.after.read_text(encoding="utf-8"))
    print(
        render(
            before,
            after,
            strategy=args.strategy,
            control=args.control,
            before_name=args.before_name,
            after_name=args.after_name,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
