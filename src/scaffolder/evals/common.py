"""Shared helpers for the external (public-dataset) evaluations.

This module is deliberately self-contained: it does not import from
``scaffolder.metrics``, ``scaffolder.queries`` or ``scaffolder.reporting`` so
that the external evaluations stay decoupled from the internal benchmark
harness.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

T = TypeVar("T")

#: Where evaluation artefacts are written.
DEFAULT_RESULTS_DIR = Path("results") / "external"

#: Default number of bootstrap resamples used for confidence intervals.
DEFAULT_BOOTSTRAP = 2000

#: Default two-sided confidence level.
DEFAULT_CONFIDENCE = 0.95


@dataclass(frozen=True, slots=True)
class Interval:
    """A point estimate with a bootstrap confidence interval."""

    point: float
    low: float
    high: float
    n: int

    def as_dict(self) -> dict[str, float | int]:
        """Return a JSON-friendly mapping."""
        return {"point": self.point, "low": self.low, "high": self.high, "n": self.n}

    def format(self, pct: bool = True, places: int = 1) -> str:
        """Render as ``point [low, high]``, optionally as percentages."""
        scale = 100.0 if pct else 1.0
        suffix = "%" if pct else ""
        return (
            f"{self.point * scale:.{places}f}{suffix} "
            f"[{self.low * scale:.{places}f}, {self.high * scale:.{places}f}]"
        )


def mean(values: Sequence[float]) -> float:
    """Arithmetic mean; 0.0 for an empty sequence."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def bootstrap_ci(
    items: Sequence[T],
    statistic: Callable[[Sequence[T]], float],
    *,
    n_boot: int = DEFAULT_BOOTSTRAP,
    seed: int = 0,
    confidence: float = DEFAULT_CONFIDENCE,
) -> Interval:
    """Percentile bootstrap CI for ``statistic`` computed over ``items``.

    Resampling is over *items* (contracts, provisions), which is the unit of
    independence in these evaluations. The RNG is seeded explicitly so results
    are byte-for-byte reproducible.

    Args:
        items: The observations to resample with replacement.
        statistic: Maps a resample to a scalar.
        n_boot: Number of bootstrap resamples.
        seed: RNG seed.
        confidence: Two-sided confidence level (e.g. 0.95).

    Returns:
        An :class:`Interval`. With fewer than two items the interval collapses
        onto the point estimate rather than reporting a spurious spread.
    """
    n = len(items)
    if n == 0:
        return Interval(0.0, 0.0, 0.0, 0)
    point = statistic(items)
    if n < 2 or n_boot <= 0:
        return Interval(point, point, point, n)

    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(n_boot):
        resample = [items[rng.randrange(n)] for _ in range(n)]
        draws.append(statistic(resample))
    draws.sort()
    tail = (1.0 - confidence) / 2.0
    low = draws[max(0, int(tail * n_boot))]
    high = draws[min(n_boot - 1, int((1.0 - tail) * n_boot))]
    return Interval(point, low, high, n)


def proportion_ci(
    flags: Sequence[bool],
    *,
    n_boot: int = DEFAULT_BOOTSTRAP,
    seed: int = 0,
    confidence: float = DEFAULT_CONFIDENCE,
) -> Interval:
    """Bootstrap CI for the proportion of ``True`` values in ``flags``."""
    return bootstrap_ci(
        flags,
        lambda xs: mean([1.0 if x else 0.0 for x in xs]),
        n_boot=n_boot,
        seed=seed,
        confidence=confidence,
    )


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    """Render a GitHub-flavoured markdown table.

    Columns are padded to a uniform width so the raw markdown stays readable.
    """
    cells = [[str(h) for h in headers]] + [[str(c) for c in row] for row in rows]
    widths = [max(len(row[i]) for row in cells) for i in range(len(headers))]
    lines = [
        "| " + " | ".join(h.ljust(w) for h, w in zip(cells[0], widths, strict=True)) + " |",
        "| " + " | ".join("-" * w for w in widths) + " |",
    ]
    lines.extend(
        "| " + " | ".join(c.ljust(w) for c, w in zip(row, widths, strict=True)) + " |"
        for row in cells[1:]
    )
    return "\n".join(lines)


def write_json(payload: dict[str, Any], path: str | Path, *, indent: int = 2) -> Path:
    """Write ``payload`` as UTF-8 JSON, creating parent directories."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=indent, sort_keys=True, default=str)
        handle.write("\n")
    return out


def cache_dir() -> Path:
    """Return the Hugging Face cache directory used by these evaluations.

    Honours ``SCAFFOLDER_EVAL_CACHE`` then ``HF_HOME``, else falls back to
    ``.cache/hf`` under the current working directory. The directory is created
    on demand and is gitignored.
    """
    for env in ("SCAFFOLDER_EVAL_CACHE", "HF_HOME"):
        value = os.environ.get(env)
        if value:
            path = Path(value)
            path.mkdir(parents=True, exist_ok=True)
            return path
    path = Path(".cache") / "hf"
    path.mkdir(parents=True, exist_ok=True)
    return path


def deterministic_sample(population: Sequence[T], k: int, seed: int) -> list[T]:
    """Take a seeded sample of ``k`` items, preserving the original order.

    Returns the whole population (in order) when ``k`` is non-positive or at
    least the population size, so ``--sample 0`` means "everything".
    """
    n = len(population)
    if k <= 0 or k >= n:
        return list(population)
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(n), k))
    return [population[i] for i in indices]
