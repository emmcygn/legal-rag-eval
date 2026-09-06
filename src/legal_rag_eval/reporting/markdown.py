"""Generate the README results section from a `BenchmarkResult`.

The README's results tables used to be hand-typed and drifted from the code. This module
renders them straight from a results JSON instead, so every number in the README traces back
to `BenchmarkResult`. Nothing here hardcodes a metric value — only column headers, labels, and
the "how to read this" prose template are fixed; every number is read from the result.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING

from legal_rag_eval.reporting.cli import (
    ComparisonLike,
    aggregate_gold_structural_metrics,
    aggregate_retrieval_metrics,
)

if TYPE_CHECKING:
    from legal_rag_eval.models import BenchmarkResult, GoldStructuralMetrics, StrategyName

_RCTS_1024 = "rcts_1024"

#: Sentinels delimiting the generated block in README.md. Exported so callers (and the
#: staleness check in ``scripts/update_readme.py``) use the same strings the writer does.
START_MARKER = "<!-- BEGIN GENERATED RESULTS -->"
END_MARKER = "<!-- END GENERATED RESULTS -->"


# -- Formatting helpers (see module docstring: 3dp for rates, 0dp for counts, "n/a" for None) --


def _fmt_rate(value: float) -> str:
    return f"{value:.3f}"


def _fmt_optional_rate(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _fmt_pct(value: float) -> str:
    return f"{value * 100:.3f}%"


def _fmt_count(value: float | int) -> str:
    return f"{value:.0f}"


def _fmt_delta(value: float) -> str:
    """Always shows the sign, per the formatting rules — a delta with a hidden sign invites
    misreading a regression as an improvement."""
    if math.isnan(value):
        return "n/a"
    return f"{value:+.3f}"


def _fmt_ci(low: float, high: float) -> str:
    if math.isnan(low) or math.isnan(high):
        return "n/a"
    return f"[{low:+.3f}, {high:+.3f}]"


def _fmt_p(value: float) -> str:
    if math.isnan(value):
        return "n/a"
    return f"{value:.3f}"


def _fmt_lodo(low: float, high: float) -> str:
    if math.isnan(low) or math.isnan(high):
        return "n/a"
    return f"[{low:+.3f}, {high:+.3f}]"


def _fmt_effect_size(value: float) -> str:
    if math.isinf(value):
        return "+inf" if value > 0 else "-inf"
    if math.isnan(value):
        return "n/a"
    return f"{value:.3f}"


def _row(*cells: str | int) -> str:
    """Render one Markdown table row from cell values."""
    return "| " + " | ".join(str(c) for c in cells) + " |"


# -- Section builders -----------------------------------------------------------


def _provenance_line(result: BenchmarkResult) -> str:
    """Timestamp, LexiChunk build, embedding model(s), seed, n documents/queries, params."""
    models = ", ".join(m.value for m in result.models) if result.models else "n/a"
    version = result.lexichunk_version or "unknown"
    commit = result.lexichunk_commit or "unknown"
    seed = result.seed if result.seed is not None else "unknown"
    n_documents = len(result.documents)
    n_queries = len({m.query_id for m in result.retrieval_metrics})

    params = result.config.get("strategy_parameters") if isinstance(result.config, dict) else None
    params_str = json.dumps(params, sort_keys=True) if params else "n/a"

    return (
        f"_Generated {result.timestamp} — LexiChunk `{version}` (commit `{commit}`); "
        f"embedding model(s): {models}; seed `{seed}`; {n_documents} documents; "
        f"{n_queries} queries. Strategy parameters: `{params_str}`._"
    )


def _structural_section(result: BenchmarkResult) -> str:
    """Structural quality table, macro-averaged over documents (n stated)."""
    if not result.structural_metrics:
        return "### Structural quality\n\n_No structural metrics available for this run._"

    aggregated = aggregate_gold_structural_metrics(result.structural_metrics)
    n_documents = len({m.document_id for m in result.structural_metrics})

    lines = [
        "### Structural quality",
        "",
        f"Macro-averaged over documents (n = {n_documents}). Ground truth is hand-checked "
        "span annotations in `gold/`; see the methodology note below for what these metrics "
        "do not measure.",
        "",
        "| Strategy | Located | Leaf Frag ↓ | Over-merge ↓ | Sub-clause Grp | Head R | "
        "Head P | Def Attach ↑ | XRef R | XRef P | Size CV | Avg Chars | Chunks |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for a in aggregated:
        lines.append(
            _row(
                a.strategy.value,
                _fmt_pct(a.localization_rate),
                _fmt_rate(a.clause_fragmentation_rate),
                _fmt_rate(a.top_level_over_merge_rate),
                _fmt_rate(a.sub_clause_grouping_rate),
                _fmt_optional_rate(a.heading_recall),
                _fmt_optional_rate(a.heading_precision),
                _fmt_rate(a.definition_attachment_recall),
                _fmt_optional_rate(a.xref_target_recall),
                _fmt_optional_rate(a.xref_target_precision),
                _fmt_rate(a.chunk_size_cv),
                _fmt_count(a.avg_chunk_chars),
                _fmt_count(a.chunk_count),
            )
        )
    return "\n".join(lines)


def _retrieval_section(result: BenchmarkResult) -> str:
    """One retrieval table per embedding model, mean over queries (n stated)."""
    if not result.retrieval_metrics:
        return "### Retrieval quality\n\n_No retrieval metrics available for this run._"

    blocks = ["### Retrieval quality"]
    models = sorted({m.embedding_model.value for m in result.retrieval_metrics})
    for model in models:
        model_metrics = [m for m in result.retrieval_metrics if m.embedding_model.value == model]
        n_queries = len({m.query_id for m in model_metrics})
        aggregated = aggregate_retrieval_metrics(model_metrics)

        lines = [
            "",
            f"**{model}** (mean over queries, n = {n_queries})",
            "",
            "| Strategy | P@1 | P@3 | P@5 | P@10 | R@1 | R@3 | R@5 | R@10 | MRR | "
            "NDCG@10 | DRM rate |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for a in aggregated:
            lines.append(
                _row(
                    a.strategy.value,
                    _fmt_rate(a.precision_at_1),
                    _fmt_rate(a.precision_at_3),
                    _fmt_rate(a.precision_at_5),
                    _fmt_rate(a.precision_at_10),
                    _fmt_rate(a.recall_at_1),
                    _fmt_rate(a.recall_at_3),
                    _fmt_rate(a.recall_at_5),
                    _fmt_rate(a.recall_at_10),
                    _fmt_rate(a.mrr),
                    _fmt_rate(a.ndcg_at_10),
                    _fmt_rate(a.drm_rate),
                )
            )
        blocks.append("\n".join(lines))
    return "\n".join(blocks)


def _comparisons_section(result: BenchmarkResult) -> str:
    """Statistical comparisons table: Δ + bootstrap CI, Holm p, Wilcoxon p, effect size, LODO."""
    rows = [c for c in result.comparisons if isinstance(c, ComparisonLike)]
    if not rows:
        return (
            "### Strategy comparisons\n\n"
            "_No `ComparisonResult` data available for this run "
            "(bootstrap CI / Holm-adjusted p / Wilcoxon / LODO)._"
        )

    blocks = [
        "### Strategy comparisons",
        "",
        "The absolute delta and its bootstrap 95% CI are the headline figures — never a "
        "relative percent change. A row is significant only if it survives Holm correction "
        "across the whole family of tests below (every metric, pair and embedding model in "
        f"this run). `{_RCTS_1024}` is the **size-matched control**: it uses the same target "
        "chunk size as LexiChunk, so it isolates what the chunking *strategy* contributes "
        "from what chunk *size* alone would contribute. The LODO range is the delta's spread "
        "across the five leave-one-document-out refits; a range straddling zero means one "
        "document carries the result.",
    ]

    # One table per embedding model. Without the split, the same metric and strategy pair
    # appears once per model with nothing in the row to tell the two apart.
    for model in sorted({c.embedding_model for c in rows}):
        model_rows = sorted(
            (c for c in rows if c.embedding_model == model),
            key=lambda c: (c.metric_name, c.strategy_a.value, c.strategy_b.value),
        )
        sizes = sorted({c.n for c in model_rows})
        n_label = str(sizes[0]) if len(sizes) == 1 else "varies"
        lines = [
            "",
            f"**{model}** ({len(model_rows)} tests, n = {n_label} queries each)",
            "",
            "| Metric | A | B | n | Δ | 95% CI | p (Holm) | p (Wilcoxon) | Effect size (d) | "
            "LODO range |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for c in model_rows:
            b_label = f"{c.strategy_b.value}"
            if c.strategy_b.value == _RCTS_1024:
                b_label = f"**{b_label} (size-matched control)**"
            holm_cell = _fmt_p(c.p_value_holm) + (" **" if c.significant_holm else "")
            lines.append(
                _row(
                    c.metric_name,
                    c.strategy_a.value,
                    b_label,
                    c.n,
                    _fmt_delta(c.delta),
                    _fmt_ci(c.ci_low, c.ci_high),
                    holm_cell,
                    _fmt_p(c.p_value_wilcoxon),
                    _fmt_effect_size(c.cohens_d),
                    _fmt_lodo(c.lodo_min_delta, c.lodo_max_delta),
                )
            )
        blocks.append("\n".join(lines))
    return "\n".join(blocks)


def _mean_chunk_lengths(
    structural_metrics: list[GoldStructuralMetrics],
) -> list[tuple[StrategyName, float]]:
    aggregated = aggregate_gold_structural_metrics(structural_metrics)
    by_strategy = sorted(aggregated, key=lambda a: a.strategy.value)
    return [(a.strategy, a.avg_chunk_chars) for a in by_strategy]


def _how_to_read_section(result: BenchmarkResult) -> str:
    """Generated (not hardcoded) guidance: mean chunk length per strategy, for size-matching."""
    if not result.structural_metrics:
        return (
            "### How to read this\n\n"
            "_Mean chunk length by strategy is not available (no structural metrics in this "
            "run), so it is not possible to tell from this report alone whether any "
            "retrieval comparison is size-matched._"
        )

    lengths = _mean_chunk_lengths(result.structural_metrics)
    lines = [
        "### How to read this",
        "",
        "Mean chunk length by strategy — a retrieval comparison is only size-matched when "
        "these are close; otherwise an apparent win may just be a chunk-size effect:",
        "",
    ]
    for strategy, avg_chars in lengths:
        lines.append(f"- `{strategy.value}`: {_fmt_count(avg_chars)} chars/chunk")
    return "\n".join(lines)


def render_results_markdown(result: BenchmarkResult) -> str:
    """Render the full generated results section as GitHub-flavoured Markdown.

    Order: provenance line, structural table, retrieval table(s), comparisons table, then a
    generated "how to read this" block. No number in the output is hand-typed — everything
    comes from ``result``.
    """
    sections = [
        _provenance_line(result),
        "",
        _structural_section(result),
        "",
        _retrieval_section(result),
        "",
        _comparisons_section(result),
        "",
        _how_to_read_section(result),
    ]
    return "\n".join(sections).strip() + "\n"


def write_results_markdown(result: BenchmarkResult, path: str | Path) -> Path:
    """Render `render_results_markdown` and write it to ``path``, creating parent dirs."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_results_markdown(result), encoding="utf-8")
    return output_path


def update_readme(
    readme_path: str | Path,
    result: BenchmarkResult,
    *,
    start_marker: str = START_MARKER,
    end_marker: str = END_MARKER,
) -> Path:
    """Replace everything between ``start_marker`` and ``end_marker`` in-place.

    Creates nothing if the markers are missing — raises `ValueError` naming them instead, so
    a README that has not yet been updated to carry the markers fails loudly rather than
    silently doing nothing (or, worse, appending).
    """
    path = Path(readme_path)
    text = path.read_text(encoding="utf-8")

    start_idx = text.find(start_marker)
    end_idx = text.find(end_marker)
    if start_idx == -1 or end_idx == -1:
        msg = (
            f"Could not find both generated-results markers in {path}: "
            f"start_marker={start_marker!r} (found={start_idx != -1}), "
            f"end_marker={end_marker!r} (found={end_idx != -1})"
        )
        raise ValueError(msg)
    if end_idx < start_idx:
        msg = f"{end_marker!r} appears before {start_marker!r} in {path}"
        raise ValueError(msg)

    before = text[: start_idx + len(start_marker)]
    after = text[end_idx:]
    generated = render_results_markdown(result)
    new_text = f"{before}\n\n{generated}\n{after}"

    path.write_text(new_text, encoding="utf-8")
    return path
