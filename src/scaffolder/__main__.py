"""CLI entry point: python -m scaffolder benchmark | benchmark-embed | report.

Every knob the README documents is read from :class:`~scaffolder.config.BenchmarkConfig`
and the resolved configuration is written into the exported JSON. The previous version of
this file hardcoded the strategy list, ``k=10``, ``alpha=0.05`` and the model list, so the
documented ``scaffolder.yaml`` and ``SCAFFOLDER_*`` overrides were inert and
``bge-base-en-v1.5`` could not be selected at all.
"""

from __future__ import annotations

import argparse
import datetime
import logging
import os
import random
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from rich.console import Console

from scaffolder.chunking import ChunkingPipeline, describe_strategies, strategies_from_config
from scaffolder.config import BenchmarkConfig, ConfigError
from scaffolder.fixtures import FixtureManager
from scaffolder.gold import load_all_gold
from scaffolder.metrics.gold import compute_gold_structural_metrics
from scaffolder.metrics.structural import compute_legacy_structural_metrics
from scaffolder.models import BenchmarkResult, EmbeddingModelName
from scaffolder.reporting.cli import render_benchmark
from scaffolder.reporting.json_export import export_json

if TYPE_CHECKING:
    from scaffolder.models import StrategyName

logger = logging.getLogger(__name__)

SEED = 42


def _make_console() -> Console:
    """Create a Rich Console that works on Windows (UTF-8 safe)."""
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    return Console(force_terminal=True)


def _pin_seeds(seed: int = SEED) -> None:
    """Pin random seeds for the run.

    Nothing downstream is actually stochastic — the FAISS index is an exact flat
    inner-product search and CPU sentence-transformer inference is deterministic — so this
    is provenance, not variance control. Repeat runs at a fixed dependency set produce
    byte-identical metrics; the only meaningful uncertainty is sampling uncertainty over
    queries, which the statistics module reports as bootstrap intervals.
    """
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _lexichunk_provenance() -> tuple[str | None, str | None]:
    """Return (version, commit) of the installed lexichunk, best effort.

    Recorded on every result so a published number can be traced to the build that
    produced it — the audit found baseline metrics silently moving between runs because
    the dependency's HEAD had moved.
    """
    version: str | None = None
    # An editable or directory install records no VCS metadata, so allow the commit to be
    # supplied explicitly when comparing two local builds of the chunker.
    commit: str | None = os.environ.get("SCAFFOLDER_LEXICHUNK_COMMIT") or None
    try:
        import lexichunk

        version = str(getattr(lexichunk, "__version__", "") or "") or None
    except ImportError:
        return None, None
    try:
        import json
        from importlib.metadata import distribution

        dist = distribution("lexichunk")
        raw = dist.read_text("direct_url.json")
        if raw and commit is None:
            info = json.loads(raw)
            commit = info.get("vcs_info", {}).get("commit_id")
    except Exception:  # noqa: BLE001 - provenance is best effort, never fatal
        pass
    return version, commit


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=None, help="Path to a scaffolder.yaml")
    parser.add_argument(
        "--strategies",
        type=str,
        default=None,
        help="Comma-separated strategy names, overriding the config",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated embedding model names, overriding the config "
        "(e.g. all-MiniLM-L6-v2,bge-base-en-v1.5)",
    )
    parser.add_argument("--top-k", type=int, default=None, help="Retrieval depth (default 10)")
    parser.add_argument(
        "--min-overlap-chars",
        type=int,
        default=None,
        help="Characters a chunk must share with a gold relevant clause to count as relevant",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for output files")
    parser.add_argument("--json", action="store_true", help="Export results to JSON")
    parser.add_argument(
        "--legacy-metrics",
        action="store_true",
        help="Also compute the superseded LexiChunk-derived structural metrics",
    )
    parser.add_argument(
        "--no-embed",
        action="store_true",
        help="Skip embedding evaluation (structural only)",
    )
    parser.add_argument("--seed", type=int, default=None, help=f"Random seed (default: {SEED})")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")


def _config_from_args(args: argparse.Namespace) -> BenchmarkConfig:
    """Build the run configuration: defaults, then YAML, then env, then CLI flags."""
    config = BenchmarkConfig.load(args.config)
    if args.strategies:
        config.strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    if args.models:
        config.embedding_models = [m.strip() for m in args.models.split(",") if m.strip()]
    if args.top_k is not None:
        config.top_k = args.top_k
    if args.min_overlap_chars is not None:
        config.relevance_min_overlap_chars = args.min_overlap_chars
    if args.output_dir is not None:
        config.output_dir = str(args.output_dir)
    if args.seed is not None:
        config.seed = args.seed
    config.validate()
    return config


def main() -> None:
    """Parse CLI arguments and dispatch to the appropriate command."""
    parser = argparse.ArgumentParser(description="legal-rag-eval benchmark")
    parser.add_argument(
        "command",
        choices=["benchmark", "benchmark-embed", "report"],
        help="Command to run",
    )
    _add_common_arguments(parser)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(name)s %(levelname)s: %(message)s",
    )

    try:
        config = _config_from_args(args)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    if args.command == "benchmark":
        run_structural_benchmark(args, config)
    elif args.command == "benchmark-embed":
        run_retrieval_benchmark(args, config)
    elif args.command == "report":
        run_report(args, config)


def _new_result(
    config: BenchmarkConfig,
    strategies: list[StrategyName],
    documents: list[str],
    models: list[EmbeddingModelName] | None = None,
) -> BenchmarkResult:
    version, commit = _lexichunk_provenance()
    return BenchmarkResult(
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        strategies=strategies,
        documents=documents,
        models=list(models or []),
        seed=config.seed,
        config=config.to_dict(),
        lexichunk_version=version,
        lexichunk_commit=commit,
    )


def _compute_structural(
    result: BenchmarkResult,
    strategy_results: list,  # type: ignore[type-arg]
    fm: FixtureManager,
    config: BenchmarkConfig,
    *,
    legacy: bool,
) -> None:
    gold_by_document = load_all_gold(config.gold_dir)
    for sr in strategy_results:
        for cs in sr.chunk_sets:
            doc = fm.get_by_id(cs.document_id)
            gold = gold_by_document.get(cs.document_id)
            if gold is None:
                logger.warning(
                    "No gold annotation for %s; structural metrics skipped for it.",
                    cs.document_id,
                )
            else:
                result.structural_metrics.append(compute_gold_structural_metrics(cs, doc, gold))
            if legacy:
                result.legacy_structural_metrics.append(compute_legacy_structural_metrics(cs, doc))


def run_structural_benchmark(args: argparse.Namespace, config: BenchmarkConfig) -> None:
    """Run structural-only benchmark (no embeddings)."""
    _pin_seeds(config.seed)
    fm = FixtureManager()
    documents = fm.load_all()

    strategies = strategies_from_config(config)
    pipeline = ChunkingPipeline(strategies)
    strategy_results = pipeline.run(documents)

    result = _new_result(
        config, [sr.strategy for sr in strategy_results], [d.id for d in documents]
    )
    result.strategy_results = strategy_results
    result.config["strategy_parameters"] = describe_strategies(strategies)
    _compute_structural(result, strategy_results, fm, config, legacy=args.legacy_metrics)

    render_benchmark(result, console=_make_console())

    if args.json:
        json_path = Path(config.output_dir) / "structural_benchmark.json"
        export_json(result, json_path)
        print(f"\nJSON exported to: {json_path}")


def _resolve_models(config: BenchmarkConfig) -> list[EmbeddingModelName]:
    """Turn the configured model names into enum members, skipping unusable ones."""
    models: list[EmbeddingModelName] = []
    for raw in config.embedding_models:
        model = EmbeddingModelName(raw)
        if model is EmbeddingModelName.VOYAGE_LAW_2 and not os.environ.get("VOYAGE_API_KEY"):
            logger.info("VOYAGE_API_KEY not set, skipping voyage-law-2.")
            continue
        models.append(model)
    if (
        config.enable_voyage
        and os.environ.get("VOYAGE_API_KEY")
        and EmbeddingModelName.VOYAGE_LAW_2 not in models
    ):
        models.append(EmbeddingModelName.VOYAGE_LAW_2)
    if not models:
        msg = "No usable embedding models: check `embedding_models` in your configuration."
        raise ConfigError(msg)
    return models


def run_retrieval_benchmark(args: argparse.Namespace, config: BenchmarkConfig) -> None:
    """Run the full benchmark: chunking, embedding, retrieval, statistics."""
    _pin_seeds(config.seed)
    from scaffolder.embedding import EmbeddingPipeline
    from scaffolder.metrics.retrieval import compute_retrieval_metrics
    from scaffolder.metrics.statistical import compute_all_comparisons
    from scaffolder.retrieval import RetrievalSimulator, build_all_indices
    from scaffolder.retrieval.simulator import load_queries_from_yaml

    fm = FixtureManager()
    documents = fm.load_all()

    strategies = strategies_from_config(config)
    pipeline = ChunkingPipeline(strategies)
    strategy_results = pipeline.run(documents)

    models = _resolve_models(config)
    result = _new_result(
        config,
        [sr.strategy for sr in strategy_results],
        [d.id for d in documents],
        models,
    )
    result.strategy_results = strategy_results
    result.config["strategy_parameters"] = describe_strategies(strategies)
    _compute_structural(result, strategy_results, fm, config, legacy=args.legacy_metrics)

    emb_pipeline = EmbeddingPipeline(models=models)
    registry = build_all_indices(strategy_results, emb_pipeline, models)

    queries = load_queries_from_yaml(config.query_dir, config.gold_dir)
    simulator = RetrievalSimulator(index_registry=registry, embedding_pipeline=emb_pipeline)
    retrieval_results = simulator.run(
        queries=queries,
        strategies=[sr.strategy for sr in strategy_results],
        models=models,
        k=config.top_k,
        min_overlap_chars=config.relevance_min_overlap_chars,
    )

    for rr in retrieval_results:
        result.retrieval_metrics.append(
            compute_retrieval_metrics(rr, min_overlap_chars=config.relevance_min_overlap_chars)
        )

    query_documents = {q.id: q.document_ids[0] for q in queries if q.document_ids}
    comparisons = compute_all_comparisons(
        result.retrieval_metrics,
        alpha=config.significance_level,
        seed=config.seed,
        n_resamples=config.bootstrap_resamples,
        query_documents=query_documents,
    )
    result.comparisons = list(comparisons)
    result.significance_results = [c.to_significance_result() for c in comparisons]

    render_benchmark(result, console=_make_console())

    if args.json:
        json_path = Path(config.output_dir) / "full_benchmark.json"
        export_json(result, json_path)
        print(f"\nJSON exported to: {json_path}")


def run_report(args: argparse.Namespace, config: BenchmarkConfig) -> None:
    """Generate an HTML report from an existing results JSON."""
    from scaffolder.reporting.html import render_html_report
    from scaffolder.reporting.json_export import load_json, reconstruct_benchmark_result

    output_dir = Path(config.output_dir)
    json_path = output_dir / "full_benchmark.json"
    if not json_path.exists():
        json_path = output_dir / "structural_benchmark.json"

    if not json_path.exists():
        msg = (
            f"No results found in {output_dir}. Run `python -m scaffolder benchmark --json` "
            f"or `python -m scaffolder benchmark-embed --json` first."
        )
        print(msg, file=sys.stderr)
        raise SystemExit(1)

    logger.info("Loading results from %s", json_path)
    result = reconstruct_benchmark_result(load_json(json_path))

    html_path = output_dir / "report.html"
    render_html_report(result, html_path)
    print(f"HTML report generated: {html_path}")


if __name__ == "__main__":
    main()
