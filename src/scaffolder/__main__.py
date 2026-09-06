"""CLI entry point: python -m scaffolder benchmark."""

from __future__ import annotations

import argparse
import datetime
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
from rich.console import Console

from scaffolder.chunking import ChunkingPipeline, get_all_strategies
from scaffolder.fixtures import FixtureManager
from scaffolder.metrics.structural import compute_structural_metrics
from scaffolder.models import BenchmarkResult, EmbeddingModelName
from scaffolder.reporting.cli import render_benchmark
from scaffolder.reporting.json_export import export_json

logger = logging.getLogger(__name__)

SEED = 42


def _make_console() -> Console:
    """Create a Rich Console that works on Windows (UTF-8 safe)."""
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    return Console(force_terminal=True)


def _pin_seeds(seed: int = SEED) -> None:
    """Pin random seeds for reproducible benchmark results."""
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def main() -> None:
    """Parse CLI arguments and dispatch to the appropriate command."""
    parser = argparse.ArgumentParser(description="sdk-scaffolder benchmark")
    parser.add_argument(
        "command",
        choices=["benchmark", "benchmark-embed", "report"],
        help="Command to run",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory for output files",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Export results to JSON",
    )
    parser.add_argument(
        "--no-embed",
        action="store_true",
        help="Skip embedding evaluation (structural only)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=f"Random seed pinned before the run (default: {SEED})",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(name)s %(levelname)s: %(message)s",
    )

    if args.command == "benchmark":
        run_structural_benchmark(args)
    elif args.command == "benchmark-embed":
        run_retrieval_benchmark(args)
    elif args.command == "report":
        run_report(args)


def run_structural_benchmark(args: argparse.Namespace) -> None:
    """Run structural-only benchmark (no embeddings)."""
    seed = getattr(args, "seed", SEED)
    _pin_seeds(seed)
    fm = FixtureManager()
    documents = fm.load_all()

    strategies = get_all_strategies()
    pipeline = ChunkingPipeline(strategies)
    strategy_results = pipeline.run(documents)

    result = BenchmarkResult(
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        strategies=[sr.strategy for sr in strategy_results],
        documents=[d.id for d in documents],
        strategy_results=strategy_results,
        seed=seed,
    )

    # Compute structural metrics for each (strategy, document) pair
    for sr in strategy_results:
        for cs in sr.chunk_sets:
            doc = fm.get_by_id(cs.document_id)
            sm = compute_structural_metrics(cs, doc)
            result.structural_metrics.append(sm)

    console = _make_console()
    render_benchmark(result, console=console)

    # Export JSON if requested
    if args.json:
        json_path = args.output_dir / "structural_benchmark.json"
        export_json(result, json_path)
        print(f"\nJSON exported to: {json_path}")


def _get_available_models() -> list[EmbeddingModelName]:
    """Return list of available embedding models.

    Always includes local models. Includes Voyage only if API key is set.
    """
    models: list[EmbeddingModelName] = [EmbeddingModelName.MINILM]

    voyage_key = os.environ.get("VOYAGE_API_KEY")
    if voyage_key:
        logger.info("VOYAGE_API_KEY found, including voyage-law-2.")
        models.append(EmbeddingModelName.VOYAGE_LAW_2)
    else:
        logger.info("VOYAGE_API_KEY not set, skipping voyage-law-2.")

    return models


def run_retrieval_benchmark(args: argparse.Namespace) -> None:
    """Run full benchmark including embeddings and retrieval."""
    seed = getattr(args, "seed", SEED)
    _pin_seeds(seed)
    from scaffolder.embedding import EmbeddingPipeline
    from scaffolder.metrics.retrieval import compute_retrieval_metrics
    from scaffolder.metrics.statistical import compute_all_significance
    from scaffolder.retrieval import RetrievalSimulator, build_all_indices
    from scaffolder.retrieval.simulator import load_queries_from_yaml

    fm = FixtureManager()
    documents = fm.load_all()

    # Phase 1: Chunking
    strategies = get_all_strategies()
    pipeline = ChunkingPipeline(strategies)
    strategy_results = pipeline.run(documents)

    models = _get_available_models()
    result = BenchmarkResult(
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        strategies=[sr.strategy for sr in strategy_results],
        documents=[d.id for d in documents],
        models=models,
        strategy_results=strategy_results,
        seed=seed,
    )

    # Phase 2: Structural metrics
    for sr in strategy_results:
        for cs in sr.chunk_sets:
            doc = fm.get_by_id(cs.document_id)
            sm = compute_structural_metrics(cs, doc)
            result.structural_metrics.append(sm)

    # Phase 3: Embedding + Indexing
    emb_pipeline = EmbeddingPipeline(models=models)
    registry = build_all_indices(strategy_results, emb_pipeline, models)

    # Phase 4: Retrieval simulation
    queries = load_queries_from_yaml()

    simulator = RetrievalSimulator(
        index_registry=registry,
        embedding_pipeline=emb_pipeline,
    )
    retrieval_results = simulator.run(
        queries=queries,
        strategies=[sr.strategy for sr in strategy_results],
        models=models,
        k=10,
    )

    # Phase 5: Retrieval metrics
    for rr in retrieval_results:
        rm = compute_retrieval_metrics(rr)
        result.retrieval_metrics.append(rm)

    # Phase 6: Statistical significance
    result.significance_results = list(compute_all_significance(result.retrieval_metrics))

    console = _make_console()
    render_benchmark(result, console=console)

    if args.json:
        json_path = args.output_dir / "full_benchmark.json"
        export_json(result, json_path)
        print(f"\nJSON exported to: {json_path}")


def run_report(args: argparse.Namespace) -> None:
    """Generate HTML report from existing benchmark results or a fresh run."""
    from scaffolder.reporting.html import render_html_report
    from scaffolder.reporting.json_export import load_json

    output_dir = args.output_dir

    # Try to load existing results
    json_path = output_dir / "full_benchmark.json"
    if not json_path.exists():
        json_path = output_dir / "structural_benchmark.json"

    if json_path.exists():
        logger.info("Loading results from %s", json_path)
        data = load_json(json_path)
        result = _reconstruct_benchmark_result(data)
    else:
        logger.info("No existing results found, running fresh structural benchmark.")
        fm = FixtureManager()
        documents = fm.load_all()
        strategies = get_all_strategies()
        pipeline = ChunkingPipeline(strategies)
        strategy_results = pipeline.run(documents)

        result = BenchmarkResult(
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            strategies=[sr.strategy for sr in strategy_results],
            documents=[d.id for d in documents],
            strategy_results=strategy_results,
        )
        for sr in strategy_results:
            for cs in sr.chunk_sets:
                doc = fm.get_by_id(cs.document_id)
                sm = compute_structural_metrics(cs, doc)
                result.structural_metrics.append(sm)

    html_path = output_dir / "report.html"
    render_html_report(result, html_path)
    print(f"HTML report generated: {html_path}")


def _reconstruct_benchmark_result(data: dict) -> BenchmarkResult:  # type: ignore[type-arg]
    """Reconstruct a BenchmarkResult from JSON data."""
    import contextlib

    from scaffolder.models import (
        RetrievalMetrics,
        SignificanceResult,
        StrategyName,
        StructuralMetrics,
    )

    result = BenchmarkResult(
        timestamp=data.get("timestamp", ""),
    )

    for s in data.get("strategies", []):
        with contextlib.suppress(ValueError):
            result.strategies.append(StrategyName(s))

    result.documents = data.get("documents", [])

    for m in data.get("models", []):
        with contextlib.suppress(ValueError):
            result.models.append(EmbeddingModelName(m))

    for sm_data in data.get("structural_metrics", []):
        with contextlib.suppress(KeyError, ValueError):
            result.structural_metrics.append(
                StructuralMetrics(
                    strategy=StrategyName(sm_data["strategy"]),
                    document_id=sm_data["document_id"],
                    clause_fragmentation_rate=sm_data["clause_fragmentation_rate"],
                    definition_preservation_rate=sm_data["definition_preservation_rate"],
                    cross_ref_resolution_rate=sm_data["cross_ref_resolution_rate"],
                    hierarchy_depth_retained=sm_data["hierarchy_depth_retained"],
                    chunk_size_cv=sm_data["chunk_size_cv"],
                    chunk_count=sm_data["chunk_count"],
                    avg_chunk_chars=sm_data["avg_chunk_chars"],
                )
            )

    for rm_data in data.get("retrieval_metrics", []):
        with contextlib.suppress(KeyError, ValueError):
            result.retrieval_metrics.append(
                RetrievalMetrics(
                    query_id=rm_data["query_id"],
                    strategy=StrategyName(rm_data["strategy"]),
                    embedding_model=EmbeddingModelName(rm_data["embedding_model"]),
                    precision_at_1=rm_data["precision_at_1"],
                    precision_at_3=rm_data["precision_at_3"],
                    precision_at_5=rm_data["precision_at_5"],
                    precision_at_10=rm_data["precision_at_10"],
                    recall_at_1=rm_data["recall_at_1"],
                    recall_at_3=rm_data["recall_at_3"],
                    recall_at_5=rm_data["recall_at_5"],
                    recall_at_10=rm_data["recall_at_10"],
                    mrr=rm_data["mrr"],
                    ndcg_at_10=rm_data["ndcg_at_10"],
                    drm_hit=rm_data["drm_hit"],
                )
            )

    for sr_data in data.get("significance_results", []):
        with contextlib.suppress(KeyError, ValueError):
            result.significance_results.append(
                SignificanceResult(
                    metric_name=sr_data["metric_name"],
                    strategy_a=StrategyName(sr_data["strategy_a"]),
                    strategy_b=StrategyName(sr_data["strategy_b"]),
                    mean_a=sr_data["mean_a"],
                    mean_b=sr_data["mean_b"],
                    improvement_pct=sr_data["improvement_pct"],
                    t_statistic=sr_data["t_statistic"],
                    p_value=sr_data["p_value"],
                    significant=sr_data["significant"],
                    effect_size=sr_data["effect_size"],
                    n_queries=sr_data["n_queries"],
                )
            )

    return result


if __name__ == "__main__":
    main()
