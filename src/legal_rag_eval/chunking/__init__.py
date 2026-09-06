"""Chunking pipeline and strategy wrappers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from legal_rag_eval.chunking.pipeline import ChunkingPipeline
from legal_rag_eval.chunking.strategies import (
    FixedSizeStrategy,
    LexiChunkContextualStrategy,
    LexiChunkStrategy,
    RCTSStrategy,
    SentenceSplitStrategy,
)
from legal_rag_eval.models import ChunkingStrategy, StrategyName

if TYPE_CHECKING:
    from collections.abc import Sequence

    from legal_rag_eval.config import BenchmarkConfig

# Every entry is (class, default constructor kwargs). The kwargs are part of the
# strategy's identity and are exported with the results, because a baseline whose
# chunk size is not stated cannot be compared to anything.
_STRATEGY_REGISTRY: dict[StrategyName, tuple[type, dict[str, Any]]] = {
    StrategyName.LEXICHUNK: (LexiChunkStrategy, {}),
    StrategyName.LEXICHUNK_CONTEXTUAL: (LexiChunkContextualStrategy, {}),
    StrategyName.RCTS_512: (
        RCTSStrategy,
        {"chunk_size": 512, "chunk_overlap": 50, "name": StrategyName.RCTS_512},
    ),
    StrategyName.RCTS_1024: (
        RCTSStrategy,
        {"chunk_size": 1024, "chunk_overlap": 100, "name": StrategyName.RCTS_1024},
    ),
    StrategyName.SENTENCE_SPLIT: (SentenceSplitStrategy, {"min_chunk_chars": 100}),
    StrategyName.FIXED_SIZE: (FixedSizeStrategy, {"chunk_size": 512}),
}

DEFAULT_STRATEGIES: tuple[StrategyName, ...] = (
    StrategyName.LEXICHUNK,
    StrategyName.LEXICHUNK_CONTEXTUAL,
    StrategyName.RCTS_512,
    StrategyName.RCTS_1024,
    StrategyName.SENTENCE_SPLIT,
    StrategyName.FIXED_SIZE,
)


def strategy_params(name: StrategyName) -> dict[str, Any]:
    """Return the constructor parameters a named strategy runs with.

    Used by the reporters so the published tables can state the configuration that
    actually ran instead of a hand-typed description of it.
    """
    if name not in _STRATEGY_REGISTRY:
        msg = f"Unknown strategy: {name}. Available: {sorted(s.value for s in _STRATEGY_REGISTRY)}"
        raise ValueError(msg)
    params = dict(_STRATEGY_REGISTRY[name][1])
    params.pop("name", None)
    return params


def get_strategy(name: StrategyName, **kwargs: Any) -> ChunkingStrategy:
    """Instantiate a chunking strategy by name, overriding its defaults with kwargs."""
    if name not in _STRATEGY_REGISTRY:
        msg = f"Unknown strategy: {name}. Available: {sorted(s.value for s in _STRATEGY_REGISTRY)}"
        raise ValueError(msg)
    cls, defaults = _STRATEGY_REGISTRY[name]
    merged = {**defaults, **kwargs}
    strategy: ChunkingStrategy = cls(**merged)
    return strategy


def get_all_strategies(
    names: Sequence[StrategyName] | None = None,
    **kwargs: Any,
) -> list[ChunkingStrategy]:
    """Instantiate the requested strategies (default: every strategy in DEFAULT_STRATEGIES)."""
    selected = list(names) if names is not None else list(DEFAULT_STRATEGIES)
    return [get_strategy(name, **kwargs) for name in selected]


def strategies_from_config(config: BenchmarkConfig) -> list[ChunkingStrategy]:
    """Instantiate exactly the strategies a BenchmarkConfig asks for, with its parameters."""
    strategies: list[ChunkingStrategy] = []
    for raw in config.strategies:
        name = StrategyName(raw)
        overrides: dict[str, Any] = {}
        if name is StrategyName.RCTS_512:
            overrides = {
                "chunk_size": config.rcts_512_chunk_size,
                "chunk_overlap": config.rcts_512_chunk_overlap,
            }
        elif name is StrategyName.RCTS_1024:
            overrides = {
                "chunk_size": config.rcts_1024_chunk_size,
                "chunk_overlap": config.rcts_1024_chunk_overlap,
            }
        elif name is StrategyName.FIXED_SIZE:
            overrides = {"chunk_size": config.fixed_chunk_size}
        elif name is StrategyName.SENTENCE_SPLIT:
            overrides = {"min_chunk_chars": config.sentence_min_chunk_size}
        strategies.append(get_strategy(name, **overrides))
    return strategies


def describe_strategies(strategies: Sequence[ChunkingStrategy]) -> dict[str, dict[str, Any]]:
    """Return {strategy name: constructor parameters} for the given live strategies."""
    return {s.name.value: strategy_params(s.name) for s in strategies}


__all__ = [
    "DEFAULT_STRATEGIES",
    "ChunkingPipeline",
    "FixedSizeStrategy",
    "LexiChunkContextualStrategy",
    "LexiChunkStrategy",
    "RCTSStrategy",
    "SentenceSplitStrategy",
    "describe_strategies",
    "get_all_strategies",
    "get_strategy",
    "strategies_from_config",
    "strategy_params",
]
