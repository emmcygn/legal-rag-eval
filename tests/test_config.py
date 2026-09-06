"""Tests for BenchmarkConfig."""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from legal_rag_eval.config import BenchmarkConfig, ConfigError, env_override


class TestBenchmarkConfigDefaults:
    """Test default configuration values."""

    def test_default_strategies(self) -> None:
        config = BenchmarkConfig()
        assert config.strategies == [
            "lexichunk",
            "lexichunk_contextual",
            "rcts_512",
            "rcts_1024",
            "sentence_split",
            "fixed_size",
        ]

    def test_default_embedding_models(self) -> None:
        config = BenchmarkConfig()
        assert config.embedding_models == ["all-MiniLM-L6-v2"]

    def test_default_k_values(self) -> None:
        config = BenchmarkConfig()
        assert config.k_values == [1, 3, 5, 10]

    def test_default_voyage_disabled(self) -> None:
        config = BenchmarkConfig()
        assert config.enable_voyage is False

    def test_default_output_formats(self) -> None:
        config = BenchmarkConfig()
        assert config.output_formats == ["cli", "json"]

    def test_default_gold_dir(self) -> None:
        config = BenchmarkConfig()
        assert config.gold_dir == "gold"

    def test_default_relevance_min_overlap_chars(self) -> None:
        config = BenchmarkConfig()
        assert config.relevance_min_overlap_chars == 100

    def test_default_bootstrap_resamples(self) -> None:
        config = BenchmarkConfig()
        assert config.bootstrap_resamples == 10000

    def test_default_seed(self) -> None:
        config = BenchmarkConfig()
        assert config.seed == 42

    def test_default_rcts_chunk_params(self) -> None:
        config = BenchmarkConfig()
        assert config.rcts_512_chunk_size == 512
        assert config.rcts_512_chunk_overlap == 50
        assert config.rcts_1024_chunk_size == 1024
        assert config.rcts_1024_chunk_overlap == 100

    def test_default_fixed_chunk_overlap_is_zero(self) -> None:
        config = BenchmarkConfig()
        assert config.fixed_chunk_overlap == 0


class TestBenchmarkConfigValidation:
    """Test validation logic."""

    def test_valid_config_passes(self) -> None:
        config = BenchmarkConfig()
        config.validate()  # Should not raise

    def test_unknown_strategy_raises(self) -> None:
        config = BenchmarkConfig(strategies=["lexichunk", "unknown_strategy"])
        with pytest.raises(ConfigError, match="Unknown strategies"):
            config.validate()

    def test_empty_strategies_raises(self) -> None:
        config = BenchmarkConfig(strategies=[])
        with pytest.raises(ConfigError, match="At least one strategy"):
            config.validate()

    def test_unknown_embedding_model_raises(self) -> None:
        config = BenchmarkConfig(embedding_models=["nonexistent-model"])
        with pytest.raises(ConfigError, match="Unknown embedding models"):
            config.validate()

    def test_voyage_without_api_key_raises(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            config = BenchmarkConfig(enable_voyage=True)
            os.environ.pop("VOYAGE_API_KEY", None)
            with pytest.raises(ConfigError, match="VOYAGE_API_KEY"):
                config.validate()

    def test_voyage_with_api_key_passes(self) -> None:
        with patch.dict(os.environ, {"VOYAGE_API_KEY": "test-key"}):
            config = BenchmarkConfig(enable_voyage=True)
            config.validate()

    def test_negative_k_value_raises(self) -> None:
        config = BenchmarkConfig(k_values=[1, -1, 5])
        with pytest.raises(ConfigError, match="k_values must be >= 1"):
            config.validate()

    def test_empty_k_values_raises(self) -> None:
        config = BenchmarkConfig(k_values=[])
        with pytest.raises(ConfigError, match="k_values must not be empty"):
            config.validate()

    def test_bad_significance_level_raises(self) -> None:
        config = BenchmarkConfig(significance_level=0.0)
        with pytest.raises(ConfigError, match="significance_level"):
            config.validate()

    def test_nonzero_fixed_chunk_overlap_raises(self) -> None:
        """FixedSizeStrategy implements no overlap, so any non-zero value is invalid --
        not just one that happens to exceed the chunk size."""
        config = BenchmarkConfig(fixed_chunk_size=100, fixed_chunk_overlap=1)
        with pytest.raises(ConfigError, match="fixed_chunk_overlap must be 0"):
            config.validate()

    def test_unknown_output_format_raises(self) -> None:
        config = BenchmarkConfig(output_formats=["cli", "pdf"])
        with pytest.raises(ConfigError, match="Unknown output formats"):
            config.validate()

    def test_rcts_512_overlap_gte_size_raises(self) -> None:
        config = BenchmarkConfig(rcts_512_chunk_size=200, rcts_512_chunk_overlap=200)
        with pytest.raises(ConfigError, match="rcts_512_chunk_overlap"):
            config.validate()

    def test_rcts_1024_overlap_gte_size_raises(self) -> None:
        config = BenchmarkConfig(rcts_1024_chunk_size=300, rcts_1024_chunk_overlap=400)
        with pytest.raises(ConfigError, match="rcts_1024_chunk_overlap"):
            config.validate()

    def test_rcts_chunk_size_too_small_raises(self) -> None:
        config = BenchmarkConfig(rcts_512_chunk_size=10, rcts_512_chunk_overlap=0)
        with pytest.raises(ConfigError, match="rcts_512_chunk_size"):
            config.validate()

    def test_relevance_min_overlap_chars_below_one_raises(self) -> None:
        config = BenchmarkConfig(relevance_min_overlap_chars=0)
        with pytest.raises(ConfigError, match="relevance_min_overlap_chars"):
            config.validate()

    def test_bootstrap_resamples_below_100_raises(self) -> None:
        config = BenchmarkConfig(bootstrap_resamples=99)
        with pytest.raises(ConfigError, match="bootstrap_resamples"):
            config.validate()

    def test_bootstrap_resamples_at_minimum_passes(self) -> None:
        config = BenchmarkConfig(bootstrap_resamples=100)
        config.validate()  # Should not raise

    def test_rcts_strategy_no_longer_valid(self) -> None:
        """'rcts' was replaced by explicitly-sized rcts_512/rcts_1024; the unsized
        legacy label must be rejected as a configured strategy even though
        StrategyName.RCTS still exists for deserializing old result JSON."""
        config = BenchmarkConfig(strategies=["rcts"])
        with pytest.raises(ConfigError, match="Unknown strategies"):
            config.validate()


class TestBenchmarkConfigYAML:
    """Test YAML loading."""

    def test_load_from_yaml(self, tmp_path: Path) -> None:
        yaml_content = {
            "strategies": ["lexichunk", "fixed_size"],
            "top_k": 5,
        }
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(yaml.dump(yaml_content))

        config = BenchmarkConfig.from_yaml(config_file)
        assert config.strategies == ["lexichunk", "fixed_size"]
        assert config.top_k == 5
        assert config.k_values == [1, 3, 5, 10]

    def test_missing_yaml_file_raises(self) -> None:
        with pytest.raises(ConfigError, match="Config file not found"):
            BenchmarkConfig.from_yaml("/nonexistent/path.yaml")

    def test_unknown_yaml_key_raises(self, tmp_path: Path) -> None:
        yaml_content = {"strategies": ["lexichunk"], "bogus_key": True}
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(yaml.dump(yaml_content))

        with pytest.raises(ConfigError, match="Unknown config keys"):
            BenchmarkConfig.from_yaml(config_file)

    def test_empty_yaml_returns_defaults(self, tmp_path: Path) -> None:
        config_file = tmp_path / "empty.yaml"
        config_file.write_text("")

        config = BenchmarkConfig.from_yaml(config_file)
        assert config.strategies == BenchmarkConfig().strategies


class TestBenchmarkConfigEnv:
    """Test environment variable overrides."""

    def test_voyage_auto_detected(self) -> None:
        with patch.dict(os.environ, {"VOYAGE_API_KEY": "test-key"}, clear=False):
            config = BenchmarkConfig.from_env()
            assert config.enable_voyage is True
            assert "voyage-law-2" in config.embedding_models

    def test_strategies_from_env(self) -> None:
        with patch.dict(
            os.environ,
            {"LEGAL_RAG_EVAL_STRATEGIES": "lexichunk,fixed_size"},
            clear=False,
        ):
            config = BenchmarkConfig.from_env()
            assert config.strategies == ["lexichunk", "fixed_size"]

    def test_top_k_from_env(self) -> None:
        with patch.dict(os.environ, {"LEGAL_RAG_EVAL_TOP_K": "20"}, clear=False):
            config = BenchmarkConfig.from_env()
            assert config.top_k == 20

    def test_gold_dir_from_env(self) -> None:
        with patch.dict(os.environ, {"LEGAL_RAG_EVAL_GOLD_DIR": "/tmp/gold"}, clear=False):
            config = BenchmarkConfig.from_env()
            assert config.gold_dir == "/tmp/gold"

    def test_relevance_min_overlap_chars_from_env(self) -> None:
        env = {"LEGAL_RAG_EVAL_RELEVANCE_MIN_OVERLAP_CHARS": "50"}
        with patch.dict(os.environ, env, clear=False):
            config = BenchmarkConfig.from_env()
            assert config.relevance_min_overlap_chars == 50

    def test_bootstrap_resamples_from_env(self) -> None:
        with patch.dict(os.environ, {"LEGAL_RAG_EVAL_BOOTSTRAP_RESAMPLES": "500"}, clear=False):
            config = BenchmarkConfig.from_env()
            assert config.bootstrap_resamples == 500

    def test_seed_from_env(self) -> None:
        with patch.dict(os.environ, {"LEGAL_RAG_EVAL_SEED": "7"}, clear=False):
            config = BenchmarkConfig.from_env()
            assert config.seed == 7


class TestDeprecatedEnvPrefix:
    """The pre-rename SCAFFOLDER_ prefix still works, and says it is going away."""

    def test_deprecated_prefix_still_read(self) -> None:
        with patch.dict(os.environ, {"SCAFFOLDER_TOP_K": "17"}, clear=True):
            with pytest.warns(DeprecationWarning, match="SCAFFOLDER_TOP_K"):
                config = BenchmarkConfig.from_env()
            assert config.top_k == 17

    def test_deprecated_prefix_applied_by_load(self) -> None:
        with patch.dict(os.environ, {"SCAFFOLDER_SEED": "11"}, clear=True):
            with pytest.warns(DeprecationWarning):
                config = BenchmarkConfig.load()
            assert config.seed == 11

    def test_current_prefix_wins_over_deprecated(self) -> None:
        env = {"LEGAL_RAG_EVAL_TOP_K": "5", "SCAFFOLDER_TOP_K": "17"}
        with patch.dict(os.environ, env, clear=True):
            config = BenchmarkConfig.from_env()
            assert config.top_k == 5

    def test_no_warning_without_deprecated_names(self) -> None:
        env = {"LEGAL_RAG_EVAL_TOP_K": "5"}
        with patch.dict(os.environ, env, clear=True), warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            assert BenchmarkConfig.from_env().top_k == 5

    def test_env_override_returns_none_when_unset(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            assert env_override("TOP_K") is None


class TestBenchmarkConfigResolvePaths:
    """Test path resolution."""

    def test_relative_paths_resolved(self) -> None:
        config = BenchmarkConfig()
        root = Path("/home/user/project")
        config.resolve_paths(root)
        assert config.fixture_dir == str(root / "src/legal_rag_eval/fixtures/documents")
        assert config.output_dir == str(root / "results")
        assert config.gold_dir == str(root / "gold")

    def test_absolute_gold_dir_unchanged(self) -> None:
        abs_path = str(Path("/absolute/gold").resolve())
        config = BenchmarkConfig(gold_dir=abs_path)
        root = Path("/home/user/project")
        config.resolve_paths(root)
        assert config.gold_dir == abs_path

    def test_absolute_paths_unchanged(self) -> None:
        abs_path = str(Path("/absolute/path").resolve())
        config = BenchmarkConfig(output_dir=abs_path)
        root = Path("/home/user/project")
        config.resolve_paths(root)
        assert config.output_dir == abs_path


class TestBenchmarkConfigToDict:
    """Test serialization."""

    def test_to_dict_roundtrip(self) -> None:
        config = BenchmarkConfig()
        d = config.to_dict()
        assert isinstance(d, dict)
        assert d["strategies"] == config.strategies
        assert d["top_k"] == config.top_k
