"""
Tests for span generator configuration loading.
"""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from eval_harness.stubs.span_generator.config import (
    DEFAULT_BATCH_EXPORT,
    DEFAULT_ENDPOINT,
    DEFAULT_LIMIT,
    DEFAULT_PROJECT_NAME,
    DEFAULT_SEED,
    DEFAULT_STUB_EMBEDDING_MODEL,
    DEFAULT_STUB_MODEL_ID,
    GeneratorConfig,
    _resolve_phoenix_endpoint,
    load_generator_config,
)


class TestGeneratorConfig:
    """Tests for GeneratorConfig dataclass and loading."""

    def test_load_config_with_all_fields(self, tmp_path: Path) -> None:
        """Test loading config with all fields specified."""
        config_content = {
            "datasets": {"legal_rag_bench": {"path": "data"}},
            "metrics": {"text_fidelity": {"threshold": 0.95}},
            "models": {"judge_model": "gpt-4o"},
            "generator": {
                "phoenix_endpoint": "http://custom:6006",
                "project_name": "custom-project",
                "default_limit": 50,
                "batch_export": False,
                "seed": 123,
                "stub_model_id": "custom-model",
                "stub_embedding_model": "custom-embedder",
            },
        }

        config_file = tmp_path / "eval_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_content, f)

        config = load_generator_config(config_file)

        assert config.phoenix_endpoint == "http://custom:6006"
        assert config.project_name == "custom-project"
        assert config.default_limit == 50
        assert config.batch_export is False
        assert config.seed == 123
        assert config.stub_model_id == "custom-model"
        assert config.stub_embedding_model == "custom-embedder"

    def test_load_config_with_defaults(self, tmp_path: Path) -> None:
        """Test loading config with missing generator section uses defaults."""
        config_content = {
            "datasets": {"legal_rag_bench": {"path": "data"}},
            "metrics": {"text_fidelity": {"threshold": 0.95}},
            "models": {"judge_model": "gpt-4o"},
        }

        config_file = tmp_path / "eval_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_content, f)

        config = load_generator_config(config_file)

        assert config.phoenix_endpoint == DEFAULT_ENDPOINT
        assert config.project_name == DEFAULT_PROJECT_NAME
        assert config.default_limit == DEFAULT_LIMIT
        assert config.batch_export == DEFAULT_BATCH_EXPORT
        assert config.seed == DEFAULT_SEED
        assert config.stub_model_id == DEFAULT_STUB_MODEL_ID
        assert config.stub_embedding_model == DEFAULT_STUB_EMBEDDING_MODEL

    def test_env_var_override(self, tmp_path: Path, monkeypatch) -> None:
        """Test environment variable override for phoenix_endpoint."""
        config_content = {
            "datasets": {"legal_rag_bench": {"path": "data"}},
            "metrics": {"text_fidelity": {"threshold": 0.95}},
            "models": {"judge_model": "gpt-4o"},
            "generator": {"phoenix_endpoint": "http://yaml:6006"},
        }

        config_file = tmp_path / "eval_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_content, f)

        # Environment variable should override YAML
        monkeypatch.setenv("PHOENIX_ENDPOINT", "http://env:6006")

        config = load_generator_config(config_file)
        assert config.phoenix_endpoint == "http://env:6006"

    def test_config_file_not_found(self) -> None:
        """Test FileNotFoundError when config file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            load_generator_config(Path("/nonexistent/path.yaml"))

    def test_resolve_phoenix_endpoint_with_env(self, monkeypatch) -> None:
        """Test _resolve_phoenix_endpoint with environment variable."""
        monkeypatch.setenv("PHOENIX_ENDPOINT", "http://env:6006")

        result = _resolve_phoenix_endpoint(None)
        assert result == "http://env:6006"

    def test_resolve_phoenix_endpoint_with_yaml(self) -> None:
        """Test _resolve_phoenix_endpoint with YAML value."""
        result = _resolve_phoenix_endpoint("http://yaml:6006")
        assert result == "http://yaml:6006"

    def test_resolve_phoenix_endpoint_default(self, monkeypatch) -> None:
        """Test _resolve_phoenix_endpoint with no value uses default."""
        # Ensure env var is not set
        monkeypatch.delenv("PHOENIX_ENDPOINT", raising=False)

        result = _resolve_phoenix_endpoint(None)
        assert result == DEFAULT_ENDPOINT

    def test_config_dataclass_is_frozen(self) -> None:
        """Test that GeneratorConfig is frozen (immutable)."""
        config = GeneratorConfig(
            phoenix_endpoint="http://localhost:6006",
            project_name="test",
            default_limit=10,
            batch_export=True,
            seed=42,
            stub_model_id="model",
            stub_embedding_model="embedder",
        )

        with pytest.raises(Exception):  # FrozenInstanceError
            config.default_limit = 20
