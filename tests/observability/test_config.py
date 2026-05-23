"""Tests for Phoenix configuration loading."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
import yaml

from eval_harness.config import load_config
from eval_harness.observability.config import get_phoenix_config


class TestPhoenixConfiguration:
    """Test Phoenix configuration loading and precedence."""

    def test_default_phoenix_config_when_not_provided(self, tmp_path):
        """Test that Phoenix config has sensible defaults when not provided."""
        config_file = tmp_path / "eval_config.yaml"
        config_data = {
            "datasets": {"legal_rag_bench": {"path": "data/rag/legal_rag_bench"}},
            "metrics": {"ragas": {"enabled": True}},
            "models": {"openai": {"model": "gpt-4o-mini"}},
        }
        config_file.write_text(yaml.dump(config_data))

        config = load_config(config_file)
        phoenix_config = get_phoenix_config(config)

        assert phoenix_config["enabled"] is False
        assert phoenix_config["endpoint"] == "http://localhost:6006"
        assert phoenix_config["export_path"] == "/tmp/phoenix_traces"

    def test_phoenix_section_in_config(self, tmp_path):
        """Test that phoenix section can be loaded from config file."""
        config_file = tmp_path / "eval_config.yaml"
        config_data = {
            "datasets": {"legal_rag_bench": {"path": "data/rag/legal_rag_bench"}},
            "metrics": {"ragas": {"enabled": True}},
            "models": {"openai": {"model": "gpt-4o-mini"}},
            "phoenix": {
                "enabled": True,
                "endpoint": "http://localhost:6006",
                "export_path": "/tmp/phoenix_traces",
            },
        }
        config_file.write_text(yaml.dump(config_data))

        config = load_config(config_file)
        phoenix_config = get_phoenix_config(config)

        assert phoenix_config["enabled"] is True
        assert phoenix_config["endpoint"] == "http://localhost:6006"
        assert phoenix_config["export_path"] == "/tmp/phoenix_traces"

    def test_phoenix_endpoint_env_variable_expansion(self, tmp_path):
        """Test PHOENIX_ENDPOINT environment variable expansion."""
        config_file = tmp_path / "eval_config.yaml"
        config_data = {
            "datasets": {"legal_rag_bench": {"path": "data/rag/legal_rag_bench"}},
            "metrics": {"ragas": {"enabled": True}},
            "models": {"openai": {"model": "gpt-4o-mini"}},
            "phoenix": {"enabled": True, "endpoint": "${PHOENIX_ENDPOINT}"},
        }
        config_file.write_text(yaml.dump(config_data))

        with patch.dict(os.environ, {"PHOENIX_ENDPOINT": "http://phoenix.example.com"}):
            config = load_config(config_file)
            phoenix_config = get_phoenix_config(config)

        assert phoenix_config["endpoint"] == "http://phoenix.example.com"

    def test_phoenix_endpoint_env_variable_with_default(self, tmp_path):
        """Test PHOENIX_ENDPOINT with default value in YAML."""
        config_file = tmp_path / "eval_config.yaml"
        config_data = {
            "datasets": {"legal_rag_bench": {"path": "data/rag/legal_rag_bench"}},
            "metrics": {"ragas": {"enabled": True}},
            "models": {"openai": {"model": "gpt-4o-mini"}},
            "phoenix": {
                "enabled": True,
                "endpoint": "${PHOENIX_ENDPOINT:-http://localhost:6006}",
            },
        }
        config_file.write_text(yaml.dump(config_data))

        config = load_config(config_file)
        phoenix_config = get_phoenix_config(config)
        assert phoenix_config["endpoint"] == "http://localhost:6006"

    def test_phoenix_enabled_false_by_default(self, tmp_path):
        """Test that Phoenix is disabled by default when not specified."""
        config_file = tmp_path / "eval_config.yaml"
        config_data = {
            "datasets": {"legal_rag_bench": {"path": "data/rag/legal_rag_bench"}},
            "metrics": {"ragas": {"enabled": True}},
            "models": {"openai": {"model": "gpt-4o-mini"}},
        }
        config_file.write_text(yaml.dump(config_data))

        config = load_config(config_file)
        phoenix_config = get_phoenix_config(config)

        assert phoenix_config["enabled"] is False

    def test_phoenix_endpoint_validation_invalid_url(self, tmp_path):
        """Test that invalid endpoint URLs are handled correctly."""
        from eval_harness.observability.phoenix_adapter import PhoenixAdapter

        # The adapter should validate the endpoint
        with pytest.raises(ValueError, match="Invalid Phoenix endpoint URL"):
            PhoenixAdapter(endpoint="invalid-url")

    def test_phoenix_endpoint_validation_valid_urls(self, tmp_path):
        """Test that valid endpoint URLs are accepted."""
        from eval_harness.observability.phoenix_adapter import PhoenixAdapter

        # HTTP endpoint
        adapter1 = PhoenixAdapter(endpoint="http://localhost:6006", enabled=False)
        assert adapter1._endpoint == "http://localhost:6006"

        # HTTPS endpoint
        adapter2 = PhoenixAdapter(endpoint="https://phoenix.example.com", enabled=False)
        assert adapter2._endpoint == "https://phoenix.example.com"

    def test_cli_precedence_over_yaml(self, tmp_path):
        """Test that CLI flags take precedence over YAML config."""
        config_file = tmp_path / "eval_config.yaml"
        config_data = {
            "datasets": {"legal_rag_bench": {"path": "data/rag/legal_rag_bench"}},
            "metrics": {"ragas": {"enabled": True}},
            "models": {"openai": {"model": "gpt-4o-mini"}},
            "phoenix": {
                "enabled": False,
                "endpoint": "http://yaml-endpoint:6006",
            },
        }
        config_file.write_text(yaml.dump(config_data))

        config = load_config(config_file)
        phoenix_config = get_phoenix_config(
            config, cli_enabled=True, cli_endpoint="http://cli-endpoint:6006"
        )

        # CLI overrides should take precedence
        assert phoenix_config["enabled"] is True
        assert phoenix_config["endpoint"] == "http://cli-endpoint:6006"

    def test_env_var_precedence_over_default(self, tmp_path):
        """Test that environment variables take precedence over defaults."""
        config_file = tmp_path / "eval_config.yaml"
        config_data = {
            "datasets": {"legal_rag_bench": {"path": "data/rag/legal_rag_bench"}},
            "metrics": {"ragas": {"enabled": True}},
            "models": {"openai": {"model": "gpt-4o-mini"}},
        }
        config_file.write_text(yaml.dump(config_data))

        with patch.dict(os.environ, {"PHOENIX_ENDPOINT": "http://env-endpoint:6006"}):
            config = load_config(config_file)
            phoenix_config = get_phoenix_config(config)

        assert phoenix_config["endpoint"] == "http://env-endpoint:6006"
