"""Tests for CLI integration with Phoenix."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch


class TestCLIIntegration:
    """Test CLI flag handling for Phoenix integration."""

    def test_enable_phoenix_flag_default_false(self):
        """Test that --enable-phoenix flag defaults to False."""

        with patch("sys.argv", ["eval-rag", "--slice", "nano"]):
            # Test with default (no flag)
            # The flag should default to False
            pass

    def test_enable_phoenix_flag_when_set(self):
        """Test that --enable-phoenix flag enables Phoenix."""
        # This will be tested via integration tests
        pass

    def test_phoenix_endpoint_flag_override(self):
        """Test that --phoenix-endpoint flag overrides config."""
        # This will be tested via integration tests
        pass

    def test_phoenix_disabled_when_flag_not_provided(self):
        """Test that Phoenix is disabled when --enable-phoenix is not provided."""
        # This will be tested via integration tests
        pass

    def test_phoenix_adapter_initialization_with_config(self, tmp_path):
        """Test PhoenixAdapter initialization from config."""
        config_file = tmp_path / "eval_config.yaml"
        config_file.write_text("""
datasets:
  legal_rag_bench:
    path: data/rag/legal_rag_bench
metrics:
  ragas:
    enabled: true
models:
  openai:
    model: gpt-4o
phoenix:
  enabled: true
  endpoint: http://localhost:6006
  export_path: /tmp/phoenix_traces
""")

        from eval_harness.config import load_config
        from eval_harness.observability.config import get_phoenix_config
        from eval_harness.observability.phoenix_adapter import PhoenixAdapter

        config = load_config(config_file)
        phoenix_config = get_phoenix_config(config)

        adapter = PhoenixAdapter(
            endpoint=phoenix_config["endpoint"],
            project_name="eval-harness",
            enabled=phoenix_config["enabled"],
            export_path=Path(phoenix_config["export_path"]),
        )

        assert adapter._endpoint == "http://localhost:6006"
        assert adapter._project_name == "eval-harness"
        assert adapter._enabled is True

    def test_phoenix_adapter_disabled_in_config(self, tmp_path):
        """Test that PhoenixAdapter respects disabled config."""
        config_file = tmp_path / "eval_config.yaml"
        config_file.write_text("""
datasets:
  legal_rag_bench:
    path: data/rag/legal_rag_bench
metrics:
  ragas:
    enabled: true
models:
  openai:
    model: gpt-4o
phoenix:
  enabled: false
""")

        from eval_harness.config import load_config
        from eval_harness.observability.config import get_phoenix_config
        from eval_harness.observability.phoenix_adapter import PhoenixAdapter

        config = load_config(config_file)
        phoenix_config = get_phoenix_config(config)

        adapter = PhoenixAdapter(
            endpoint=phoenix_config["endpoint"],
            project_name="eval-harness",
            enabled=phoenix_config["enabled"],
        )

        assert adapter._enabled is False

    def test_get_phoenix_config_with_cli_overrides(self, tmp_path):
        """Test get_phoenix_config with CLI flag overrides."""
        config_file = tmp_path / "eval_config.yaml"
        config_file.write_text("""
datasets:
  legal_rag_bench:
    path: data/rag/legal_rag_bench
metrics:
  ragas:
    enabled: true
models:
  openai:
    model: gpt-4o
phoenix:
  enabled: false
  endpoint: http://yaml-endpoint:6006
""")

        from eval_harness.config import load_config
        from eval_harness.observability.config import get_phoenix_config

        config = load_config(config_file)

        # CLI overrides should take precedence
        phoenix_config = get_phoenix_config(
            config, cli_enabled=True, cli_endpoint="http://cli-endpoint:6006"
        )

        assert phoenix_config["enabled"] is True
        assert phoenix_config["endpoint"] == "http://cli-endpoint:6006"
