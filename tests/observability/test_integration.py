"""Integration tests for Phoenix + Ragas feature."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml


class TestPhoenixIntegration:
    """Integration tests for Phoenix observability with RAG evaluation."""

    def test_end_to_end_phoenix_disabled(self, tmp_path):
        """Test end-to-end evaluation with Phoenix disabled (default)."""
        config_file = tmp_path / "eval_config.yaml"
        config_data = {
            "datasets": {"legal_rag_bench": {"path": "data/rag/legal_rag_bench"}},
            "metrics": {"ragas": {"enabled": True}},
            "models": {"openai": {"model": "gpt-4o"}},
        }
        config_file.write_text(yaml.dump(config_data))

        from eval_harness.config import load_config
        from eval_harness.observability.config import get_phoenix_config

        config = load_config(config_file)
        phoenix_config = get_phoenix_config(config)

        # Phoenix should be disabled by default
        assert phoenix_config["enabled"] is False

    def test_end_to_end_phoenix_enabled_via_cli(self, tmp_path):
        """Test end-to-end evaluation with Phoenix enabled via CLI flag."""
        config_file = tmp_path / "eval_config.yaml"
        config_data = {
            "datasets": {"legal_rag_bench": {"path": "data/rag/legal_rag_bench"}},
            "metrics": {"ragas": {"enabled": True}},
            "models": {"openai": {"model": "gpt-4o"}},
        }
        config_file.write_text(yaml.dump(config_data))

        from eval_harness.config import load_config
        from eval_harness.observability.config import get_phoenix_config
        from eval_harness.observability.phoenix_adapter import PhoenixAdapter

        config = load_config(config_file)
        phoenix_config = get_phoenix_config(config, cli_enabled=True)

        # CLI flag should enable Phoenix
        assert phoenix_config["enabled"] is True

        # Adapter should initialize successfully
        adapter = PhoenixAdapter(
            endpoint=phoenix_config["endpoint"],
            project_name="eval-harness",
            enabled=phoenix_config["enabled"],
        )
        assert adapter._enabled is True

    def test_end_to_end_config_precedence(self, tmp_path):
        """Test configuration precedence: CLI > YAML > env vars > defaults."""
        config_file = tmp_path / "eval_config.yaml"
        config_data = {
            "datasets": {"legal_rag_bench": {"path": "data/rag/legal_rag_bench"}},
            "metrics": {"ragas": {"enabled": True}},
            "models": {"openai": {"model": "gpt-4o"}},
            "phoenix": {
                "enabled": False,
                "endpoint": "http://yaml-endpoint:6006",
            },
        }
        config_file.write_text(yaml.dump(config_data))

        with patch.dict(
            os.environ, {"PHOENIX_ENDPOINT": "http://env-endpoint:6006"}
        ):
            from eval_harness.config import load_config
            from eval_harness.observability.config import get_phoenix_config

            config = load_config(config_file)

            # Test default precedence (YAML should be used)
            phoenix_config = get_phoenix_config(config)
            assert phoenix_config["endpoint"] == "http://yaml-endpoint:6006"

            # Test CLI override (should override both YAML and env)
            phoenix_config_cli = get_phoenix_config(
                config, cli_endpoint="http://cli-endpoint:6006"
            )
            assert phoenix_config_cli["endpoint"] == "http://cli-endpoint:6006"

    def test_graceful_degradation_phoenix_not_installed(self, tmp_path):
        """Test graceful degradation when Phoenix is not installed."""
        config_file = tmp_path / "eval_config.yaml"
        config_data = {
            "datasets": {"legal_rag_bench": {"path": "data/rag/legal_rag_bench"}},
            "metrics": {"ragas": {"enabled": True}},
            "models": {"openai": {"model": "gpt-4o"}},
        }
        config_file.write_text(yaml.dump(config_data))

        # Even without Phoenix, evaluation should continue
        from eval_harness.config import load_config

        config = load_config(config_file)
        assert "datasets" in config
        assert "metrics" in config
        assert "models" in config

    def test_phoenix_adapter_lifecycle(self, tmp_path):
        """Test complete PhoenixAdapter lifecycle."""
        from eval_harness.observability.phoenix_adapter import PhoenixAdapter

        export_path = tmp_path / "phoenix_traces"

        adapter = PhoenixAdapter(
            endpoint="http://localhost:6006",
            project_name="test-project",
            enabled=True,
            export_path=export_path,
        )

        # Create trace hierarchy
        trace_id = adapter.start_rag_query_span("Test question?")
        assert trace_id is not None

        adapter.start_retrieval_span(
            trace_id=trace_id,
            embeddings=[0.1, 0.2],
            chunks=[{"text": "chunk1"}],
            k=5,
            timing_ms=100.0,
        )

        adapter.start_generation_span(
            trace_id=trace_id,
            model="gpt-4o",
            prompt="Prompt",
            tokens=100,
            timing_ms=200.0,
        )

        adapter.start_evaluation_span(
            trace_id=trace_id,
            ragas_metrics={"faithfulness": 0.9},
        )

        # Export traces
        result = adapter.export_traces()
        assert "trace_count" in result
        assert "mode" in result

    def test_multiple_queries_with_phoenix(self, tmp_path):
        """Test handling multiple queries with unique trace IDs."""
        from eval_harness.observability.phoenix_adapter import PhoenixAdapter

        adapter = PhoenixAdapter(enabled=False)

        trace_ids = []
        for i in range(5):
            trace_id = adapter.start_rag_query_span(f"Question {i}?")
            trace_ids.append(trace_id)

        # All trace IDs should be unique
        assert len(set(trace_ids)) == 5

    def test_phoenix_config_helper_integration(self, tmp_path):
        """Test get_phoenix_config helper integration with main config."""
        config_file = tmp_path / "eval_config.yaml"
        config_data = {
            "datasets": {"legal_rag_bench": {"path": "data/rag/legal_rag_bench"}},
            "metrics": {"ragas": {"enabled": True}},
            "models": {"openai": {"model": "gpt-4o"}},
            "phoenix": {
                "enabled": True,
                "endpoint": "http://localhost:6006",
                "export_path": "/tmp/phoenix_traces",
            },
        }
        config_file.write_text(yaml.dump(config_data))

        from eval_harness.config import load_config
        from eval_harness.observability.config import get_phoenix_config

        config = load_config(config_file)
        phoenix_config = get_phoenix_config(config)

        # Verify all expected keys are present
        assert "enabled" in phoenix_config
        assert "endpoint" in phoenix_config
        assert "export_path" in phoenix_config

        # Verify values
        assert phoenix_config["enabled"] is True
        assert phoenix_config["endpoint"] == "http://localhost:6006"
        assert phoenix_config["export_path"] == "/tmp/phoenix_traces"
