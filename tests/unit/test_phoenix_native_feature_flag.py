"""Tests for Phoenix native feature flag implementation."""

import pytest

from eval_harness.observability.config_phoenix_native import (
    get_phoenix_native_config,
)


class TestPhoenixNativeFeatureFlag:
    """Test suite for use_phoenix_native feature flag."""

    def test_feature_flag_true_from_config(self, tmp_path):
        """Test that use_phoenix_native: true loads correctly."""
        config_content = """
        datasets:
          legalbench_rag:
            path: /data/legalbench

        metrics:
          text_fidelity:
            threshold: 0.95

        models:
          judge_model: claude-opus-4-7
          temperature: 0

        phoenix_native:
          use_phoenix_native: true
        """
        config_file = tmp_path / "eval_config.yaml"
        config_file.write_text(config_content)

        from eval_harness.config import load_config

        config = load_config(config_file)
        phoenix_config = get_phoenix_native_config(config)

        assert phoenix_config["use_phoenix_native"] is True

    def test_feature_flag_false_from_config(self, tmp_path):
        """Test that use_phoenix_native: false loads correctly."""
        config_content = """
        datasets:
          legalbench_rag:
            path: /data/legalbench

        metrics:
          text_fidelity:
            threshold: 0.95

        models:
          judge_model: claude-opus-4-7
          temperature: 0

        phoenix_native:
          use_phoenix_native: false
        """
        config_file = tmp_path / "eval_config.yaml"
        config_file.write_text(config_content)

        from eval_harness.config import load_config

        config = load_config(config_file)
        phoenix_config = get_phoenix_native_config(config)

        assert phoenix_config["use_phoenix_native"] is False

    def test_feature_flag_default_value(self, tmp_path):
        """Test that default value is False when flag not specified."""
        config_content = """
        datasets:
          legalbench_rag:
            path: /data/legalbench

        metrics:
          text_fidelity:
            threshold: 0.95

        models:
          judge_model: claude-opus-4-7
          temperature: 0
        """
        config_file = tmp_path / "eval_config.yaml"
        config_file.write_text(config_content)

        from eval_harness.config import load_config

        config = load_config(config_file)
        phoenix_config = get_phoenix_native_config(config)

        # Default should be False for safe rollback
        assert phoenix_config["use_phoenix_native"] is False

    def test_feature_flag_missing_phoenix_native_section(self, tmp_path):
        """Test that missing phoenix_native section returns default."""
        config_content = """
        datasets:
          legalbench_rag:
            path: /data/legalbench

        metrics:
          text_fidelity:
            threshold: 0.95

        models:
          judge_model: claude-opus-4-7
          temperature: 0

        phoenix:
          endpoint: http://localhost:6006
        """
        config_file = tmp_path / "eval_config.yaml"
        config_file.write_text(config_content)

        from eval_harness.config import load_config

        config = load_config(config_file)
        phoenix_config = get_phoenix_native_config(config)

        # Should return default when phoenix_native section missing
        assert phoenix_config["use_phoenix_native"] is False

    def test_feature_flag_backward_compatibility(self, tmp_path):
        """Test that existing configs without flag still load correctly."""
        # Old config format without phoenix_native section
        config_content = """
        datasets:
          legalbench_rag:
            path: /data/legalbench

        metrics:
          text_fidelity:
            threshold: 0.95

        models:
          judge_model: claude-opus-4-7
          temperature: 0
        """
        config_file = tmp_path / "eval_config.yaml"
        config_file.write_text(config_content)

        from eval_harness.config import load_config

        # Should not raise error
        config = load_config(config_file)
        phoenix_config = get_phoenix_native_config(config)

        assert phoenix_config["use_phoenix_native"] is False
        # Should still have old phoenix config available
        from eval_harness.observability.config import get_phoenix_config

        old_phoenix_config = get_phoenix_config(config)
        assert old_phoenix_config["enabled"] is False
