"""Tests for eval_config.yaml loading and validation."""

import pytest

from eval_harness.config import expand_env_vars, load_config


class TestConfigLoading:
    """Test suite for configuration file loading."""

    def test_valid_config_loads(self, tmp_path):
        """Test that a valid YAML config loads successfully."""
        config_content = """
        datasets:
          omnidocbench:
            path: /data/omnidocbench
          dp_bench:
            path: /data/dp_bench
          legalbench_rag:
            path: /data/legalbench

        metrics:
          text_fidelity:
            threshold: 0.95
          structure_recall:
            threshold: 0.90

        models:
          judge_model: claude-opus-4-7
          temperature: 0

        phoenix:
          endpoint: http://localhost:6006
        """
        config_file = tmp_path / "eval_config.yaml"
        config_file.write_text(config_content)

        config = load_config(config_file)

        assert config["datasets"]["omnidocbench"]["path"] == "/data/omnidocbench"
        assert config["metrics"]["text_fidelity"]["threshold"] == 0.95
        assert config["models"]["judge_model"] == "claude-opus-4-7"
        assert config["models"]["temperature"] == 0
        assert config["phoenix"]["endpoint"] == "http://localhost:6006"

    def test_environment_variable_expansion(self, tmp_path, monkeypatch):
        """Test that environment variables are expanded in paths."""
        monkeypatch.setenv("DATA_DIR", "/custom/data")

        config_content = """
        datasets:
          omnidocbench:
            path: ${DATA_DIR}/omnidocbench
          dp_bench:
            path: ${DATA_DIR}/dp_bench

        metrics:
          text_fidelity:
            threshold: 0.95

        models:
          judge_model: claude-opus-4-7
          temperature: 0
        """
        config_file = tmp_path / "eval_config.yaml"
        config_file.write_text(config_content)

        config = load_config(config_file)

        assert config["datasets"]["omnidocbench"]["path"] == "/custom/data/omnidocbench"
        assert config["datasets"]["dp_bench"]["path"] == "/custom/data/dp_bench"

    def test_missing_required_field_raises_error(self, tmp_path):
        """Test that missing required top-level sections raise error."""
        # Missing required 'datasets' section
        config_content = """
        metrics:
          text_fidelity:
            threshold: 0.95

        models:
          judge_model: claude-opus-4-7
          temperature: 0
        """
        config_file = tmp_path / "eval_config.yaml"
        config_file.write_text(config_content)

        with pytest.raises(ValueError, match="Missing required sections.*datasets"):
            load_config(config_file)

    def test_empty_file_raises_error(self, tmp_path):
        """Test that an empty config file raises error."""
        config_file = tmp_path / "eval_config.yaml"
        config_file.write_text("")

        with pytest.raises(ValueError, match="Config file is empty or invalid"):
            load_config(config_file)

    def test_nonexistent_file_raises_error(self, tmp_path):
        """Test that a missing config file raises error."""
        config_file = tmp_path / "nonexistent.yaml"

        with pytest.raises(FileNotFoundError):
            load_config(config_file)


class TestExpandEnvVars:
    """Test suite for environment variable expansion."""

    def test_expand_single_env_var(self, monkeypatch):
        """Test expanding a single environment variable."""
        monkeypatch.setenv("TEST_VAR", "test_value")

        result = expand_env_vars("${TEST_VAR}/path")
        assert result == "test_value/path"

    def test_expand_multiple_env_vars(self, monkeypatch):
        """Test expanding multiple environment variables."""
        monkeypatch.setenv("VAR1", "path1")
        monkeypatch.setenv("VAR2", "path2")

        result = expand_env_vars("${VAR1}/${VAR2}")
        assert result == "path1/path2"

    def test_no_env_var_returns_unchanged(self):
        """Test that strings without env vars are returned unchanged."""
        result = expand_env_vars("/plain/path")
        assert result == "/plain/path"

    def test_missing_env_var_raises_error(self, monkeypatch):
        """Test that missing env vars raise clear error."""
        monkeypatch.delenv("NONEXISTENT", raising=False)

        with pytest.raises(ValueError, match="Environment variable.*not set"):
            expand_env_vars("${NONEXISTENT}/path")
