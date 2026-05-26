"""Tests for config schemas - StubConfig and CandidateConfig."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from pydantic import ValidationError

from eval_harness.replay.candidate_config import CandidateConfig


class TestStubConfigValidation:
    """Test StubConfig validation."""

    def test_chunk_size_range_validation(self, tmp_path):
        """Test that chunk_size must be between 1 and 8192."""
        from eval_harness.stubs.service.config import StubConfig

        config_path = tmp_path / "test.yaml"

        # Test valid chunk_size
        valid_config = {
            "chunking_strategy": "fixed",
            "chunk_size": 512,
            "chunk_overlap": 50,
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "port": 8081,
            "corpus_path": "data/corpus",
        }
        with open(config_path, "w") as f:
            yaml.dump(valid_config, f)

        config = StubConfig.from_yaml_file(config_path)
        assert config.chunk_size == 512

        # Test chunk_size too small
        invalid_small = valid_config.copy()
        invalid_small["chunk_size"] = 0
        with open(config_path, "w") as f:
            yaml.dump(invalid_small, f)

        with pytest.raises(ValidationError):
            StubConfig.from_yaml_file(config_path)

        # Test chunk_size too large
        invalid_large = valid_config.copy()
        invalid_large["chunk_size"] = 8193
        with open(config_path, "w") as f:
            yaml.dump(invalid_large, f)

        with pytest.raises(ValidationError):
            StubConfig.from_yaml_file(config_path)

    def test_chunk_overlap_constraint(self, tmp_path):
        """Test that chunk_overlap must be less than chunk_size."""
        from eval_harness.stubs.service.config import StubConfig

        config_path = tmp_path / "test.yaml"

        # Test valid overlap
        valid_config = {
            "chunking_strategy": "fixed",
            "chunk_size": 512,
            "chunk_overlap": 100,
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "port": 8081,
            "corpus_path": "data/corpus",
        }
        with open(config_path, "w") as f:
            yaml.dump(valid_config, f)

        config = StubConfig.from_yaml_file(config_path)
        assert config.chunk_overlap == 100

        # Test overlap >= chunk_size (invalid)
        invalid = valid_config.copy()
        invalid["chunk_overlap"] = 512
        with open(config_path, "w") as f:
            yaml.dump(invalid, f)

        with pytest.raises(ValidationError):
            StubConfig.from_yaml_file(config_path)

    def test_port_format_validation(self, tmp_path):
        """Test that port must be valid and unique."""
        from eval_harness.stubs.service.config import StubConfig

        config_path = tmp_path / "test.yaml"

        # Test valid port
        valid_config = {
            "chunking_strategy": "fixed",
            "chunk_size": 512,
            "chunk_overlap": 50,
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "port": 8081,
            "corpus_path": "data/corpus",
        }
        with open(config_path, "w") as f:
            yaml.dump(valid_config, f)

        config = StubConfig.from_yaml_file(config_path)
        assert config.port == 8081

        # Test port out of range
        invalid = valid_config.copy()
        invalid["port"] = 70000
        with open(config_path, "w") as f:
            yaml.dump(invalid, f)

        with pytest.raises(ValidationError):
            StubConfig.from_yaml_file(config_path)

    def test_export_spans_default_true(self):
        """Test that export_spans defaults to True."""
        from eval_harness.stubs.service.config import StubConfig

        config = StubConfig(
            chunking_strategy="fixed",
            chunk_size=512,
            chunk_overlap=50,
            port=8081,
            corpus_path=Path("/fake/corpus"),
        )
        assert config.export_spans is True

    def test_export_spans_can_be_disabled(self, tmp_path):
        """Test that export_spans can be set to False."""
        from eval_harness.stubs.service.config import StubConfig

        config_path = tmp_path / "test.yaml"

        config_data = {
            "chunking_strategy": "fixed",
            "chunk_size": 512,
            "chunk_overlap": 50,
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "port": 8081,
            "corpus_path": "data/corpus",
            "export_spans": False,
        }
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        config = StubConfig.from_yaml_file(config_path)
        assert config.export_spans is False


class TestStubConfigTracerInitialization:
    """Test StubConfig tracer initialization based on export_spans flag."""

    @patch("eval_harness.stubs.service.tracing.setup_phoenix_tracer")
    def test_tracer_initialized_when_export_spans_true(self, mock_setup_tracer):
        """Test that tracer is initialized when export_spans=True."""
        from eval_harness.stubs.service import create_app
        from eval_harness.stubs.service.config import StubConfig

        mock_setup_tracer.return_value = (MagicMock(), MagicMock())

        config = StubConfig(
            chunking_strategy="fixed",
            chunk_size=512,
            chunk_overlap=50,
            port=8081,
            corpus_path=Path("/fake/corpus"),
            export_spans=True,
        )

        app = create_app(config)

        # Verify tracer was initialized
        mock_setup_tracer.assert_called_once()
        assert app.state.tracer is not None
        assert app.state.export_spans is True

    @patch("eval_harness.stubs.service.tracing.setup_phoenix_tracer")
    def test_tracer_not_initialized_when_export_spans_false(self, mock_setup_tracer):
        """Test that tracer is NOT initialized when export_spans=False."""
        from eval_harness.stubs.service import create_app
        from eval_harness.stubs.service.config import StubConfig

        config = StubConfig(
            chunking_strategy="fixed",
            chunk_size=512,
            chunk_overlap=50,
            port=8081,
            corpus_path=Path("/fake/corpus"),
            export_spans=False,
        )

        app = create_app(config)

        # Verify tracer was NOT initialized
        mock_setup_tracer.assert_not_called()
        assert app.state.tracer is None
        assert app.state.export_spans is False


class TestCandidateSpecValidation:
    """Test CandidateSpec validation."""

    def test_service_url_validation(self, tmp_path):
        """Test that service_url must start with http:// or https://."""
        config_path = tmp_path / "candidate.yaml"

        # Test valid URLs
        for valid_url in ["http://localhost:8081/query", "https://api.example.com/query"]:
            valid_config = {
                "name": "test-candidate",
                "description": "Test candidate",
                "candidate": {
                    "service_url": valid_url,
                    "service_version": "1.0.0",
                    "contract_version": "1.0",
                },
            }
            with open(config_path, "w") as f:
                yaml.dump(valid_config, f)

            config = CandidateConfig.from_yaml_file(config_path)
            assert config.candidate.service_url == valid_url

        # Test invalid URL (missing http/https)
        invalid_config = {
            "name": "test-candidate",
            "description": "Test candidate",
            "candidate": {
                "service_url": "localhost:8081/query",
                "service_version": "1.0.0",
                "contract_version": "1.0",
            },
        }
        with open(config_path, "w") as f:
            yaml.dump(invalid_config, f)

        with pytest.raises(ValueError, match="must start with http:// or https://"):
            CandidateConfig.from_yaml_file(config_path)

    def test_from_yaml_file_with_invalid_yaml(self, tmp_path):
        """Test that from_yaml_file handles invalid YAML gracefully."""
        config_path = tmp_path / "candidate.yaml"

        # Write invalid YAML
        with open(config_path, "w") as f:
            f.write("invalid: yaml: content: [[[[")

        with pytest.raises(yaml.YAMLError):
            CandidateConfig.from_yaml_file(config_path)

    def test_from_yaml_file_missing_file(self, tmp_path):
        """Test that from_yaml_file raises FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError, match="Candidate config not found"):
            CandidateConfig.from_yaml_file(tmp_path / "nonexistent.yaml")


class TestCorpusPathResolution:
    """Test corpus_path resolution relative to YAML file location."""

    def test_corpus_path_resolution_relative(self, tmp_path):
        """Test that corpus_path is resolved relative to YAML file location."""
        from eval_harness.stubs.service.config import StubConfig

        # Create subdirectories - config is in configs/, data is in data/
        subdir = tmp_path / "configs"
        subdir.mkdir()
        corpus_dir = tmp_path / "data" / "corpus"
        corpus_dir.mkdir(parents=True)

        config_path = subdir / "stub.yaml"

        # Use relative path - config is one level down from tmp_path
        config_data = {
            "chunking_strategy": "fixed",
            "chunk_size": 512,
            "chunk_overlap": 50,
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "port": 8081,
            "corpus_path": "../data/corpus",  # One level up, then into data
        }
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        config = StubConfig.from_yaml_file(config_path)

        # Path should be resolved relative to config file and absolute
        assert config.corpus_path.is_absolute()
        assert config.corpus_path == corpus_dir

    def test_corpus_path_absolute(self, tmp_path):
        """Test that absolute corpus_path is preserved."""
        from eval_harness.stubs.service.config import StubConfig

        config_path = tmp_path / "stub.yaml"
        corpus_dir = tmp_path / "corpus"

        config_data = {
            "chunking_strategy": "fixed",
            "chunk_size": 512,
            "chunk_overlap": 50,
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "port": 8081,
            "corpus_path": str(corpus_dir),
        }
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        config = StubConfig.from_yaml_file(config_path)

        assert config.corpus_path == corpus_dir
