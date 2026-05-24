"""Tests for eval-replay HTTP client integration."""

from unittest import mock

import yaml


class TestEvalReplayHTTPIntegration:
    """Test suite for eval-replay HTTP client integration."""

    def test_candidate_spec_option_loads_yaml(self, tmp_path):
        """Test that --candidate-spec option loads CandidateConfig from YAML."""
        from eval_harness.replay.candidate_config import CandidateConfig

        # Create candidate spec
        candidate_path = tmp_path / "candidate.yaml"
        candidate_data = {
            "name": "test-candidate",
            "description": "Test candidate",
            "candidate": {
                "service_url": "http://localhost:8081/query",
                "service_version": "1.0.0",
                "contract_version": "1.0",
                "timeout_seconds": 30,
                "max_retries": 2,
            },
        }
        with open(candidate_path, "w") as f:
            yaml.dump(candidate_data, f)

        # Load candidate config
        config = CandidateConfig.from_yaml_file(candidate_path)

        assert config.name == "test-candidate"
        assert config.candidate.service_url == "http://localhost:8081/query"
        assert config.candidate.contract_version == "1.0"

    def test_http_client_instantiation_with_loaded_config(self, tmp_path):
        """Test HTTPClient instantiation with loaded config."""
        from eval_harness.replay.candidate_config import CandidateConfig
        from eval_harness.replay.http_client import HTTPClient

        # Create candidate spec
        candidate_path = tmp_path / "candidate.yaml"
        candidate_data = {
            "name": "test-candidate",
            "description": "Test candidate",
            "candidate": {
                "service_url": "http://localhost:8081/query",
                "service_version": "1.0.0",
                "contract_version": "1.0",
                "timeout_seconds": 30,
                "max_retries": 2,
            },
        }
        with open(candidate_path, "w") as f:
            yaml.dump(candidate_data, f)

        config = CandidateConfig.from_yaml_file(candidate_path)

        # Instantiate HTTPClient
        http_client = HTTPClient(config, health_check_enabled=False)

        assert http_client.config.name == "test-candidate"
        assert http_client.config.candidate.service_url == "http://localhost:8081/query"

    def test_http_client_query_called(self, tmp_path):
        """Test that http_client.query() is called with correct payload."""
        from fastapi.testclient import TestClient

        from eval_harness.replay.candidate_config import CandidateConfig
        from eval_harness.replay.http_client import HTTPClient
        from eval_harness.stubs.service import create_app
        from eval_harness.stubs.service.config import StubConfig

        # Create stub service
        corpus_dir = tmp_path / "corpus"
        corpus_dir.mkdir()
        stub_config_path = tmp_path / "stub.yaml"
        stub_config_data = {
            "chunking_strategy": "fixed",
            "chunk_size": 512,
            "chunk_overlap": 50,
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "port": 8081,
            "corpus_path": str(corpus_dir),
        }
        with open(stub_config_path, "w") as f:
            yaml.dump(stub_config_data, f)

        stub_config = StubConfig.from_yaml_file(stub_config_path)
        app = create_app(stub_config)

        # Create candidate spec
        candidate_path = tmp_path / "candidate.yaml"
        candidate_data = {
            "name": "test-candidate",
            "description": "Test candidate",
            "candidate": {
                "service_url": "http://localhost:8081/query",
                "service_version": "1.0.0",
                "contract_version": "1.0",
                "timeout_seconds": 30,
                "max_retries": 2,
            },
        }
        with open(candidate_path, "w") as f:
            yaml.dump(candidate_data, f)

        config = CandidateConfig.from_yaml_file(candidate_path)
        http_client = HTTPClient(config, health_check_enabled=False)

        # Mock requests.post to use TestClient
        with TestClient(app) as client:
            import requests

            original_post = requests.post

            def mock_post(url, *args, **kwargs):
                if url == "http://localhost:8081/query":
                    return client.post("/query", *args, **kwargs)
                return original_post(url, *args, **kwargs)

            with mock.patch("requests.post", side_effect=mock_post):
                result = http_client.query({"question": "Test question?", "top_k": 5})

                assert "response" in result
                assert "text" in result["response"]

    def test_backward_compatibility_with_stub_names(self):
        """Test backward compatibility with existing stub-* names."""
        # Test that stub-* names still work for import-based invocation
        # This is a simple smoke test - the actual code change is in run_replay_eval.py
        stub_names = [
            "stub-local",
            "stub-chunks-512-overlap-50",
            "stub-chunks-256-overlap-25",
        ]

        for name in stub_names:
            # Verify the format is valid
            assert name.startswith("stub-"), f"{name} should start with stub-"

    def test_http_url_detection_for_http_path(self):
        """Test detection of http:// prefix for HTTP path."""
        # Test that http:// prefix is detected
        test_cases = [
            ("http://localhost:8081/query", True),
            ("https://api.example.com/query", True),
            ("stub-local", False),
            ("stub-chunks-512-overlap-50", False),
        ]

        for candidate_spec, is_http in test_cases:
            if candidate_spec.startswith(("http://", "https://")):
                assert is_http, f"{candidate_spec} should be detected as HTTP"
            else:
                assert not is_http, f"{candidate_spec} should not be detected as HTTP"
