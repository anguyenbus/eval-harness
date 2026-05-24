"""Tests for HTTPClient."""

from unittest import mock

import pytest
import yaml


class TestHTTPClient:
    """Test suite for HTTP client."""

    def test_successful_query_with_candidate_config(self, tmp_path):
        """Test successful query() call with CandidateConfig."""
        from fastapi.testclient import TestClient

        from eval_harness.replay.candidate_config import CandidateConfig
        from eval_harness.replay.http_client import HTTPClient
        from eval_harness.stubs.service import create_app
        from eval_harness.stubs.service.config import StubConfig

        # Create and start stub service
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

        candidate_config = CandidateConfig.from_yaml_file(candidate_path)

        # Test with mock server
        with TestClient(app) as client:
            # Patch requests.post to use TestClient
            import requests

            original_post = requests.post

            def mock_post(url, *args, **kwargs):
                if url == "http://localhost:8081/query":
                    return client.post("/query", *args, **kwargs)
                return original_post(url, *args, **kwargs)

            with mock.patch("requests.post", side_effect=mock_post):
                http_client = HTTPClient(candidate_config, health_check_enabled=False)
                result = http_client.query({"question": "Test question?", "top_k": 5})

                assert "response" in result
                assert "text" in result["response"]

    def test_exponential_backoff_retry(self, tmp_path):
        """Test exponential backoff retry on failure."""
        from eval_harness.replay.candidate_config import CandidateConfig
        from eval_harness.replay.http_client import HTTPClient

        # Create candidate spec
        candidate_path = tmp_path / "candidate.yaml"
        candidate_data = {
            "name": "test-candidate",
            "description": "Test candidate",
            "candidate": {
                "service_url": "http://localhost:9999/query",  # Non-existent server
                "service_version": "1.0.0",
                "contract_version": "1.0",
                "timeout_seconds": 1,
                "max_retries": 2,
            },
        }
        with open(candidate_path, "w") as f:
            yaml.dump(candidate_data, f)

        candidate_config = CandidateConfig.from_yaml_file(candidate_path)

        http_client = HTTPClient(candidate_config, health_check_enabled=False)

        # Should fail after max_retries
        import time

        start = time.time()
        with pytest.raises(RuntimeError, match="Failed to call candidate"):
            http_client.query({"question": "Test?", "top_k": 5})
        elapsed = time.time() - start

        # Should have waited for exponential backoff:
        # 2^0 + 2^1 = 1 + 2 = 3 seconds minimum
        assert elapsed >= 3

    def test_timeout_enforcement_from_candidate_config(self, tmp_path):
        """Test timeout enforcement from candidate config."""
        from eval_harness.replay.candidate_config import CandidateConfig
        from eval_harness.replay.http_client import HTTPClient

        # Create candidate spec with short timeout
        candidate_path = tmp_path / "candidate.yaml"
        candidate_data = {
            "name": "test-candidate",
            "description": "Test candidate",
            "candidate": {
                "service_url": "http://localhost:9999/query",
                "service_version": "1.0.0",
                "contract_version": "1.0",
                "timeout_seconds": 1,
                "max_retries": 0,
            },
        }
        with open(candidate_path, "w") as f:
            yaml.dump(candidate_data, f)

        candidate_config = CandidateConfig.from_yaml_file(candidate_path)

        http_client = HTTPClient(candidate_config, health_check_enabled=False)

        # Should timeout quickly
        import time

        start = time.time()
        with pytest.raises(RuntimeError):
            http_client.query({"question": "Test?", "top_k": 5})
        elapsed = time.time() - start

        # Should timeout within configured time + small overhead
        assert elapsed < 5

    def test_header_propagation(self, tmp_path):
        """Test header propagation (X-Candidate-Version, X-Contract-Version)."""
        from fastapi.testclient import TestClient

        from eval_harness.replay.candidate_config import CandidateConfig
        from eval_harness.replay.http_client import HTTPClient
        from eval_harness.stubs.service import create_app
        from eval_harness.stubs.service.config import StubConfig

        # Create and start stub service
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
                "service_version": "1.2.3",
                "contract_version": "1.1",
                "timeout_seconds": 30,
                "max_retries": 2,
            },
        }
        with open(candidate_path, "w") as f:
            yaml.dump(candidate_data, f)

        candidate_config = CandidateConfig.from_yaml_file(candidate_path)

        # Track headers received
        received_headers = {}

        # Test with actual HTTP client using TestClient
        with TestClient(app) as client:
            import requests

            original_post = requests.post

            def mock_post(url, *args, **kwargs):
                if url == "http://localhost:8081/query":
                    # Capture headers
                    if "headers" in kwargs:
                        received_headers.update(kwargs["headers"])
                    return client.post("/query", *args, **kwargs)
                return original_post(url, *args, **kwargs)

            with mock.patch("requests.post", side_effect=mock_post):
                http_client = HTTPClient(candidate_config, health_check_enabled=False)
                http_client.query({"question": "Test?", "top_k": 5})

                # Check headers were sent
                assert "X-Candidate-Version" in received_headers
                assert received_headers["X-Candidate-Version"] == "1.2.3"
                assert "X-Contract-Version" in received_headers
                assert received_headers["X-Contract-Version"] == "1.1"

    def test_health_check_before_first_query(self, tmp_path):
        """Test health check before first query."""
        from fastapi.testclient import TestClient

        from eval_harness.replay.candidate_config import CandidateConfig
        from eval_harness.replay.http_client import HTTPClient
        from eval_harness.stubs.service import create_app
        from eval_harness.stubs.service.config import StubConfig

        # Create and start stub service
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

        candidate_config = CandidateConfig.from_yaml_file(candidate_path)

        # Test health check
        with TestClient(app) as client:
            import requests

            original_post = requests.post
            original_get = requests.get

            def mock_post(url, *args, **kwargs):
                if url == "http://localhost:8081/query":
                    return client.post("/query", *args, **kwargs)
                return original_post(url, *args, **kwargs)

            def mock_get(url, *args, **kwargs):
                if url == "http://localhost:8081/health":
                    return client.get("/health", *args, **kwargs)
                return original_get(url, *args, **kwargs)

            with mock.patch("requests.post", side_effect=mock_post):
                with mock.patch("requests.get", side_effect=mock_get):
                    http_client = HTTPClient(
                        candidate_config, health_check_enabled=True
                    )
                    # Health check should succeed
                    assert http_client.check_health() is True
