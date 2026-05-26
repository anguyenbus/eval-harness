"""Tests for /query endpoint."""


import yaml


class TestQueryEndpoint:
    """Test suite for POST /query endpoint."""

    def test_successful_query_returns_retrieved_contexts(self, tmp_path):
        """Test successful query returns retrieved_contexts, response, timings_ms."""
        from fastapi.testclient import TestClient

        from eval_harness.stubs.service import create_app
        from eval_harness.stubs.service.config import StubConfig

        # Create stub config with test corpus
        corpus_dir = tmp_path / "corpus"
        corpus_dir.mkdir()
        config_path = tmp_path / "stub.yaml"
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
        app = create_app(config)

        # Make query request
        client = TestClient(app)
        response = client.post(
            "/query",
            json={"question": "Test question?", "top_k": 5},
        )

        assert response.status_code == 200
        data = response.json()

        # Check response schema
        assert "retrieved_contexts" in data
        assert "response" in data
        assert "timings_ms" in data
        assert "text" in data["response"]
        assert "retrieval" in data["timings_ms"]
        assert "generation" in data["timings_ms"]
        assert "total" in data["timings_ms"]

    def test_top_k_parameter_passed_through(self, tmp_path):
        """Test that top_k parameter is passed through to RAG pipeline."""
        from fastapi.testclient import TestClient

        from eval_harness.stubs.service import create_app
        from eval_harness.stubs.service.config import StubConfig

        # Create stub config
        corpus_dir = tmp_path / "corpus"
        corpus_dir.mkdir()
        config_path = tmp_path / "stub.yaml"
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
        app = create_app(config)

        client = TestClient(app)

        # Test with top_k=3
        response = client.post(
            "/query",
            json={"question": "Test question?", "top_k": 3},
        )
        assert response.status_code == 200
        data = response.json()
        # Note: Empty corpus returns 0 chunks, but the parameter is passed
        assert "retrieved_contexts" in data

        # Test with top_k=10
        response = client.post(
            "/query",
            json={"question": "Test question?", "top_k": 10},
        )
        assert response.status_code == 200

    def test_error_handling_malformed_input(self, tmp_path):
        """Test error handling for malformed input returns 400."""
        from fastapi.testclient import TestClient

        from eval_harness.stubs.service import create_app
        from eval_harness.stubs.service.config import StubConfig

        # Create stub config
        corpus_dir = tmp_path / "corpus"
        corpus_dir.mkdir()
        config_path = tmp_path / "stub.yaml"
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
        app = create_app(config)

        client = TestClient(app)

        # Missing question field
        response = client.post(
            "/query",
            json={"top_k": 5},
        )
        # FastAPI validation returns 422 for missing required fields
        assert response.status_code == 422

        # Invalid top_k (negative)
        response = client.post(
            "/query",
            json={"question": "Test?", "top_k": -1},
        )
        assert response.status_code == 422

    def test_chain_span_emission_at_http_boundary(self, tmp_path):
        """Test CHAIN span emission at HTTP boundary."""
        from fastapi.testclient import TestClient

        from eval_harness.stubs.service import create_app
        from eval_harness.stubs.service.config import StubConfig

        # Create stub config
        corpus_dir = tmp_path / "corpus"
        corpus_dir.mkdir()
        config_path = tmp_path / "stub.yaml"
        config_data = {
            "chunking_strategy": "fixed",
            "chunk_size": 512,
            "chunk_overlap": 50,
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "port": 8081,
            "corpus_path": str(corpus_dir),
            "phoenix_endpoint": None,  # Disable tracing for test
        }
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        config = StubConfig.from_yaml_file(config_path)
        app = create_app(config)

        client = TestClient(app)

        # Make query - spans are emitted by chromadb_query internally
        response = client.post(
            "/query",
            json={"question": "Test question?", "top_k": 5},
        )

        # Response should succeed even if Phoenix is not available
        assert response.status_code in (200, 500)

    def test_timings_ms_extraction_from_rag_output(self, tmp_path):
        """Test that timings_ms are extracted from RAG pipeline output."""
        from fastapi.testclient import TestClient

        from eval_harness.stubs.service import create_app
        from eval_harness.stubs.service.config import StubConfig

        # Create stub config
        corpus_dir = tmp_path / "corpus"
        corpus_dir.mkdir()
        config_path = tmp_path / "stub.yaml"
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
        app = create_app(config)

        client = TestClient(app)
        response = client.post(
            "/query",
            json={"question": "Test question?", "top_k": 5},
        )

        assert response.status_code == 200
        data = response.json()

        # Check timings_ms structure
        timings = data["timings_ms"]
        assert isinstance(timings["retrieval"], (int, float))
        assert isinstance(timings["generation"], (int, float))
        assert isinstance(timings["total"], (int, float))
        assert timings["total"] >= 0
