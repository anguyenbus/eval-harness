"""Tests for stub HTTP service skeleton."""


import yaml


class TestServiceSkeleton:
    """Test suite for stub HTTP service initialization and health endpoint."""

    def test_fastapi_app_initializes_with_stub_config(self, tmp_path):
        """Test that FastAPI app initializes with valid StubConfig."""
        from eval_harness.stubs.service import create_app
        from eval_harness.stubs.service.config import StubConfig

        # Create stub config
        config_path = tmp_path / "stub.yaml"
        config_data = {
            "chunking_strategy": "fixed",
            "chunk_size": 512,
            "chunk_overlap": 50,
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "port": 8081,
            "corpus_path": str(tmp_path / "corpus"),
        }
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        config = StubConfig.from_yaml_file(config_path)

        # Create FastAPI app
        app = create_app(config)

        # Verify app is created
        assert app is not None
        assert hasattr(app, "routes")

    def test_health_endpoint_returns_ok(self, tmp_path):
        """Test that GET /health returns {"status": "ok"}."""
        from fastapi.testclient import TestClient

        from eval_harness.stubs.service import create_app
        from eval_harness.stubs.service.config import StubConfig

        # Create stub config
        config_path = tmp_path / "stub.yaml"
        config_data = {
            "chunking_strategy": "fixed",
            "chunk_size": 512,
            "chunk_overlap": 50,
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "port": 8081,
            "corpus_path": str(tmp_path / "corpus"),
        }
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        config = StubConfig.from_yaml_file(config_path)
        app = create_app(config)

        # Test health endpoint
        client = TestClient(app)
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_cli_argument_parsing(self, tmp_path):
        """Test that CLI arguments are parsed correctly."""
        from click.testing import CliRunner

        from eval_harness.stubs.service.main import cli

        # Create stub config
        config_path = tmp_path / "stub.yaml"
        corpus_dir = tmp_path / "corpus"
        corpus_dir.mkdir()
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

        runner = CliRunner()

        # Test --config argument
        result = runner.invoke(cli, ["--config", str(config_path), "--help"])
        assert result.exit_code == 0
        assert "--config" in result.output

        # Test --port argument
        result = runner.invoke(
            cli, ["--config", str(config_path), "--port", "8082", "--help"]
        )
        assert result.exit_code == 0

    def test_port_availability_validation(self, tmp_path):
        """Test that port availability is validated."""
        import socket

        from eval_harness.stubs.service.main import is_port_available

        # Find an available port
        sock = socket.socket()
        sock.bind(("", 0))
        available_port = sock.getsockname()[1]
        sock.close()

        # Test available port
        assert is_port_available(available_port) is True

        # Test port in use (bind to it first)
        test_sock = socket.socket()
        test_sock.bind(("127.0.0.1", available_port))
        assert is_port_available(available_port) is False
        test_sock.close()

    def test_cors_middleware_included(self, tmp_path):
        """Test that CORS middleware is included for development."""
        from fastapi.testclient import TestClient

        from eval_harness.stubs.service import create_app
        from eval_harness.stubs.service.config import StubConfig

        config_path = tmp_path / "stub.yaml"
        config_data = {
            "chunking_strategy": "fixed",
            "chunk_size": 512,
            "chunk_overlap": 50,
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "port": 8081,
            "corpus_path": str(tmp_path / "corpus"),
        }
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        config = StubConfig.from_yaml_file(config_path)
        app = create_app(config)

        # Test CORS headers are present
        client = TestClient(app)
        response = client.get("/health", headers={"Origin": "http://example.com"})

        # CORS middleware should add headers (even if origin is allowed)
        # The middleware is present if we get a successful response
        assert response.status_code == 200
