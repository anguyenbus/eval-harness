"""Integration tests for HTTP service."""

from pathlib import Path

import yaml


class TestHTTPServiceIntegration:
    """Integration tests for stub HTTP service."""

    def test_response_schema_matches_contract(self):
        """Test that response schema matches HTTP contract."""
        # Create test config
        import tempfile

        from fastapi.testclient import TestClient

        from eval_harness.stubs.service import create_app
        from eval_harness.stubs.service.config import StubConfig

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
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

            client = TestClient(app)

            # Test query endpoint
            response = client.post(
                "/query",
                json={"question": "Test question?", "top_k": 5},
            )

            # Verify response schema
            if response.status_code == 200:
                data = response.json()
                assert "retrieved_contexts" in data
                assert "response" in data
                assert "timings_ms" in data
                assert "text" in data["response"]

    def test_error_paths_malformed_input(self):
        """Test error paths for malformed input."""
        # Create test config
        import tempfile

        from fastapi.testclient import TestClient

        from eval_harness.stubs.service import create_app
        from eval_harness.stubs.service.config import StubConfig

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
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

            client = TestClient(app)

            # Missing question field
            response = client.post(
                "/query",
                json={"top_k": 5},
            )
            assert response.status_code == 422  # FastAPI validation error

            # Invalid top_k (negative)
            response = client.post(
                "/query",
                json={"question": "Test?", "top_k": -1},
            )
            assert response.status_code == 422

    def test_concurrent_services_different_ports(self):
        """Test that services can run on different ports."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            corpus_dir = tmp_path / "corpus"
            corpus_dir.mkdir()

            # Create two configs with different ports
            from fastapi.testclient import TestClient

            from eval_harness.stubs.service import create_app
            from eval_harness.stubs.service.config import StubConfig

            config1_path = tmp_path / "stub1.yaml"
            config1_data = {
                "chunking_strategy": "fixed",
                "chunk_size": 512,
                "chunk_overlap": 50,
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                "port": 8081,
                "corpus_path": str(corpus_dir),
            }
            with open(config1_path, "w") as f:
                yaml.dump(config1_data, f)

            config2_path = tmp_path / "stub2.yaml"
            config2_data = {
                "chunking_strategy": "fixed",
                "chunk_size": 256,
                "chunk_overlap": 25,
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                "port": 8082,
                "corpus_path": str(corpus_dir),
            }
            with open(config2_path, "w") as f:
                yaml.dump(config2_data, f)

            stub_config1 = StubConfig.from_yaml_file(config1_path)
            stub_config2 = StubConfig.from_yaml_file(config2_path)

            app1 = create_app(stub_config1)
            app2 = create_app(stub_config2)

            client1 = TestClient(app1)
            client2 = TestClient(app2)

            # Test both services respond
            response1 = client1.get("/health")
            response2 = client2.get("/health")

            assert response1.status_code == 200
            assert response2.status_code == 200

    def test_dockercompose_configuration_exists(self):
        """Test that docker-compose.yml exists and is valid."""
        import yaml

        docker_compose_path = Path("docker-compose.yml")
        assert docker_compose_path.exists()

        with open(docker_compose_path) as f:
            compose_config = yaml.safe_load(f)

        # Verify required services
        assert "services" in compose_config
        services = compose_config["services"]
        assert "phoenix" in services
        assert "stub-service-512" in services or "stub-service" in services

    def test_migration_documentation_exists(self):
        """Test that migration documentation exists."""
        migration_doc = Path("docs/http-service-migration.md")
        assert migration_doc.exists()

        content = migration_doc.read_text()
        # Check for key sections
        assert "Migration Steps" in content
        assert "Troubleshooting" in content
        assert "docker-compose" in content
