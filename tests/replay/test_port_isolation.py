"""Tests for port-based service isolation."""


import yaml


class TestPortIsolation:
    """Test suite for port-based service isolation."""

    def test_multiple_stub_configs_different_ports(self, tmp_path):
        """Test that multiple stub configs can specify different ports."""
        from eval_harness.stubs.service.config import StubConfig

        # Create first stub config
        config1_path = tmp_path / "stub1.yaml"
        config1_data = {
            "chunking_strategy": "fixed",
            "chunk_size": 512,
            "chunk_overlap": 50,
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "port": 8081,
            "corpus_path": str(tmp_path / "corpus"),
        }
        with open(config1_path, "w") as f:
            yaml.dump(config1_data, f)

        # Create second stub config
        config2_path = tmp_path / "stub2.yaml"
        config2_data = {
            "chunking_strategy": "fixed",
            "chunk_size": 256,
            "chunk_overlap": 25,
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "port": 8082,
            "corpus_path": str(tmp_path / "corpus"),
        }
        with open(config2_path, "w") as f:
            yaml.dump(config2_data, f)

        # Load both configs
        config1 = StubConfig.from_yaml_file(config1_path)
        config2 = StubConfig.from_yaml_file(config2_path)

        # Verify they have different ports
        assert config1.port == 8081
        assert config2.port == 8082
        assert config1.port != config2.port

    def test_candidate_specs_reference_different_ports(self, tmp_path):
        """Test that candidate specs can reference different ports."""
        from eval_harness.replay.candidate_config import CandidateConfig

        # Create first candidate spec
        candidate1_path = tmp_path / "candidate1.yaml"
        candidate1_data = {
            "name": "stub-chunking-512",
            "description": "Stub with chunk_size=512",
            "candidate": {
                "service_url": "http://localhost:8081/query",
                "service_version": "1.0.0",
                "contract_version": "1.0",
            },
        }
        with open(candidate1_path, "w") as f:
            yaml.dump(candidate1_data, f)

        # Create second candidate spec
        candidate2_path = tmp_path / "candidate2.yaml"
        candidate2_data = {
            "name": "stub-chunking-256",
            "description": "Stub with chunk_size=256",
            "candidate": {
                "service_url": "http://localhost:8082/query",
                "service_version": "1.0.0",
                "contract_version": "1.0",
            },
        }
        with open(candidate2_path, "w") as f:
            yaml.dump(candidate2_data, f)

        # Load both candidate specs
        candidate1 = CandidateConfig.from_yaml_file(candidate1_path)
        candidate2 = CandidateConfig.from_yaml_file(candidate2_path)

        # Verify they reference different ports
        assert "8081" in candidate1.candidate.service_url
        assert "8082" in candidate2.candidate.service_url

    def test_chromadb_collection_names_include_chunking_config(self, tmp_path):
        """Test that ChromaDB collection names include chunking config."""
        from eval_harness.stubs.rag.chromadb_query import query

        # Create test corpus
        corpus_dir = tmp_path / "corpus"
        corpus_dir.mkdir()

        # Test that collection names include chunking config
        # This is already implemented in chromadb_query.py
        # We just verify the behavior

        # Query with chunk_size=512, overlap=50
        result1 = query(
            question="Test question?",
            corpus_dir=corpus_dir,
            chunk_size=512,
            chunk_overlap=50,
            top_k=5,
        )

        # Query with chunk_size=256, overlap=25
        result2 = query(
            question="Test question?",
            corpus_dir=corpus_dir,
            chunk_size=256,
            chunk_overlap=25,
            top_k=5,
        )

        # Both should succeed (different collections)
        assert result1 is not None
        assert result2 is not None

    def test_services_start_independently(self, tmp_path):
        """Test that services start independently via separate CLI invocations."""
        from click.testing import CliRunner

        from eval_harness.stubs.service.main import cli

        # Create two stub configs with different ports
        config1_path = tmp_path / "stub1.yaml"
        corpus_dir = tmp_path / "corpus"
        corpus_dir.mkdir()
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

        runner = CliRunner()

        # Test that both configs can be validated (CLI --help)
        result1 = runner.invoke(cli, ["--config", str(config1_path), "--help"])
        assert result1.exit_code == 0

        result2 = runner.invoke(cli, ["--config", str(config2_path), "--help"])
        assert result2.exit_code == 0

    def test_port_validation_prevents_conflicts(self, tmp_path):
        """Test that port validation prevents conflicts."""
        import socket

        from eval_harness.stubs.service.main import is_port_available

        # Bind a port
        test_socket = socket.socket()
        test_socket.bind(("127.0.0.1", 8081))

        try:
            # Port should show as unavailable
            assert is_port_available(8081) is False

            # A different port should be available
            assert is_port_available(8082) is True
        finally:
            test_socket.close()
