"""Pytest fixtures for replay integration tests."""


import pytest
import yaml


@pytest.fixture
def sample_candidate_config(tmp_path):
    """
    Create a sample candidate config YAML for testing.

    Returns:
        Path to the candidate config YAML file.

    """
    candidate_path = tmp_path / "candidate.yaml"
    candidate_data = {
        "name": "test-candidate",
        "description": "Test candidate for HTTP evaluation",
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

    return candidate_path


@pytest.fixture
def sample_stub_config(tmp_path):
    """
    Create a sample stub config YAML for testing.

    Returns:
        Path to the stub config YAML file.

    """
    stub_path = tmp_path / "stub.yaml"
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()

    stub_data = {
        "chunking_strategy": "fixed",
        "chunk_size": 512,
        "chunk_overlap": 50,
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "port": 8081,
        "corpus_path": str(corpus_dir),
    }
    with open(stub_path, "w") as f:
        yaml.dump(stub_data, f)

    return stub_path


@pytest.fixture
def available_port():
    """
    Find and return an available port for testing.

    Returns:
        Available port number.

    """
    import socket

    sock = socket.socket()
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port
