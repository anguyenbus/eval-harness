"""
Tests for Phoenix dataset management.
"""

from pathlib import Path

import pytest


def test_create_phoenix_dataset_validates_corpus_dir():
    """Test that create_phoenix_dataset validates corpus directory exists."""
    from eval_harness.experiments.datasets import create_phoenix_dataset

    # Skip if Phoenix is not installed
    pytest.importorskip("phoenix.client")

    from phoenix.client import Client

    # Create a real client (will fail to connect but that's ok for this test)
    try:
        mock_client = Client(endpoint="http://localhost:9999")  # Non-existent endpoint
    except Exception:
        pytest.skip("Could not create Phoenix client")

    with pytest.raises(ValueError, match="Corpus directory does not exist"):
        create_phoenix_dataset(
            client=mock_client,
            corpus_dir=Path("/nonexistent/path"),
        )


@pytest.mark.skipif("os.environ.get('PHOENIX_ENABLED') != '1'")
def test_get_phoenix_dataset_returns_none_when_unavailable():
    """Test that get_phoenix_dataset returns None when Phoenix is unavailable."""
    from eval_harness.experiments.datasets import get_phoenix_dataset

    result = get_phoenix_dataset(client=None, slice_name="nano")
    assert result is None
