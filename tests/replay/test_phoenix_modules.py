"""
Tests for Phoenix replay modules.
"""


import pytest

from eval_harness.replay.phoenix_client import PhoenixClient


class TestPhoenixClient:
    """Tests for PhoenixClient."""

    def test_phoenix_client_initialization_with_mock(self) -> None:
        """Test PhoenixClient initialization with mock."""
        # Test with no actual Phoenix connection
        client = PhoenixClient(
            endpoint="http://localhost:6006",
            project_name="test-project",
        )

        assert client._endpoint == "http://localhost:6006"
        assert client._project_name == "test-project"
        # Without actual Phoenix, _client will be None

    def test_phoenix_client_without_connection(self) -> None:
        """Test PhoenixClient handles no connection gracefully."""
        client = PhoenixClient()

        assert client._client is None
        assert not client.is_connected()

    def test_query_root_spans_when_disconnected(self) -> None:
        """Test query_root_spans raises error when disconnected."""
        client = PhoenixClient()
        # Simulate disconnected state
        client._client = None

        with pytest.raises(ConnectionError):
            client.query_root_spans()
