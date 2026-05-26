"""Tests for Phoenix Client Datasets API extension."""

from unittest.mock import MagicMock

import pandas as pd


class TestPhoenixClientDatasetsAPI:
    """Test suite for PhoenixClient datasets API integration."""

    def test_dataset_creation_through_phoenix_client(self):
        """Test dataset creation through PhoenixClient."""
        from eval_harness.replay.phoenix_client_datasets import (
            PhoenixClientWithDatasets,
        )

        # Mock base Phoenix client
        mock_base_client = MagicMock()
        mock_dataset = MagicMock()
        mock_dataset.dataset_id = "test-dataset-id"
        mock_base_client.datasets.create_dataset.return_value = mock_dataset

        # Create client with datasets support
        client = PhoenixClientWithDatasets(
            endpoint="http://localhost:6006",
            base_client=mock_base_client
        )

        # Test dataset creation
        test_df = pd.DataFrame({
            "question": ["What is contract law?"],
            "expected_answer": ["Contract law governs..."]
        })

        result = client.create_dataset(
            name="test-dataset",
            dataframe=test_df,
            input_keys=["question"],
            output_keys=["expected_answer"]
        )

        # Verify dataset was created
        assert result["dataset_id"] == "test-dataset-id"

    def test_dataset_retrieval(self):
        """Test dataset retrieval through PhoenixClient."""
        from eval_harness.replay.phoenix_client_datasets import (
            PhoenixClientWithDatasets,
        )

        # Mock base Phoenix client
        mock_base_client = MagicMock()
        mock_dataset = MagicMock()
        mock_dataset.dataset_id = "test-dataset-id"
        mock_base_client.datasets.get_dataset.return_value = mock_dataset

        # Create client with datasets support
        client = PhoenixClientWithDatasets(
            endpoint="http://localhost:6006",
            base_client=mock_base_client
        )

        # Test dataset retrieval
        result = client.get_dataset(dataset_id="test-dataset-id")

        # Verify dataset was retrieved
        mock_base_client.datasets.get_dataset.assert_called_once_with("test-dataset-id")

    def test_dataset_version_listing(self):
        """Test dataset version listing through PhoenixClient."""
        from eval_harness.replay.phoenix_client_datasets import (
            PhoenixClientWithDatasets,
        )

        # Mock base Phoenix client
        mock_base_client = MagicMock()
        mock_versions = [
            MagicMock(version_id="v1", created_at="2024-01-01"),
            MagicMock(version_id="v2", created_at="2024-01-02"),
        ]
        mock_base_client.datasets.get_dataset_versions.return_value = mock_versions

        # Create client with datasets support
        client = PhoenixClientWithDatasets(
            endpoint="http://localhost:6006",
            base_client=mock_base_client
        )

        # Test version listing
        result = client.list_dataset_versions(dataset_id="test-dataset-id")

        # Verify versions were listed
        assert len(result) == 2

    def test_error_handling_consistent_with_existing_patterns(self):
        """Test error handling matches existing PhoenixClient patterns."""
        from eval_harness.replay.phoenix_client_datasets import (
            PhoenixClientWithDatasets,
        )

        # Mock base Phoenix client that raises exception
        mock_base_client = MagicMock()
        mock_base_client.datasets.create_dataset.side_effect = Exception("Connection error")

        # Create client with datasets support
        client = PhoenixClientWithDatasets(
            endpoint="http://localhost:6006",
            base_client=mock_base_client
        )

        # Test error handling - should return empty/error dict like existing patterns
        test_df = pd.DataFrame({
            "question": ["What is contract law?"],
            "expected_answer": ["Contract law governs..."]
        })

        result = client.create_dataset(
            name="test-dataset",
            dataframe=test_df,
            input_keys=["question"],
            output_keys=["expected_answer"]
        )

        # Should handle error gracefully
        assert "error" in result or result["dataset_id"] is None
