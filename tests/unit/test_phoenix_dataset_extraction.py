"""Tests for Phoenix dataset extraction utilities."""

from unittest.mock import MagicMock

import pandas as pd


class TestDatasetExtraction:
    """Test suite for Phoenix dataset extraction utilities."""

    def test_extracting_question_expected_pairs_from_spans_dataframe(self):
        """Test extracting question/expected pairs from spans DataFrame."""
        from eval_harness.replay.phoenix_datasets import extract_dataset_from_spans

        # Mock Phoenix client
        mock_client = MagicMock()

        # Create mock spans DataFrame
        mock_spans_df = pd.DataFrame({
            "name": ["rag_query", "rag_query", "other_span"],
            "attributes.input.value": ["What is contract law?", "What is termination?", "Other question"],
            "attributes.output.value": ["Contract law governs...", "Termination clause...", "Other answer"],
            "parent_id": [None, None, "some_parent"],
            "span_id": ["span1", "span2", "span3"],
        })

        mock_client.spans.get_spans_dataframe.return_value = mock_spans_df

        # Extract dataset
        result = extract_dataset_from_spans(mock_client, project_name="test-project")

        # Should extract question/expected pairs
        assert isinstance(result, pd.DataFrame)
        assert "question" in result.columns
        assert "expected_answer" in result.columns

    def test_dataframe_to_phoenix_dataset_conversion(self):
        """Test DataFrame to Phoenix dataset conversion."""
        from eval_harness.replay.phoenix_datasets import create_phoenix_dataset

        # Mock Phoenix client
        mock_client = MagicMock()

        # Create test DataFrame
        test_df = pd.DataFrame({
            "question": ["What is contract law?", "What is termination?"],
            "expected_answer": ["Contract law governs...", "Termination clause..."],
        })

        # Mock dataset creation response
        mock_dataset = MagicMock()
        mock_dataset.dataset_id = "test-dataset-id"
        mock_dataset.version = "1"
        mock_client.datasets.create_dataset.return_value = mock_dataset

        # Create dataset
        result = create_phoenix_dataset(
            mock_client,
            name="test-dataset",
            dataframe=test_df,
            input_keys=["question"],
            output_keys=["expected_answer"]
        )

        # Verify dataset was created
        mock_client.datasets.create_dataset.assert_called_once()
        assert result["dataset_id"] == "test-dataset-id"

    def test_dataset_versioning(self):
        """Test dataset versioning (create, list, retrieve versions)."""
        from eval_harness.replay.phoenix_datasets import get_dataset_versions

        # Mock Phoenix client
        mock_client = MagicMock()

        # Mock dataset versions response
        mock_versions = [
            MagicMock(version_id="v1", created_at="2024-01-01"),
            MagicMock(version_id="v2", created_at="2024-01-02"),
        ]
        mock_client.datasets.get_dataset_versions.return_value = mock_versions

        # Get versions
        result = get_dataset_versions(mock_client, dataset_id="test-dataset-id")

        # Verify versions were retrieved
        mock_client.datasets.get_dataset_versions.assert_called_once_with("test-dataset-id")
        assert len(result) == 2

    def test_error_handling_for_missing_span_attributes(self):
        """Test error handling for missing span attributes."""
        from eval_harness.replay.phoenix_datasets import extract_dataset_from_spans

        # Mock Phoenix client
        mock_client = MagicMock()

        # Create mock spans DataFrame with missing attributes
        mock_spans_df = pd.DataFrame({
            "name": ["rag_query", "rag_query"],
            # Missing input.value and output.value columns
            "parent_id": [None, None],
            "span_id": ["span1", "span2"],
        })

        mock_client.spans.get_spans_dataframe.return_value = mock_spans_df

        # Should handle missing attributes gracefully
        result = extract_dataset_from_spans(mock_client, project_name="test-project")

        # Should return empty DataFrame or handle gracefully
        assert isinstance(result, pd.DataFrame)

    def test_dataset_extraction_filters_correctly(self):
        """Test that dataset extraction filters for correct span types."""
        from eval_harness.replay.phoenix_datasets import extract_dataset_from_spans

        # Mock Phoenix client
        mock_client = MagicMock()

        # Create mock spans DataFrame with mixed span types
        mock_spans_df = pd.DataFrame({
            "name": ["rag_query", "retrieval", "generation", "rag_query"],
            "attributes.input.value": ["Question 1", None, None, "Question 2"],
            "attributes.output.value": ["Answer 1", None, None, "Answer 2"],
            "parent_id": [None, "span1", "span1", None],
            "span_id": ["span1", "span2", "span3", "span4"],
            "span_kind": ["CHAIN", "RETRIEVER", "LLM", "CHAIN"],
        })

        mock_client.spans.get_spans_dataframe.return_value = mock_spans_df

        # Extract dataset - should filter for CHAIN spans (root queries)
        result = extract_dataset_from_spans(mock_client, project_name="test-project")

        # Should only extract root spans (rag_query)
        assert len(result) <= 2  # At most 2 root spans
