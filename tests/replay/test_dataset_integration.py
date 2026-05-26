"""
Integration tests for Phoenix dataset management.

PHOENIX NATIVE MIGRATION: Phase 2.4 - Phase 2 Integration Testing
Tests for end-to-end dataset extraction, upload, download, and validation.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd


class TestDatasetExtractionIntegration:
    """Tests for end-to-end dataset extraction and upload."""

    def test_extract_and_create_dataset(self) -> None:
        """Test extracting spans and creating a Phoenix dataset."""
        from eval_harness.replay.phoenix_datasets import (
            create_phoenix_dataset,
            extract_dataset_from_spans,
        )

        # Mock Phoenix client
        mock_client = MagicMock()

        # Mock spans DataFrame
        mock_spans_df = pd.DataFrame({
            "parent_id": [None, None],
            "name": ["rag_query", "rag_query"],
            "attributes.input.value": ["What is contract law?", "What is tort law?"],
            "attributes.output.value": [
                "Contract law governs agreements...",
                "Tort law deals with civil wrongs...",
            ],
            "span_id": ["span1", "span2"],
        })
        mock_client.spans.get_spans_dataframe.return_value = mock_spans_df

        # Extract dataset
        extracted_df = extract_dataset_from_spans(
            client=mock_client,
            project_name="test-project",
            span_name="rag_query",
        )

        assert len(extracted_df) == 2
        assert "question" in extracted_df.columns
        assert "expected_answer" in extracted_df.columns
        assert extracted_df["question"].iloc[0] == "What is contract law?"

        # Mock dataset creation
        mock_dataset = MagicMock()
        mock_dataset.dataset_id = "test-dataset-id"
        mock_dataset.version = "1"
        mock_client.datasets.create_dataset.return_value = mock_dataset

        # Create Phoenix dataset
        result = create_phoenix_dataset(
            client=mock_client,
            name="test-dataset",
            dataframe=extracted_df,
            input_keys=["question"],
            output_keys=["expected_answer"],
        )

        assert result["dataset_id"] == "test-dataset-id"
        assert result["version"] == "1"
        mock_client.datasets.create_dataset.assert_called_once()

    def test_extract_dataset_with_missing_attributes(self) -> None:
        """Test extracting dataset when some spans have missing attributes."""
        from eval_harness.replay.phoenix_datasets import extract_dataset_from_spans

        mock_client = MagicMock()

        # Mock spans DataFrame with some missing attributes
        mock_spans_df = pd.DataFrame({
            "parent_id": [None, None, None],
            "name": ["rag_query", "rag_query", "rag_query"],
            "attributes.input.value": ["Question 1", "Question 2", None],
            "attributes.output.value": ["Answer 1", None, "Answer 3"],
            "span_id": ["span1", "span2", "span3"],
        })
        mock_client.spans.get_spans_dataframe.return_value = mock_spans_df

        extracted_df = extract_dataset_from_spans(
            client=mock_client,
            project_name="test-project",
        )

        # Should only include spans with both question and answer
        assert len(extracted_df) == 1
        assert extracted_df["question"].iloc[0] == "Question 1"

    def test_extract_dataset_empty_project(self) -> None:
        """Test extracting dataset from project with no spans."""
        from eval_harness.replay.phoenix_datasets import extract_dataset_from_spans

        mock_client = MagicMock()

        # Mock empty spans DataFrame
        mock_spans_df = pd.DataFrame({
            "parent_id": [],
            "name": [],
            "attributes.input.value": [],
            "attributes.output.value": [],
            "span_id": [],
        })
        mock_client.spans.get_spans_dataframe.return_value = mock_spans_df

        extracted_df = extract_dataset_from_spans(
            client=mock_client,
            project_name="empty-project",
        )

        assert len(extracted_df) == 0
        assert "question" in extracted_df.columns
        assert "expected_answer" in extracted_df.columns


class TestDatasetRetrievalIntegration:
    """Tests for dataset retrieval and versioning."""

    def test_get_dataset_versions(self) -> None:
        """Test retrieving dataset versions."""
        from eval_harness.replay.phoenix_datasets import get_dataset_versions

        mock_client = MagicMock()

        # Mock version objects
        mock_v1 = MagicMock()
        mock_v1.version_id = "v1"
        mock_v1.created_at = "2025-01-01T00:00:00Z"

        mock_v2 = MagicMock()
        mock_v2.version_id = "v2"
        mock_v2.created_at = "2025-01-02T00:00:00Z"

        mock_client.datasets.get_dataset_versions.return_value = [mock_v1, mock_v2]

        versions = get_dataset_versions(mock_client, "test-dataset-id")

        assert len(versions) == 2
        assert versions[0]["version_id"] == "v1"
        assert versions[1]["version_id"] == "v2"
        mock_client.datasets.get_dataset_versions.assert_called_once_with("test-dataset-id")

    def test_get_dataset_by_id(self) -> None:
        """Test retrieving a specific dataset."""
        from eval_harness.replay.phoenix_datasets import get_dataset

        mock_client = MagicMock()

        # Mock dataset object
        mock_dataset_df = pd.DataFrame({
            "question": ["Test question"],
            "expected_answer": ["Test answer"],
        })
        mock_client.datasets.get_dataset.return_value = mock_dataset_df

        dataset = get_dataset(mock_client, "test-dataset-id", version="v1")

        assert dataset is not None
        assert len(dataset) == 1
        mock_client.datasets.get_dataset.assert_called_once_with("test-dataset-id", version="v1")


class TestDatasetValidationIntegration:
    """Tests for dataset validation and schema checking."""

    def test_validate_dataset_schema(self) -> None:
        """Test validating dataset has correct schema."""
        from click.testing import CliRunner

        from eval_harness.cli.dataset import validate_dataset

        runner = CliRunner()

        with runner.isolated_filesystem():
            # Create valid dataset
            df = pd.DataFrame({
                "question": ["Q1", "Q2"],
                "expected_answer": ["A1", "A2"],
            })
            df.to_csv("valid.csv", index=False)

            result = runner.invoke(
                validate_dataset,
                ["--file", "valid.csv"]
            )

            assert result.exit_code == 0
            assert "passed" in result.output.lower() or "success" in result.output.lower()

    def test_validate_dataset_with_custom_schema(self) -> None:
        """Test validating dataset with custom input/output keys."""
        from click.testing import CliRunner

        from eval_harness.cli.dataset import validate_dataset

        runner = CliRunner()

        with runner.isolated_filesystem():
            # Create dataset with custom column names
            df = pd.DataFrame({
                "custom_question": ["Q1"],
                "custom_answer": ["A1"],
            })
            df.to_csv("custom.csv", index=False)

            result = runner.invoke(
                validate_dataset,
                ["--file", "custom.csv",
                 "--input-keys", "custom_question",
                 "--output-keys", "custom_answer"]
            )

            assert result.exit_code == 0

    def test_validate_dataset_detects_missing_columns(self) -> None:
        """Test validation detects missing required columns."""
        from click.testing import CliRunner

        from eval_harness.cli.dataset import validate_dataset

        runner = CliRunner()

        with runner.isolated_filesystem():
            # Create dataset missing expected_answer
            df = pd.DataFrame({
                "question": ["Q1", "Q2"],
            })
            df.to_csv("missing_column.csv", index=False)

            result = runner.invoke(
                validate_dataset,
                ["--file", "missing_column.csv"]
            )

            assert result.exit_code != 0
            assert "failed" in result.output.lower() or "missing" in result.output.lower()


class TestPhoenixClientDatasetsIntegration:
    """Tests for PhoenixClientWithDatasets integration."""

    def test_client_create_and_retrieve_dataset(self) -> None:
        """Test creating and retrieving dataset through PhoenixClientWithDatasets."""
        from eval_harness.replay.phoenix_client_datasets import (
            PhoenixClientWithDatasets,
        )

        # Mock base client
        mock_base_client = MagicMock()

        # Mock dataset creation response
        mock_dataset = MagicMock()
        mock_dataset.dataset_id = "test-id"
        mock_dataset.version = "1"
        mock_base_client.datasets.create_dataset.return_value = mock_dataset

        client = PhoenixClientWithDatasets(
            endpoint="http://localhost:6006",
            base_client=mock_base_client
        )

        df = pd.DataFrame({
            "question": ["Test"],
            "expected_answer": ["Answer"],
        })

        result = client.create_dataset(
            name="test-dataset",
            dataframe=df,
            input_keys=["question"],
            output_keys=["expected_answer"],
        )

        assert result["dataset_id"] == "test-id"

        # Mock retrieval
        mock_base_client.datasets.get_dataset.return_value = df
        retrieved = client.get_dataset("test-id")

        assert retrieved is not None
        mock_base_client.datasets.get_dataset.assert_called_once_with("test-id")

    def test_client_list_dataset_versions(self) -> None:
        """Test listing dataset versions through PhoenixClientWithDatasets."""
        from eval_harness.replay.phoenix_client_datasets import (
            PhoenixClientWithDatasets,
        )

        mock_base_client = MagicMock()

        # Mock versions
        mock_v1 = MagicMock()
        mock_v1.version_id = "v1"
        mock_v1.created_at = "2025-01-01T00:00:00Z"

        mock_base_client.datasets.get_dataset_versions.return_value = [mock_v1]

        client = PhoenixClientWithDatasets(
            endpoint="http://localhost:6006",
            base_client=mock_base_client
        )

        versions = client.list_dataset_versions("test-id")

        assert len(versions) == 1
        assert versions[0]["version_id"] == "v1"
        mock_base_client.datasets.get_dataset_versions.assert_called_once_with("test-id")
