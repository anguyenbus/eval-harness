"""
Tests for Phoenix dataset management CLI commands.

PHOENIX NATIVE MIGRATION: Phase 2.3 - CLI Dataset Management Commands
Tests for dataset list, upload, download, validate commands.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner


class TestDatasetListCommand:
    """Tests for eval-dataset list command."""

    def test_dataset_list_help(self) -> None:
        """Test that dataset list --help displays help text."""
        from eval_harness.cli.dataset import dataset

        runner = CliRunner()
        result = runner.invoke(dataset, ["list", "--help"])

        assert result.exit_code == 0
        assert "list" in result.output.lower()

    def test_dataset_list_no_datasets(self) -> None:
        """Test dataset list when no datasets exist."""
        from eval_harness.cli.dataset import dataset

        runner = CliRunner()

        with patch("eval_harness.cli.dataset.PhoenixClientWithDatasets") as mock_client_class:
            mock_instance = MagicMock()
            mock_instance.is_connected.return_value = True
            mock_instance.list_dataset_versions.return_value = []
            mock_client_class.return_value = mock_instance

            result = runner.invoke(dataset, ["list"])

            assert result.exit_code == 0
            assert "phoenix dataset management" in result.output.lower()

    def test_dataset_list_with_datasets_by_id(self) -> None:
        """Test dataset list with existing datasets using --dataset-id."""
        from eval_harness.cli.dataset import dataset

        runner = CliRunner()

        with patch("eval_harness.cli.dataset.PhoenixClientWithDatasets") as mock_client_class:
            mock_instance = MagicMock()
            mock_instance.is_connected.return_value = True
            mock_instance.list_dataset_versions.return_value = [
                {"version_id": "v1", "created_at": "2025-01-01T00:00:00Z"},
            ]
            mock_client_class.return_value = mock_instance

            result = runner.invoke(dataset, ["list", "--dataset-id", "test-dataset-id"])

            assert result.exit_code == 0
            assert "v1" in result.output
            mock_instance.list_dataset_versions.assert_called_once_with("test-dataset-id")

    def test_dataset_list_connection_error(self) -> None:
        """Test dataset list when Phoenix is not connected."""
        from eval_harness.cli.dataset import dataset

        runner = CliRunner()

        with patch("eval_harness.cli.dataset.PhoenixClientWithDatasets") as mock_client_class:
            mock_instance = MagicMock()
            mock_instance.is_connected.return_value = False
            mock_client_class.return_value = mock_instance

            result = runner.invoke(dataset, ["list"])

            assert result.exit_code == 1
            assert "not connected" in result.output.lower()

    def test_dataset_list_custom_endpoint(self) -> None:
        """Test dataset list with custom Phoenix endpoint."""
        from eval_harness.cli.dataset import dataset

        runner = CliRunner()

        with patch("eval_harness.cli.dataset.PhoenixClientWithDatasets") as mock_client_class:
            mock_instance = MagicMock()
            mock_instance.is_connected.return_value = True
            mock_instance.list_dataset_versions.return_value = []
            mock_client_class.return_value = mock_instance

            # Group options must come before the subcommand
            result = runner.invoke(
                dataset, ["--endpoint", "http://custom:6006", "list"]
            )

            # Just check that it runs successfully
            assert result.exit_code == 0


class TestDatasetUploadCommand:
    """Tests for eval-dataset upload command."""

    def test_dataset_upload_help(self) -> None:
        """Test that dataset upload --help displays help text."""
        from eval_harness.cli.dataset import dataset

        runner = CliRunner()
        result = runner.invoke(dataset, ["upload", "--help"])

        assert result.exit_code == 0
        assert "upload" in result.output.lower()

    def test_dataset_upload_from_spans(self) -> None:
        """Test dataset upload extracted from spans."""
        from eval_harness.cli.dataset import dataset

        runner = CliRunner()

        with patch("eval_harness.cli.dataset.PhoenixClientWithDatasets") as mock_client_class:
            with patch("eval_harness.cli.dataset.extract_dataset_from_spans") as mock_extract:
                with patch("eval_harness.cli.dataset.create_phoenix_dataset") as mock_create:
                    mock_instance = MagicMock()
                    mock_instance.is_connected.return_value = True
                    mock_instance._client = MagicMock()
                    mock_client_class.return_value = mock_instance

                    import pandas as pd

                    mock_df = pd.DataFrame({
                        "question": ["What is contract law?"],
                        "expected_answer": ["Contract law governs..."],
                    })
                    mock_extract.return_value = mock_df

                    mock_create.return_value = {
                        "dataset_id": "test-dataset-id",
                        "version": "1",
                    }

                    result = runner.invoke(
                        dataset, ["upload", "--name", "test-dataset", "--from-spans"]
                    )

                    assert result.exit_code == 0
                    assert "test-dataset-id" in result.output
                    mock_extract.assert_called_once()
                    mock_create.assert_called_once()

    def test_dataset_upload_from_file(self) -> None:
        """Test dataset upload from CSV file."""
        from eval_harness.cli.dataset import dataset

        runner = CliRunner()

        with runner.isolated_filesystem():
            # Create test CSV file
            import pandas as pd

            test_df = pd.DataFrame({
                "question": ["What is contract law?"],
                "expected_answer": ["Contract law governs..."],
            })
            test_df.to_csv("test_dataset.csv", index=False)

            with patch("eval_harness.cli.dataset.PhoenixClientWithDatasets") as mock_client_class:
                with patch("eval_harness.cli.dataset.create_phoenix_dataset") as mock_create:
                    mock_instance = MagicMock()
                    mock_instance.is_connected.return_value = True
                    mock_instance._client = MagicMock()
                    mock_client_class.return_value = mock_instance

                    mock_create.return_value = {
                        "dataset_id": "test-dataset-id",
                        "version": "1",
                    }

                    result = runner.invoke(
                        dataset, ["upload", "--name", "test-dataset", "--file", "test_dataset.csv"]
                    )

                    assert result.exit_code == 0
                    assert "test-dataset-id" in result.output

    def test_dataset_upload_missing_file(self) -> None:
        """Test dataset upload with missing file."""
        from eval_harness.cli.dataset import dataset

        runner = CliRunner()

        result = runner.invoke(
            dataset, ["upload", "--name", "test", "--file", "missing.csv"]
        )

        # Click validates path exists before our code runs
        assert result.exit_code != 0
        assert "does not exist" in result.output.lower()

    def test_dataset_upload_requires_name(self) -> None:
        """Test dataset upload requires --name option."""
        from eval_harness.cli.dataset import dataset

        runner = CliRunner()

        result = runner.invoke(dataset, ["upload", "--from-spans"])

        assert result.exit_code != 0
        # Click provides error for missing required option

    def test_dataset_upload_requires_source(self) -> None:
        """Test dataset upload requires --from-spans or --file."""
        from eval_harness.cli.dataset import dataset

        runner = CliRunner()

        result = runner.invoke(dataset, ["upload", "--name", "test"])

        assert result.exit_code != 0
        assert ("must specify" in result.output.lower() or "from-spans" in result.output.lower())


class TestDatasetDownloadCommand:
    """Tests for eval-dataset download command."""

    def test_dataset_download_help(self) -> None:
        """Test that dataset download --help displays help text."""
        from eval_harness.cli.dataset import dataset

        runner = CliRunner()
        result = runner.invoke(dataset, ["download", "--help"])

        assert result.exit_code == 0
        assert "download" in result.output.lower()

    def test_dataset_download_to_csv(self) -> None:
        """Test dataset download to CSV file."""
        from eval_harness.cli.dataset import dataset

        runner = CliRunner()

        with runner.isolated_filesystem():
            with patch("eval_harness.cli.dataset.PhoenixClientWithDatasets") as mock_client_class:
                import pandas as pd

                mock_df = pd.DataFrame({
                    "question": ["What is contract law?"],
                    "expected_answer": ["Contract law governs..."],
                })

                mock_instance = MagicMock()
                mock_instance.is_connected.return_value = True
                mock_instance.get_dataset.return_value = mock_df
                mock_client_class.return_value = mock_instance

                result = runner.invoke(
                    dataset, ["download", "--dataset-id", "test-id", "--output", "output.csv"]
                )

                assert result.exit_code == 0
                assert Path("output.csv").exists()

    def test_dataset_download_requires_dataset_id(self) -> None:
        """Test dataset download requires --dataset-id option."""
        from eval_harness.cli.dataset import dataset

        runner = CliRunner()

        result = runner.invoke(dataset, ["download", "--output", "output.csv"])

        assert result.exit_code != 0
        # Click provides error for missing required option

    def test_dataset_download_connection_error(self) -> None:
        """Test dataset download when Phoenix is not connected."""
        from eval_harness.cli.dataset import dataset

        runner = CliRunner()

        with patch("eval_harness.cli.dataset.PhoenixClientWithDatasets") as mock_client_class:
            mock_instance = MagicMock()
            mock_instance.is_connected.return_value = False
            mock_client_class.return_value = mock_instance

            result = runner.invoke(
                dataset, ["download", "--dataset-id", "test-id", "--output", "output.csv"]
            )

            assert result.exit_code == 1
            assert "not connected" in result.output.lower()


class TestDatasetValidateCommand:
    """Tests for eval-dataset validate command."""

    def test_dataset_validate_help(self) -> None:
        """Test that dataset validate --help displays help text."""
        from eval_harness.cli.dataset import dataset

        runner = CliRunner()
        result = runner.invoke(dataset, ["validate", "--help"])

        assert result.exit_code == 0
        assert "validate" in result.output.lower()

    def test_dataset_validate_valid_file(self) -> None:
        """Test dataset validate with valid CSV file."""
        from eval_harness.cli.dataset import dataset

        runner = CliRunner()

        with runner.isolated_filesystem():
            # Create valid CSV file with required columns
            import pandas as pd

            test_df = pd.DataFrame({
                "question": ["What is contract law?"],
                "expected_answer": ["Contract law governs..."],
            })
            test_df.to_csv("valid_dataset.csv", index=False)

            result = runner.invoke(dataset, ["validate", "--file", "valid_dataset.csv"])

            assert result.exit_code == 0
            assert "passed" in result.output.lower() or "success" in result.output.lower()

    def test_dataset_validate_missing_columns(self) -> None:
        """Test dataset validate with missing required columns."""
        from eval_harness.cli.dataset import dataset

        runner = CliRunner()

        with runner.isolated_filesystem():
            # Create CSV file missing expected_answer column
            import pandas as pd

            test_df = pd.DataFrame({
                "question": ["What is contract law?"],
            })
            test_df.to_csv("invalid_dataset.csv", index=False)

            result = runner.invoke(dataset, ["validate", "--file", "invalid_dataset.csv"])

            assert result.exit_code != 0
            assert "failed" in result.output.lower() or "missing" in result.output.lower()

    def test_dataset_validate_empty_values(self) -> None:
        """Test dataset validate with empty values."""
        from eval_harness.cli.dataset import dataset

        runner = CliRunner()

        with runner.isolated_filesystem():
            # Create CSV file with empty values
            import pandas as pd

            test_df = pd.DataFrame({
                "question": ["What is contract law?", ""],
                "expected_answer": ["Contract law governs...", "Test answer"],
            })
            test_df.to_csv("partial_empty.csv", index=False)

            result = runner.invoke(dataset, ["validate", "--file", "partial_empty.csv"])

            # Should warn about empty values but may pass validation
            assert result.exit_code in [0, 1]
            assert ("empty" in result.output.lower() or "warning" in result.output.lower() or
                    "passed" in result.output.lower() or "success" in result.output.lower())

    def test_dataset_validate_missing_file(self) -> None:
        """Test dataset validate with missing file."""
        from eval_harness.cli.dataset import dataset

        runner = CliRunner()

        result = runner.invoke(dataset, ["validate", "--file", "missing.csv"])

        # Click validates path exists
        assert result.exit_code != 0
        assert "does not exist" in result.output.lower()

    def test_dataset_validate_custom_schema(self) -> None:
        """Test dataset validate with custom schema columns."""
        from eval_harness.cli.dataset import dataset

        runner = CliRunner()

        with runner.isolated_filesystem():
            # Create CSV file with custom columns
            import pandas as pd

            test_df = pd.DataFrame({
                "custom_question": ["What is contract law?"],
                "custom_answer": ["Contract law governs..."],
            })
            test_df.to_csv("custom_dataset.csv", index=False)

            result = runner.invoke(
                dataset,
                ["validate", "--file", "custom_dataset.csv",
                 "--input-keys", "custom_question",
                 "--output-keys", "custom_answer"]
            )

            assert result.exit_code == 0
            assert "passed" in result.output.lower() or "success" in result.output.lower()
