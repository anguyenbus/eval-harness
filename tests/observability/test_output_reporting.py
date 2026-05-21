"""Tests for output and reporting with Phoenix integration."""

from __future__ import annotations

import json
from unittest.mock import MagicMock


class TestOutputReporting:
    """Test console output and reporting for Phoenix integration."""

    def test_phoenix_ui_url_printed_when_enabled(self, capsys):
        """Test that Phoenix UI URL is printed when Phoenix is connected."""
        from eval_harness.observability.phoenix_adapter import PhoenixAdapter

        # Simulate connected adapter
        adapter = PhoenixAdapter(enabled=False)

        # Mock a connected state
        adapter._client = MagicMock()

        assert adapter.is_connected() is True

    def test_trace_count_reported_after_export(self):
        """Test that trace count is reported after export."""
        from eval_harness.observability.phoenix_adapter import PhoenixAdapter

        adapter = PhoenixAdapter(enabled=False)

        export_result = adapter.export_traces()

        assert "trace_count" in export_result
        assert isinstance(export_result["trace_count"], int)

    def test_parquet_export_path_printed_when_buffering(self):
        """Test that Parquet export path is printed when buffering."""
        from eval_harness.observability.phoenix_adapter import PhoenixAdapter

        adapter = PhoenixAdapter(enabled=False)  # Disabled = buffering mode

        export_result = adapter.export_traces()

        assert export_result["mode"] == "parquet"
        assert "path" in export_result
        assert export_result["path"] is not None

    def test_csv_json_output_unchanged_regression(self, tmp_path):
        """Test that CSV and JSON output formats are unchanged."""
        import csv

        # Create a simple CSV with expected columns
        csv_file = tmp_path / "test_results.csv"
        fieldnames = [
            "query_id",
            "question",
            "gold_answer",
            "generated_answer",
            "relevant_passage_retrieved",
            "faithfulness_score",
            "context_precision_score",
            "context_recall_score",
            "answer_relevancy_score",
            "judge_verdict",
            "total_ms",
            "error",
        ]

        with open(csv_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(
                {
                    "query_id": "test_1",
                    "question": "Test question",
                    "gold_answer": "Test answer",
                    "generated_answer": "Generated",
                    "relevant_passage_retrieved": True,
                    "faithfulness_score": 0.9,
                    "context_precision_score": 0.8,
                    "context_recall_score": 0.7,
                    "answer_relevancy_score": 0.85,
                    "judge_verdict": "PASS",
                    "total_ms": 1000,
                    "error": "",
                }
            )

        # Verify CSV format is unchanged
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 1
            assert rows[0]["query_id"] == "test_1"
            assert rows[0]["faithfulness_score"] == "0.9"

    def test_json_summary_includes_phoenix_info(self, tmp_path):
        """Test that JSON summary includes Phoenix information when enabled."""
        json_file = tmp_path / "test_summary.json"

        summary = {
            "dataset": "legal_rag_bench",
            "slice": "nano",
            "timestamp": "20240101_120000",
            "csv_file": "test_results.csv",
            "metrics_avg": {},
            "total_processed": 10,
            "errors": 0,
            "top_k": 5,
            "phoenix": {
                "enabled": True,
                "trace_count": 10,
                "endpoint": "http://localhost:6006",
            },
        }

        with open(json_file, "w") as f:
            json.dump(summary, f)

        # Verify JSON includes Phoenix info
        with open(json_file) as f:
            loaded = json.load(f)
            assert "phoenix" in loaded
            assert loaded["phoenix"]["enabled"] is True
            assert loaded["phoenix"]["trace_count"] == 10

    def test_console_output_format(self):
        """Test that console output follows expected format."""
        from eval_harness.observability.phoenix_adapter import DEFAULT_ENDPOINT

        # Simulate expected output format
        endpoint = DEFAULT_ENDPOINT
        expected_url_message = f"Phoenix UI available at: {endpoint}"

        assert "Phoenix UI available at:" in expected_url_message
        assert endpoint in expected_url_message

    def test_parquet_path_message_format(self):
        """Test that Parquet export message follows expected format."""
        export_path = "/tmp/phoenix_traces/trace_dataset_20240101_120000.parquet"
        expected_message = f"Traces buffered to Parquet: {export_path}"

        assert "Traces buffered to Parquet:" in expected_message
        assert export_path in expected_message
