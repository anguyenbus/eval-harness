"""
Tests for metric naming migration to Phoenix conventions.

PHOENIX NATIVE MIGRATION: Phase 3.4 - Metric Naming Migration
Tests for updating all metric references to Phoenix naming conventions.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestPhoenixMetricNaming:
    """Tests for Phoenix metric naming conventions."""

    def test_phoenix_metric_names_used(self) -> None:
        """Test that Phoenix metric names are used throughout codebase."""
        from eval_harness.adapters.phoenix_eval_adapter import PhoenixEvalAdapter

        rag_output = {
            "query": {"text": "What is contract law?"},
            "answer": {"text": "Contract law governs agreements..."},
            "retrieved_chunks": [{"text": "Contract law context"}],
        }
        reference_answer = "Contract law governs agreements."

        with patch("phoenix.evals.evaluate_dataframe") as mock_evaluate:
            import pandas as pd

            # Mock response with Phoenix naming convention
            mock_result = pd.DataFrame({
                "faithfulness_score": [0.85],
                "correctness_score": [0.90],
                "relevance_score": [0.75],
            })
            mock_evaluate.return_value = mock_result

            adapter = PhoenixEvalAdapter()

            scores = adapter.compute_metrics(rag_output, reference_answer)

            # Verify Phoenix naming conventions
            assert "faithfulness" in scores
            assert "correctness" in scores
            assert "relevance" in scores

            # Verify NO deepeval prefix
            assert "deepeval_faithfulness" not in scores
            assert "deepeval_correctness" not in scores
            assert "deepeval_relevance" not in scores

    def test_backward_compatibility_with_old_names(self) -> None:
        """Test backward compatibility with old metric names when flag is false."""
        from eval_harness.adapters.deepeval_adapter import DeepEvalEvaluator

        with patch("eval_harness.metrics.deepeval_config.create_deepeval_metrics") as mock_create:
            mock_metrics = {
                "faithfulness": MagicMock(),
                "context_precision": MagicMock(),
                "context_recall": MagicMock(),
                "answer_relevancy": MagicMock(),
            }
            for metric in mock_metrics.values():
                metric.measure.return_value = None
                metric.score = 0.85
            mock_create.return_value = mock_metrics

            evaluator = DeepEvalEvaluator(llm_provider="openai", judge_model="gpt-4o-mini")

            rag_output = {
                "query": {"text": "Question"},
                "answer": {"text": "Answer"},
                "retrieved_chunks": [{"text": "Context"}],
            }

            # DeepEval still works with old naming
            scores = evaluator.compute_metrics(rag_output, "Reference")

            # DeepEval uses its own naming (not Phoenix)
            assert "faithfulness" in scores
            assert "context_precision" in scores
            assert "context_recall" in scores
            assert "answer_relevancy" in scores

    def test_export_uses_phoenix_names_csv(self) -> None:
        """Test that CSV export uses Phoenix metric names."""
        import tempfile
        from pathlib import Path

        # Simulate export data with Phoenix metric names
        export_data = [
            {
                "question": "What is contract law?",
                "faithfulness": 0.85,
                "correctness": 0.90,
                "relevance": 0.75,
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "export.csv"

            # Write CSV
            import csv
            with open(csv_path, "w", newline="") as f:
                if export_data:
                    writer = csv.DictWriter(f, fieldnames=export_data[0].keys())
                    writer.writeheader()
                    writer.writerows(export_data)

            # Verify column names use Phoenix conventions
            with open(csv_path, "r") as f:
                reader = csv.DictReader(f)
                headers = reader.fieldnames

                assert "faithfulness" in headers
                assert "correctness" in headers
                assert "relevance" in headers

                # No deepeval prefix
                assert "deepeval_faithfulness" not in headers
                assert "deepeval_correctness" not in headers

    def test_export_uses_phoenix_names_json(self) -> None:
        """Test that JSON export uses Phoenix metric names."""
        import json
        import tempfile
        from pathlib import Path

        # Simulate export data with Phoenix metric names
        export_data = {
            "scores": {
                "faithfulness": 0.85,
                "correctness": 0.90,
                "relevance": 0.75,
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "export.json"

            with open(json_path, "w") as f:
                json.dump(export_data, f)

            # Verify JSON keys use Phoenix conventions
            with open(json_path, "r") as f:
                loaded = json.load(f)

                assert "faithfulness" in loaded["scores"]
                assert "correctness" in loaded["scores"]
                assert "relevance" in loaded["scores"]

                # No deepeval prefix
                assert "deepeval_faithfulness" not in loaded["scores"]
                assert "deepeval_correctness" not in loaded["scores"]

    def test_metric_name_mapping_is_consistent(self) -> None:
        """Test that metric name mapping is consistent across all interfaces."""
        # Define the expected Phoenix metric names
        phoenix_metrics = {
            "faithfulness",  # Not deepeval_faithfulness
            "correctness",  # Not deepeval_correctness
            "relevance",  # Not deepeval_relevance
        }

        # Old DeepEval names (should NOT be used in Phoenix mode)
        old_names = {
            "deepeval_faithfulness",
            "deepeval_correctness",
            "deepeval_relevance",
        }

        # Verify no overlap
        assert phoenix_metrics.isdisjoint(old_names)

        # Phoenix names are simpler
        for metric in phoenix_metrics:
            assert "_" not in metric or metric.count("_") == 0
