"""End-to-end integration tests for RAG eval pipeline."""

import json
import time

import pytest
import yaml


class TestRAGPipeline:
    """Integration tests for complete RAG evaluation workflow."""

    def test_full_rag_eval_with_stub_rag(self, tmp_path):
        """Test complete RAG eval workflow using stub RAG."""
        # Create LegalBench-RAG fixture
        corpus_dir = tmp_path / "legalbench" / "cuad"
        corpus_dir.mkdir(parents=True)

        queries_data = {
            "queries": [
                {
                    "query_id": "q001",
                    "query_text": "What is the termination clause?",
                    "gold_spans": [[0, 50]],
                    "gold_answer_text": "The contract can be terminated with 30 days notice.",
                },
                {
                    "query_id": "q002",
                    "query_text": "What is the payment schedule?",
                    "gold_spans": [[100, 150]],
                    "gold_answer_text": "Payments are due monthly.",
                },
            ]
        }

        (corpus_dir / "queries.json").write_text(json.dumps(queries_data))

        # Create config
        config_data = {
            "datasets": {
                "legalbench_rag": {"path": str(tmp_path / "legalbench")},
            },
            "metrics": {},
            "models": {"judge_model": "claude-opu-4-7", "temperature": 0},
        }

        (tmp_path / "eval_config.yaml").write_text(yaml.dump(config_data))
        (tmp_path / "results").mkdir()

        # Run RAG eval
        import sys

        from eval_harness.runners.run_rag_eval import main

        original_argv = sys.argv
        sys.argv = [
            "eval-rag",
            "--dataset",
            "legalbench_rag",
            "--slice",
            "mini",
            "--rag",
            "stub",
            "--config",
            str(tmp_path / "eval_config.yaml"),
            "--output-dir",
            str(tmp_path / "results"),
        ]

        start_time = time.time()
        with pytest.raises(SystemExit) as exc_info:
            main()
        elapsed = time.time() - start_time

        sys.argv = original_argv

        # Should complete successfully
        assert exc_info.value.code == 0
        assert elapsed < 10, "Pipeline should complete quickly for stub"

    def test_csv_and_html_output(self, tmp_path):
        """Test that RAG eval produces CSV and HTML output."""
        # Create LegalBench-RAG fixture
        corpus_dir = tmp_path / "legalbench" / "maud"
        corpus_dir.mkdir(parents=True)

        queries_data = {
            "queries": [
                {
                    "query_id": "q001",
                    "query_text": "Test question?",
                    "gold_spans": [[0, 25]],
                    "gold_answer_text": "Test answer.",
                }
            ]
        }

        (corpus_dir / "queries.json").write_text(json.dumps(queries_data))

        # Create config
        config_data = {
            "datasets": {
                "legalbench_rag": {"path": str(tmp_path / "legalbench")},
            },
            "metrics": {},
            "models": {"judge_model": "claude-opus-4-7", "temperature": 0},
        }

        (tmp_path / "eval_config.yaml").write_text(yaml.dump(config_data))
        (tmp_path / "results").mkdir()

        # Run RAG eval
        import sys

        from eval_harness.runners.run_rag_eval import main

        original_argv = sys.argv
        sys.argv = [
            "eval-rag",
            "--dataset",
            "legalbench_rag",
            "--slice",
            "mini",
            "--rag",
            "stub",
            "--config",
            str(tmp_path / "eval_config.yaml"),
            "--output-dir",
            str(tmp_path / "results"),
        ]

        with pytest.raises(SystemExit):
            main()

        sys.argv = original_argv

        # Check CSV output
        csv_path = tmp_path / "results" / "legalbench_rag_mini_results.csv"
        assert csv_path.exists(), "CSV output file should be created"

        import pandas as pd

        df = pd.read_csv(csv_path)
        assert len(df) > 0

        # Generate HTML summary
        from eval_harness.reporting.html_summary import generate_summary

        html_path = tmp_path / "rag_summary.html"
        generate_summary(csv_path, html_path)

        assert html_path.exists()
