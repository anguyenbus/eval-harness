"""End-to-end integration tests for parsing eval pipeline."""

import json
import time
from pathlib import Path

import pytest
import yaml


class TestParsingPipeline:
    """Integration tests for complete parsing evaluation workflow."""

    def test_full_parsing_eval_with_stub_parser(self, tmp_path):
        """Test complete parsing eval workflow using stub parser."""
        # Create OmniDocBench fixture
        omnidocbench_dir = tmp_path / "omnidocbench"
        omnidocbench_dir.mkdir()

        json_data = [
            {
                "page_info": {
                    "page_no": 1,
                    "height": 792,
                    "width": 612,
                    "image_path": "page1.jpg",
                    "page_attribute": {
                        "language": "english",
                        "data_source": "research_report",
                        "fuzzy_scan": False,
                        "watermark": False,
                        "colorful_backgroud": False,
                        "layout": "single_column",
                    },
                },
                "layout_dets": [],
                "extra": {},
            }
        ]

        (omnidocbench_dir / "OmniDocBench.json").write_text(json.dumps(json_data))

        # Create config
        config_data = {
            "datasets": {
                "omnidocbench": {"path": str(omnidocbench_dir)},
                "dp_bench": {"path": str(tmp_path / "dp_bench")},
                "legalbench_rag": {"path": str(tmp_path / "legalbench")},
            },
            "metrics": {},
            "models": {"judge_model": "claude-opus-4-7", "temperature": 0},
        }

        (tmp_path / "eval_config.yaml").write_text(yaml.dump(config_data))
        (tmp_path / "results").mkdir()

        # Run parsing eval
        from eval_harness.runners.run_parsing_eval import main

        import sys

        original_argv = sys.argv
        sys.argv = [
            "eval-parsing",
            "--dataset",
            "omnidocbench",
            "--parser",
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
        assert elapsed < 30, "Pipeline should complete in under 30 seconds for single page"

    def test_csv_output_generation(self, tmp_path):
        """Test that CSV output is generated correctly."""
        # Create DP-Bench fixture (matching expected structure)
        dp_bench_dataset_dir = tmp_path / "dp_bench" / "dataset"
        dp_bench_pdfs_dir = dp_bench_dataset_dir / "pdfs"
        dp_bench_pdfs_dir.mkdir(parents=True)

        # Create reference.json with one document
        ref_data = {
            "doc001.pdf": {
                "elements": [
                    {
                        "category": "Paragraph",
                        "content": {"text": "Sample text"},
                        "page": 1,
                    }
                ]
            }
        }
        (dp_bench_dataset_dir / "reference.json").write_text(json.dumps(ref_data))
        (dp_bench_pdfs_dir / "doc001.pdf").write_bytes(b"%PDF-1.4")

        # Create config
        config_data = {
            "datasets": {
                "omnidocbench": {"path": str(tmp_path / "omnidocbench")},
                "dp_bench": {"path": str(tmp_path / "dp_bench")},
                "legalbench_rag": {"path": str(tmp_path / "legalbench")},
            },
            "metrics": {},
            "models": {"judge_model": "claude-opus-4-7", "temperature": 0},
        }

        (tmp_path / "eval_config.yaml").write_text(yaml.dump(config_data))
        (tmp_path / "results").mkdir()

        # Run eval
        from eval_harness.runners.run_parsing_eval import main

        import sys

        original_argv = sys.argv
        sys.argv = [
            "eval-parsing",
            "--dataset",
            "dp_bench",
            "--parser",
            "stub",
            "--config",
            str(tmp_path / "eval_config.yaml"),
            "--output-dir",
            str(tmp_path / "results"),
        ]

        with pytest.raises(SystemExit):
            main()

        sys.argv = original_argv

        # Check CSV output (filename includes parser type)
        csv_path = tmp_path / "results" / "dp_bench_stub_results.csv"
        assert csv_path.exists(), "CSV output file should be created"

        import pandas as pd

        df = pd.read_csv(csv_path)
        assert "query_id" in df.columns
        assert "question_id" in df.columns
        assert "score" in df.columns
        assert "label" in df.columns
        assert "error" in df.columns

    def test_html_report_generation(self, tmp_path):
        """Test that HTML report is generated correctly."""
        # Create CSV results
        from eval_harness.reporting.html_summary import generate_summary

        csv_path = tmp_path / "results.csv"
        import pandas as pd

        df = pd.DataFrame({
            "query_id": ["q001", "q002"],
            "question_id": ["text_fidelity", "structure_recall"],
            "score": [0.95, 0.80],
            "label": ["pass", "fail"],
            "error": ["", ""],
        })
        df.to_csv(csv_path, index=False)

        html_path = tmp_path / "summary.html"
        generate_summary(csv_path, html_path)

        assert html_path.exists()

        html = html_path.read_text()
        assert "<html>" in html or "<HTML" in html
        assert "pass" in html
        assert "fail" in html

    def test_regression_check_with_mock_baseline(self, tmp_path):
        """Test regression check with mock baseline."""
        from eval_harness.reporting.regression_check import check_regression

        # Create mock baseline
        baseline_path = tmp_path / "baseline.json"
        baseline_data = {
            "metrics": {
                "text_fidelity": {"score": 0.95, "severity": "major"},
                "structure_recall": {"score": 0.90, "severity": "blocker"},
            }
        }
        baseline_path.write_text(json.dumps(baseline_data))

        # Create current results (no regression)
        current_path = tmp_path / "current.json"
        current_data = {
            "metrics": {
                "text_fidelity": {"score": 0.96, "severity": "major"},  # Improved
                "structure_recall": {"score": 0.90, "severity": "blocker"},  # Same
            }
        }
        current_path.write_text(json.dumps(current_data))

        # Should not raise
        check_regression(current_path, baseline_path)

        # Create current results with regression
        current_data_regression = {
            "metrics": {
                "text_fidelity": {"score": 0.95, "severity": "major"},
                "structure_recall": {"score": 0.80, "severity": "blocker"},  # Regressed!
            }
        }
        current_path.write_text(json.dumps(current_data_regression))

        # Should raise
        with pytest.raises(RuntimeError, match="Regression detected"):
            check_regression(current_path, baseline_path)

    @pytest.mark.slow
    def test_pipeline_runtime_under_2_minutes(self, tmp_path):
        """Verify complete pipeline runs in under 2 minutes."""
        # Create multi-page OmniDocBench fixture
        omnidocbench_dir = tmp_path / "omnidocbench"
        omnidocbench_dir.mkdir()

        json_data = [
            {
                "page_info": {
                    "page_no": i,
                    "height": 792,
                    "width": 612,
                    "image_path": f"page{i}.jpg",
                    "page_attribute": {
                        "language": "english",
                        "data_source": "research_report",
                        "fuzzy_scan": False,
                        "watermark": False,
                        "colorful_backgroud": False,
                        "layout": "single_column",
                    },
                },
                "layout_dets": [],
                "extra": {},
            }
            for i in range(1, 11)  # 10 pages
        ]

        (omnidocbench_dir / "OmniDocBench.json").write_text(json.dumps(json_data))

        # Create config
        config_data = {
            "datasets": {
                "omnidocbench": {"path": str(omnidocbench_dir)},
                "dp_bench": {"path": str(tmp_path / "dp_bench")},
                "legalbench_rag": {"path": str(tmp_path / "legalbench")},
            },
            "metrics": {},
            "models": {"judge_model": "claude-opus-4-7", "temperature": 0},
        }

        (tmp_path / "eval_config.yaml").write_text(yaml.dump(config_data))
        (tmp_path / "results").mkdir()

        # Run parsing eval
        from eval_harness.runners.run_parsing_eval import main

        import sys

        original_argv = sys.argv
        sys.argv = [
            "eval-parsing",
            "--dataset",
            "omnidocbench",
            "--parser",
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

        assert exc_info.value.code == 0
        assert elapsed < 120, f"Pipeline took {elapsed:.1f}s, should be under 2 minutes"
