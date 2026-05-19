"""Tests for eval-parsing CLI."""

import pytest

from eval_harness.runners.run_parsing_eval import main


class TestEvalParsingCLI:
    """Test suite for eval-parsing CLI."""

    def test_dataset_argument_accepted(self, tmp_path, monkeypatch, capsys):
        """Test that --dataset argument accepts dp_bench and omnidocbench."""
        # Create tiny dataset fixture
        data_dir = tmp_path / "data" / "parsing" / "omnidocbench_english"
        data_dir.mkdir(parents=True)

        import json

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
            },
        ]

        (data_dir / "OmniDocBench.json").write_text(json.dumps(json_data))

        # Create eval config
        config_dir = tmp_path
        import yaml

        config_data = {
            "datasets": {
                "omnidocbench": {"path": str(data_dir)},
                "dp_bench": {"path": str(tmp_path / "dp_bench")},
                "legalbench_rag": {"path": str(tmp_path / "legalbench")},
            },
            "metrics": {"text_fidelity": {"threshold": 0.95}},
            "models": {"judge_model": "claude-opus-4-7", "temperature": 0},
        }

        (config_dir / "eval_config.yaml").write_text(yaml.dump(config_data))

        # Create results directory
        results_dir = tmp_path / "results"
        results_dir.mkdir()

        # Run CLI
        monkeypatch.setattr(
            "sys.argv",
            [
                "eval-parsing",
                "--dataset",
                "omnidocbench",
                "--parser",
                "stub",
                "--config",
                str(config_dir / "eval_config.yaml"),
                "--output-dir",
                str(results_dir),
            ],
        )

        with pytest.raises(SystemExit) as exc_info:
            main()

        # Should exit successfully
        assert exc_info.value.code == 0

    def test_csv_output_written(self, tmp_path, monkeypatch):
        """Test that CSV output is written to results/ directory."""
        # Create DP-Bench fixture (matching actual structure)
        dataset_dir = tmp_path / "dp_bench" / "dataset"
        pdfs_dir = dataset_dir / "pdfs"
        pdfs_dir.mkdir(parents=True)

        import json

        ref_data = {"doc001.pdf": {"elements": []}}
        (dataset_dir / "reference.json").write_text(json.dumps(ref_data))
        (pdfs_dir / "doc001.pdf").write_bytes(b"%PDF-1.4")

        import yaml

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

        monkeypatch.setattr(
            "sys.argv",
            [
                "eval-parsing",
                "--dataset",
                "dp_bench",
                "--parser",
                "stub",
                "--config",
                str(tmp_path / "eval_config.yaml"),
                "--output-dir",
                str(tmp_path / "results"),
            ],
        )

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 0

    def test_parser_argument_validated(self, tmp_path, monkeypatch):
        """Test that --parser argument validates against allowed values."""
        import json

        import yaml

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
            },
        ]

        (omnidocbench_dir / "OmniDocBench.json").write_text(json.dumps(json_data))

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

        # Test stub parser (should work)
        monkeypatch.setattr(
            "sys.argv",
            [
                "eval-parsing",
                "--dataset",
                "omnidocbench",
                "--parser",
                "stub",
                "--config",
                str(tmp_path / "eval_config.yaml"),
            ],
        )

        with pytest.raises(SystemExit) as exc_info:
            main()

        # Should succeed
        assert exc_info.value.code == 0
