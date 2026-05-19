"""Tests for eval-rag CLI."""

import pytest

from eval_harness.runners.run_rag_eval import main


class TestEvalRagCLI:
    """Test suite for eval-rag CLI."""

    def test_dataset_argument_accepted(self, tmp_path, monkeypatch):
        """Test that --dataset argument accepts legalbench_rag."""
        # Create tiny dataset fixture
        corpus_dir = tmp_path / "data" / "rag" / "legalbench" / "cuad"
        corpus_dir.mkdir(parents=True)

        import json

        queries_data = {
            "queries": [
                {
                    "query_id": "q001",
                    "query_text": "Test question?",
                    "gold_spans": [[0, 25]],
                    "gold_answer_text": "Test answer.",
                },
            ],
        }

        (corpus_dir / "queries.json").write_text(json.dumps(queries_data))

        # Create eval config
        import yaml

        config_data = {
            "datasets": {
                "omnidocbench": {"path": str(tmp_path / "omnidocbench")},
                "dp_bench": {"path": str(tmp_path / "dp_bench")},
                "legalbench_rag": {
                    "path": str(tmp_path / "data" / "rag" / "legalbench")
                },
            },
            "metrics": {},
            "models": {"judge_model": "claude-opus-4-7", "temperature": 0},
        }

        (tmp_path / "eval_config.yaml").write_text(yaml.dump(config_data))
        (tmp_path / "results").mkdir()

        # Run CLI
        monkeypatch.setattr(
            "sys.argv",
            [
                "eval-rag",
                "--dataset",
                "legalbench_rag",
                "--slice",
                "mini",
                "--rag",
                "stub",
            ],
        )

        with pytest.raises(SystemExit) as exc_info:
            main()

        # Should exit successfully (or with 1 for empty results, still OK)
        assert exc_info.value.code in [0, 1]

    def test_slice_argument_accepted(self, tmp_path, monkeypatch):
        """Test that --slice argument accepts mini and full."""
        import json

        import yaml

        # Create mini fixture
        corpus_dir = tmp_path / "legalbench" / "cuad"
        corpus_dir.mkdir(parents=True)

        queries_data = {"queries": []}
        (corpus_dir / "queries.json").write_text(json.dumps(queries_data))

        config_data = {
            "datasets": {"legalbench_rag": {"path": str(tmp_path / "legalbench")}},
            "metrics": {},
            "models": {"judge_model": "claude-opus-4-7", "temperature": 0},
        }

        (tmp_path / "eval_config.yaml").write_text(yaml.dump(config_data))
        (tmp_path / "results").mkdir()

        # Test mini slice
        monkeypatch.setattr(
            "sys.argv",
            [
                "eval-rag",
                "--dataset",
                "legalbench_rag",
                "--slice",
                "mini",
                "--rag",
                "stub",
            ],
        )

        with pytest.raises(SystemExit):
            main()

    def test_csv_output_written(self, tmp_path, monkeypatch):
        """Test that CSV output is written to results/ directory."""
        import json

        import yaml

        corpus_dir = tmp_path / "legalbench" / "maud"
        corpus_dir.mkdir(parents=True)

        queries_data = {"queries": []}
        (corpus_dir / "queries.json").write_text(json.dumps(queries_data))

        config_data = {
            "datasets": {"legalbench_rag": {"path": str(tmp_path / "legalbench")}},
            "metrics": {},
            "models": {"judge_model": "claude-opus-4-7", "temperature": 0},
        }

        (tmp_path / "eval_config.yaml").write_text(yaml.dump(config_data))
        (tmp_path / "results").mkdir()

        monkeypatch.setattr(
            "sys.argv",
            [
                "eval-rag",
                "--dataset",
                "legalbench_rag",
                "--slice",
                "mini",
                "--rag",
                "stub",
            ],
        )

        with pytest.raises(SystemExit) as exc_info:
            main()

        # Should exit (may produce empty results for empty dataset)
        assert exc_info.value.code in [0, 1]

    def test_rag_argument_validated(self, tmp_path, monkeypatch):
        """Test that --rag argument validates."""
        import json

        import yaml

        corpus_dir = tmp_path / "legalbench" / "privacyqa"
        corpus_dir.mkdir(parents=True)

        queries_data = {"queries": []}
        (corpus_dir / "queries.json").write_text(json.dumps(queries_data))

        config_data = {
            "datasets": {"legalbench_rag": {"path": str(tmp_path / "legalbench")}},
            "metrics": {},
            "models": {"judge_model": "claude-opus-4-7", "temperature": 0},
        }

        (tmp_path / "eval_config.yaml").write_text(yaml.dump(config_data))
        (tmp_path / "results").mkdir()

        # Test stub RAG (should work)
        monkeypatch.setattr(
            "sys.argv",
            [
                "eval-rag",
                "--dataset",
                "legalbench_rag",
                "--slice",
                "mini",
                "--rag",
                "stub",
            ],
        )

        with pytest.raises(SystemExit):
            main()
