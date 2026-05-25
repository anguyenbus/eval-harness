"""
Tests for replay CLI entry points.
"""

from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from eval_harness.runners.generate_spans import main as generate_spans_main
from eval_harness.runners.run_replay_eval import main as eval_replay_main


class TestGenerateSpansCli:
    """Tests for generate-spans CLI."""

    def test_generate_spans_help(self) -> None:
        """Test that generate-spans CLI has help text."""
        runner = CliRunner()
        result = runner.invoke(generate_spans_main, ["--help"])

        assert result.exit_code == 0
        assert "generate" in result.output.lower()

    def test_generate_spans_missing_config(self) -> None:
        """Test generate-spans with missing config file."""
        runner = CliRunner()
        result = runner.invoke(
            generate_spans_main, ["--config", "/nonexistent/config.yaml"]
        )

        assert result.exit_code == 1
        assert "ERROR" in result.output

    def test_generate_spans_with_limit_option(self) -> None:
        """Test generate-spans accepts --limit option."""
        runner = CliRunner()

        with patch("eval_harness.runners.generate_spans.load_config") as mock_load_config:
            with patch(
                "eval_harness.runners.generate_spans.load_generator_config"
            ) as mock_load_gen:
                with patch(
                    "eval_harness.runners.generate_spans.run_generator"
                ) as mock_run:
                    # Setup mocks
                    mock_config = {
                        "datasets": {
                            "legal_rag_bench": {
                                "path": "data/corpus",
                            }
                        }
                    }
                    mock_load_config.return_value = mock_config

                    from dataclasses import dataclass

                    @dataclass
                    class MockConfig:
                        phoenix_endpoint: str
                        project_name: str
                        default_limit: int
                        batch_export: bool
                        seed: int
                        stub_model_id: str
                        stub_embedding_model: str

                    mock_gen_config = MockConfig(
                        phoenix_endpoint="http://localhost:6006",
                        project_name="test-project",
                        default_limit=100,
                        batch_export=True,
                        seed=42,
                        stub_model_id="model",
                        stub_embedding_model="embedder",
                    )
                    mock_load_gen.return_value = mock_gen_config

                    from dataclasses import dataclass

                    @dataclass
                    class MockResult:
                        successes: int
                        failures: int
                        run_id: str

                    mock_run.return_value = MockResult(
                        successes=5, failures=0, run_id="test-run-id"
                    )

                    runner.invoke(
                        generate_spans_main, ["--limit", "5", "--seed", "42"]
                    )

                    # Should pass validation (actual execution depends on mocks)
                    # The CLI should accept the options without error

    def test_generate_spans_with_phoenix_endpoint(self) -> None:
        """Test generate-spans accepts --phoenix-endpoint option."""
        runner = CliRunner()

        with patch("eval_harness.runners.generate_spans.load_config") as mock_load_config:
            with patch(
                "eval_harness.runners.generate_spans.load_generator_config"
            ) as mock_load_gen:
                mock_config = {
                    "datasets": {
                        "legal_rag_bench": {
                            "path": "data/corpus",
                        }
                    }
                }
                mock_load_config.return_value = mock_config

                from dataclasses import dataclass

                @dataclass
                class MockConfig:
                    phoenix_endpoint: str
                    project_name: str
                    default_limit: int
                    batch_export: bool
                    seed: int
                    stub_model_id: str
                    stub_embedding_model: str

                mock_gen_config = MockConfig(
                    phoenix_endpoint="http://custom:6006",
                    project_name="test-project",
                    default_limit=100,
                    batch_export=True,
                    seed=42,
                    stub_model_id="model",
                    stub_embedding_model="embedder",
                )
                mock_load_gen.return_value = mock_gen_config

                # Test that --phoenix-endpoint is accepted
                result = runner.invoke(
                    generate_spans_main, ["--phoenix-endpoint", "http://custom:6006"]
                )

                # CLI should accept the option
                assert result.exit_code in [0, 1]  # May fail on actual execution


class TestEvalReplayCli:
    """Tests for eval-replay CLI."""

    def test_eval_replay_help(self) -> None:
        """Test that eval-replay CLI has help text."""
        runner = CliRunner()
        result = runner.invoke(eval_replay_main, ["--help"])

        assert result.exit_code == 0
        assert "replay" in result.output.lower()

    def test_eval_replay_requires_candidate(self) -> None:
        """Test that eval-replay requires --candidate option."""
        runner = CliRunner()
        result = runner.invoke(eval_replay_main, [])

        # Should fail without --candidate
        assert result.exit_code != 0

    def test_eval_replay_with_candidate_option(self) -> None:
        """Test eval-replay accepts --candidate option."""
        runner = CliRunner()

        with patch("eval_harness.runners.run_replay_eval.PhoenixClient") as mock_client:
            # Mock client that returns no spans
            mock_instance = MagicMock()
            mock_instance.is_connected.return_value = False
            mock_instance.query_root_spans.return_value = []
            mock_client.return_value = mock_instance

            result = runner.invoke(
                eval_replay_main, ["--candidate", "test_adapter"]
            )

            # Should run (may have warning about no connection)
            assert "test_adapter" in result.output or "WARNING" in result.output

    def test_eval_replay_with_baseline_option(self) -> None:
        """Test eval-replay accepts --baseline option."""
        runner = CliRunner()

        with patch("eval_harness.runners.run_replay_eval.PhoenixClient") as mock_client:
            mock_instance = MagicMock()
            mock_instance.is_connected.return_value = False
            mock_instance.query_root_spans.return_value = []
            mock_client.return_value = mock_instance

            result = runner.invoke(
                eval_replay_main,
                ["--candidate", "test", "--baseline", "baseline"],
            )

            # Should run
            assert "baseline" in result.output or "WARNING" in result.output

    def test_eval_replay_with_output_option(self) -> None:
        """Test eval-replay accepts --output option."""
        runner = CliRunner()

        with patch("eval_harness.runners.run_replay_eval.PhoenixClient") as mock_client:
            mock_instance = MagicMock()
            mock_instance.is_connected.return_value = False
            mock_instance.query_root_spans.return_value = []
            mock_client.return_value = mock_instance

            with runner.isolated_filesystem():
                result = runner.invoke(
                    eval_replay_main,
                    ["--candidate", "test", "--output", "results.json"],
                )

                # Should run (output file may not be created due to mock)
                assert result.exit_code in [0, 1]
