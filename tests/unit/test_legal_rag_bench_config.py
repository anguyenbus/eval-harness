"""Tests for Legal RAG Bench configuration loading."""

from eval_harness.config import load_config


class TestLegalRagBenchConfig:
    """Test suite for Legal RAG Bench configuration."""

    def test_legal_rag_bench_section_loads(self, tmp_path):
        """Test that legal_rag_bench section loads successfully."""
        config_content = """
        datasets:
          omnidocbench:
            path: /data/omnidocbench
          legal_rag_bench:
            path: ${DATA_DIR:-/data}/rag/legal_rag_bench
            hf_token_path: ${HF_TOKEN_PATH:-~/.huggingface/token}
            k_values: [5, 10, 20]

        metrics:
          text_fidelity:
            threshold: 0.95

        models:
          judge_model: claude-opus-4-7
          temperature: 0
        """
        config_file = tmp_path / "eval_config.yaml"
        config_file.write_text(config_content)

        config = load_config(config_file)

        assert "legal_rag_bench" in config["datasets"]
        assert (
            config["datasets"]["legal_rag_bench"]["path"] == "/data/rag/legal_rag_bench"
        )
        assert (
            config["datasets"]["legal_rag_bench"]["hf_token_path"]
            == "~/.huggingface/token"
        )
        assert config["datasets"]["legal_rag_bench"]["k_values"] == [5, 10, 20]

    def test_environment_variable_expansion_in_paths(self, tmp_path, monkeypatch):
        """Test that environment variables expand in legal_rag_bench paths."""
        monkeypatch.setenv("DATA_DIR", "/custom/data")
        monkeypatch.setenv("HF_TOKEN_PATH", "/custom/hf/token")

        config_content = """
        datasets:
          legal_rag_bench:
            path: ${DATA_DIR}/rag/legal_rag_bench
            hf_token_path: ${HF_TOKEN_PATH}

        metrics:
          text_fidelity:
            threshold: 0.95

        models:
          judge_model: claude-opus-4-7
          temperature: 0
        """
        config_file = tmp_path / "eval_config.yaml"
        config_file.write_text(config_content)

        config = load_config(config_file)

        assert (
            config["datasets"]["legal_rag_bench"]["path"]
            == "/custom/data/rag/legal_rag_bench"
        )
        assert (
            config["datasets"]["legal_rag_bench"]["hf_token_path"] == "/custom/hf/token"
        )

    def test_default_values_in_configuration(self, tmp_path):
        """Test that default values work in legal_rag_bench section."""
        config_content = """
        datasets:
          legal_rag_bench:
            path: ${DATA_DIR:-./data}/rag/legal_rag_bench
            hf_token_path: ${HF_TOKEN_PATH:-~/.huggingface/token}

        metrics:
          text_fidelity:
            threshold: 0.95

        models:
          judge_model: claude-opus-4-7
          temperature: 0
        """
        config_file = tmp_path / "eval_config.yaml"
        config_file.write_text(config_content)

        config = load_config(config_file)

        # Should use defaults when env vars not set
        assert (
            config["datasets"]["legal_rag_bench"]["path"]
            == "./data/rag/legal_rag_bench"
        )
        assert (
            config["datasets"]["legal_rag_bench"]["hf_token_path"]
            == "~/.huggingface/token"
        )

    def test_ragas_judge_model_configuration(self, tmp_path):
        """Test that RAGAS judge model settings can be configured."""
        config_content = """
        datasets:
          legal_rag_bench:
            path: /data/rag/legal_rag_bench
            ragas:
              judge_model: gpt-4o
              judge_model_provider: openai
              temperature: 0

        metrics:
          text_fidelity:
            threshold: 0.95

        models:
          judge_model: claude-opus-4-7
          temperature: 0
        """
        config_file = tmp_path / "eval_config.yaml"
        config_file.write_text(config_content)

        config = load_config(config_file)

        assert "ragas" in config["datasets"]["legal_rag_bench"]
        assert config["datasets"]["legal_rag_bench"]["ragas"]["judge_model"] == "gpt-4o"
        assert (
            config["datasets"]["legal_rag_bench"]["ragas"]["judge_model_provider"]
            == "openai"
        )

    def test_dataset_cache_path_configuration(self, tmp_path):
        """Test that dataset cache path is configurable."""
        config_content = """
        datasets:
          legal_rag_bench:
            cache_path: /custom/cache/legal_rag_bench

        metrics:
          text_fidelity:
            threshold: 0.95

        models:
          judge_model: claude-opus-4-7
          temperature: 0
        """
        config_file = tmp_path / "eval_config.yaml"
        config_file.write_text(config_content)

        config = load_config(config_file)

        assert (
            config["datasets"]["legal_rag_bench"]["cache_path"]
            == "/custom/cache/legal_rag_bench"
        )
