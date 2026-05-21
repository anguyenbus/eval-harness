"""Tests for DeepEval configuration module."""

from unittest.mock import patch

import pytest

from eval_harness.metrics.deepeval_config import (
    create_deepeval_metrics,
    get_deepeval_config,
    get_deepeval_llm,
)


class TestDeepEvalConfig:
    """Test suite for DeepEval configuration."""

    def test_deepeval_imports_available(self):
        """Test that DeepEval imports are available."""
        from deepeval.metrics import (
            AnswerRelevancyMetric,
            ContextualPrecisionMetric,
            ContextualRecallMetric,
            FaithfulnessMetric,
        )
        from deepeval.test_case import LLMTestCase

        assert LLMTestCase is not None
        assert FaithfulnessMetric is not None
        assert ContextualPrecisionMetric is not None
        assert ContextualRecallMetric is not None
        assert AnswerRelevancyMetric is not None

    def test_get_deepeval_llm_openai(self, monkeypatch):
        """Test OpenAI provider initialization with API key."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        llm = get_deepeval_llm(provider="openai", model="gpt-4o")
        assert llm is not None
        assert llm.get_model_name() == "gpt-4o"

    def test_get_deepeval_llm_bedrock(self, monkeypatch):
        """Test Bedrock provider initialization."""
        # NOTE: Bedrock tests require aiobotocore dependency
        # This test is skipped unless the dependency is installed
        try:
            import aiobotocore  # noqa: F401
        except ImportError:
            pytest.skip("aiobotocore not installed")

        # Mock AWS credentials
        with patch.dict(
            "os.environ",
            {
                "AWS_ACCESS_KEY_ID": "test-key",
                "AWS_SECRET_ACCESS_KEY": "test-secret",
                "AWS_REGION": "us-east-1",
            },
        ):
            llm = get_deepeval_llm(
                provider="bedrock", model="anthropic.claude-3-5-sonnet-20241022-v2:0"
            )
            assert llm is not None

    def test_get_deepeval_llm_missing_api_key(self, monkeypatch):
        """Test that missing API key raises ValueError."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        with pytest.raises(ValueError, match="OPENAI_API_KEY environment variable"):
            get_deepeval_llm(provider="openai", model="gpt-4o")

    def test_get_deepeval_llm_invalid_provider(self, monkeypatch):
        """Test that invalid provider raises ValueError."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        with pytest.raises(ValueError, match="Unsupported provider"):
            get_deepeval_llm(provider="invalid", model="gpt-4o")

    def test_create_deepeval_metrics(self, monkeypatch):
        """Test that create_deepeval_metrics creates all metrics."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        metrics = create_deepeval_metrics()

        assert "faithfulness" in metrics
        assert "context_precision" in metrics
        assert "context_recall" in metrics
        assert "answer_relevancy" in metrics

    def test_get_deepeval_config_defaults(self):
        """Test default values from get_deepeval_config."""
        config = {
            "datasets": {
                "legal_rag_bench": {
                    "deepeval": {
                        "enabled": True,
                        "judge_model": "gpt-4o",
                        "judge_model_provider": "openai",
                        "temperature": 0.0,
                        "max_concurrent": 10,
                    }
                }
            }
        }

        deepeval_config = get_deepeval_config(config)

        assert deepeval_config["enabled"] is True
        assert deepeval_config["judge_model"] == "gpt-4o"
        assert deepeval_config["judge_model_provider"] == "openai"
        assert deepeval_config["temperature"] == 0.0
        assert deepeval_config["max_concurrent"] == 10

    def test_get_deepeval_config_yaml_fallback(self):
        """Test that YAML config is used when CLI args not provided."""
        config = {
            "datasets": {
                "legal_rag_bench": {
                    "deepeval": {
                        "enabled": True,
                        "judge_model": "gpt-4o-mini",
                        "judge_model_provider": "bedrock",
                        "temperature": 0.3,
                        "max_concurrent": 20,
                    }
                }
            }
        }

        deepeval_config = get_deepeval_config(config)

        assert deepeval_config["judge_model"] == "gpt-4o-mini"
        assert deepeval_config["judge_model_provider"] == "bedrock"
        assert deepeval_config["temperature"] == 0.3
        assert deepeval_config["max_concurrent"] == 20

    def test_get_deepeval_config_cli_precedence(self):
        """Test that CLI args override YAML config."""
        config = {
            "datasets": {
                "legal_rag_bench": {
                    "deepeval": {
                        "enabled": True,
                        "judge_model": "gpt-4o",
                        "judge_model_provider": "openai",
                        "temperature": 0.0,
                        "max_concurrent": 10,
                    }
                }
            }
        }

        deepeval_config = get_deepeval_config(
            config,
            cli_judge_model="gpt-4o-mini",
            cli_provider="bedrock",
            cli_temperature=0.5,
            cli_max_concurrent=25,
        )

        assert deepeval_config["judge_model"] == "gpt-4o-mini"
        assert deepeval_config["judge_model_provider"] == "bedrock"
        assert deepeval_config["temperature"] == 0.5
        assert deepeval_config["max_concurrent"] == 25

    def test_get_deepeval_config_env_var_override(self, monkeypatch):
        """Test that DEEPEVAL_MAX_CONCURRENT env var overrides YAML."""
        monkeypatch.setenv("DEEPEVAL_MAX_CONCURRENT", "15")

        config = {
            "datasets": {
                "legal_rag_bench": {
                    "deepeval": {
                        "enabled": True,
                        "judge_model": "gpt-4o",
                        "judge_model_provider": "openai",
                        "temperature": 0.0,
                        # Note: max_concurrent in YAML is lower than env var
                        "max_concurrent": 10,
                    }
                }
            }
        }

        deepeval_config = get_deepeval_config(config)

        # The env var should override the YAML value when no CLI value is provided
        assert deepeval_config["max_concurrent"] == 15
