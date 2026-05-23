"""
Generator configuration for synthetic span generation.

This module provides configuration loading for the span generator,
reading from the [generator] section of eval_config.yaml.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from beartype import beartype
from beartype.typing import Optional

from eval_harness.config import load_config

# Constants
DEFAULT_ENDPOINT: Final[str] = "http://localhost:6006"
DEFAULT_PROJECT_NAME: Final[str] = "case-assistant-synthetic"
DEFAULT_LIMIT: Final[int] = 100
DEFAULT_BATCH_EXPORT: Final[bool] = True
DEFAULT_SEED: Final[int] = 42
DEFAULT_STUB_MODEL_ID: Final[str] = "anthropic.claude-3-5-sonnet-20241022-v2:0"
DEFAULT_STUB_EMBEDDING_MODEL: Final[str] = "sentence-transformers/all-MiniLM-L6-v2"
CONFIG_SECTION: Final[str] = "generator"


@beartype
@dataclass(frozen=True)
class GeneratorConfig:
    """
    Configuration for synthetic span generator.

    Attributes:
        phoenix_endpoint: Phoenix server endpoint URL.
        project_name: Phoenix project name for grouping traces.
        default_limit: Default number of questions to process.
        batch_export: Whether to use batch span export.
        seed: Random seed for reproducibility.
        stub_model_id: Model ID for stub LLM generator.
        stub_embedding_model: Model ID for stub embedder.

    """

    phoenix_endpoint: str
    project_name: str
    default_limit: int
    batch_export: bool
    seed: int
    stub_model_id: str
    stub_embedding_model: str


@beartype
def load_generator_config(
    config_path: Path = Path("eval_config.yaml"),
) -> GeneratorConfig:
    """
    Load generator configuration from eval_config.yaml.

    Args:
        config_path: Path to eval_config.yaml file.

    Returns:
        GeneratorConfig instance with values from [generator] section.

    Raises:
        FileNotFoundError: If config file does not exist.
        ValueError: If config is invalid or missing required fields.

    """
    config = load_config(config_path)
    generator_config = config.get(CONFIG_SECTION, {})

    # Resolve phoenix_endpoint with environment variable override
    phoenix_endpoint = _resolve_phoenix_endpoint(
        generator_config.get("phoenix_endpoint")
    )

    return GeneratorConfig(
        phoenix_endpoint=phoenix_endpoint,
        project_name=generator_config.get("project_name", DEFAULT_PROJECT_NAME),
        default_limit=generator_config.get("default_limit", DEFAULT_LIMIT),
        batch_export=generator_config.get("batch_export", DEFAULT_BATCH_EXPORT),
        seed=generator_config.get("seed", DEFAULT_SEED),
        stub_model_id=generator_config.get("stub_model_id", DEFAULT_STUB_MODEL_ID),
        stub_embedding_model=generator_config.get(
            "stub_embedding_model", DEFAULT_STUB_EMBEDDING_MODEL
        ),
    )


def _resolve_phoenix_endpoint(yaml_value: Optional[str]) -> str:
    """
    Resolve Phoenix endpoint from YAML config or environment variable.

    Args:
        yaml_value: Value from YAML config.

    Returns:
        Resolved Phoenix endpoint URL.

    """
    # Environment variable takes precedence
    env_endpoint = os.environ.get("PHOENIX_ENDPOINT")
    if env_endpoint:
        return env_endpoint

    # Use YAML value or default
    return yaml_value if yaml_value else DEFAULT_ENDPOINT
