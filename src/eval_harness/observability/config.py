"""
Phoenix configuration loading helpers.

This module provides functions for loading Phoenix-specific configuration
from the main eval_config.yaml file, with support for environment variable
expansion and CLI flag overrides.
"""

from __future__ import annotations

import os
from typing import Any, Final

from beartype import beartype
from beartype.typing import Dict

# Constants
DEFAULT_ENDPOINT: Final[str] = "http://localhost:6006"
DEFAULT_ENABLED: Final[bool] = False
DEFAULT_EXPORT_PATH: Final[str] = "/tmp/phoenix_traces"


@beartype
def get_phoenix_config(
    config: Dict[str, Any],
    cli_enabled: bool | None = None,
    cli_endpoint: str | None = None,
) -> Dict[str, Any]:
    """
    Get Phoenix configuration with proper precedence.

    Precedence order: CLI flags > YAML config > environment variables > defaults

    Args:
        config: Full configuration dictionary from load_config().
        cli_enabled: Override for enabled flag from CLI.
        cli_endpoint: Override for endpoint from CLI.

    Returns:
        Dictionary with Phoenix configuration:
            - enabled: bool (whether Phoenix tracing is enabled)
            - endpoint: str (Phoenix server URL)
            - export_path: str (path for buffering traces)
            - mode: str ("spans" or "native", defaults to "spans")

    Example:
        >>> config = load_config(Path("eval_config.yaml"))
        >>> phoenix_config = get_phoenix_config(
        ...     config,
        ...     cli_enabled=True,
        ...     cli_endpoint="http://localhost:6006"
        ... )
        >>> print(phoenix_config["endpoint"])

    """
    # Get phoenix section from config (may not exist)
    phoenix_yaml = config.get("phoenix", {})

    # Apply CLI overrides with highest precedence
    enabled = _resolve_enabled(phoenix_yaml, cli_enabled)
    endpoint = _resolve_endpoint(phoenix_yaml, cli_endpoint)
    export_path = _resolve_export_path(phoenix_yaml)
    mode = _resolve_mode(phoenix_yaml)

    return {
        "enabled": enabled,
        "endpoint": endpoint,
        "export_path": export_path,
        "mode": mode,
    }


def _resolve_enabled(
    phoenix_yaml: Dict[str, Any],
    cli_enabled: bool | None,
) -> bool:
    """
    Resolve enabled flag with precedence.

    Args:
        phoenix_yaml: Phoenix section from YAML config.
        cli_enabled: CLI override for enabled flag.

    Returns:
        Boolean indicating whether Phoenix is enabled.

    """
    if cli_enabled is not None:
        return cli_enabled
    return phoenix_yaml.get("enabled", DEFAULT_ENABLED)


def _resolve_endpoint(
    phoenix_yaml: Dict[str, Any],
    cli_endpoint: str | None,
) -> str:
    """
    Resolve endpoint with precedence.

    Args:
        phoenix_yaml: Phoenix section from YAML config.
        cli_endpoint: CLI override for endpoint.

    Returns:
        Phoenix server endpoint URL.

    """
    # CLI flag has highest precedence
    if cli_endpoint is not None:
        return cli_endpoint

    # YAML config may contain environment variable reference
    yaml_endpoint = phoenix_yaml.get("endpoint")

    if yaml_endpoint:
        # Expand environment variables if present
        if isinstance(yaml_endpoint, str) and "${" in yaml_endpoint:
            return _expand_env_var(yaml_endpoint, DEFAULT_ENDPOINT)
        return yaml_endpoint

    # Environment variable
    env_endpoint = os.environ.get("PHOENIX_ENDPOINT")
    if env_endpoint:
        return env_endpoint

    # Default
    return DEFAULT_ENDPOINT


@beartype
def _resolve_mode(phoenix_yaml: Dict[str, Any]) -> str:
    """
    Resolve Phoenix mode with precedence.

    Args:
        phoenix_yaml: Phoenix section from YAML config.

    Returns:
        Phoenix mode: "spans" (manual spans) or "native" (experiment API).

    """
    # YAML config
    yaml_mode = phoenix_yaml.get("mode", "spans")

    # Validate
    if yaml_mode not in ("spans", "native"):
        print(f"[WARN] Invalid phoenix mode '{yaml_mode}', defaulting to 'spans'")
        return "spans"

    return yaml_mode


@beartype
def _resolve_export_path(phoenix_yaml: Dict[str, Any]) -> str:
    """
    Resolve export path with precedence.

    Args:
        phoenix_yaml: Phoenix section from YAML config.

    Returns:
        Path for buffering Parquet traces.

    """
    yaml_path = phoenix_yaml.get("export_path")

    if yaml_path:
        # Expand environment variables if present
        if isinstance(yaml_path, str) and "${" in yaml_path:
            return _expand_env_var(yaml_path, DEFAULT_EXPORT_PATH)
        return yaml_path

    # Environment variable
    env_path = os.environ.get("PHOENIX_EXPORT_PATH")
    if env_path:
        return env_path

    # Default
    return DEFAULT_EXPORT_PATH


def _expand_env_var(value: str, default: str) -> str:
    """
    Expand environment variable in a string value.

    Supports ${VAR_NAME} and ${VAR_NAME:-default} syntax.

    Args:
        value: String potentially containing ${VAR_NAME} or ${VAR_NAME:-default}.
        default: Default value if env var is not set.

    Returns:
        String with environment variable expanded.

    """
    import re

    pattern = re.compile(r"\$\{([^}:]+)(?::-([^}]*))?\}")

    def replace_var(match: re.Match) -> str:
        var_name = match.group(1)
        yaml_default = match.group(2)

        if var_name in os.environ:
            return os.environ[var_name]
        elif yaml_default is not None:
            return yaml_default
        else:
            return default

    return pattern.sub(replace_var, value)
