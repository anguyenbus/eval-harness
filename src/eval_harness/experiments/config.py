"""
Phoenix Native feature flag configuration.

Controls whether to use Phoenix's native experiment API vs manual span creation.
"""

from __future__ import annotations

from typing import Any, Final

from beartype import beartype
from beartype.typing import Dict

# Constants
DEFAULT_PHOENIX_NATIVE_ENABLED: Final[bool] = False


@beartype
def get_phoenix_native_config(
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Get Phoenix Native configuration with safe defaults.

    Reads from config.yaml:
        phoenix_native:
            use_phoenix_native: true/false

    Also respects environment variable PHOENIX_NATIVE_ENABLED for CI/CD.

    Args:
        config: Full configuration dictionary from load_config().

    Returns:
        Dictionary with:
            - use_phoenix_native: bool (whether to use Phoenix-native experiments)

    """
    import os

    # Environment variable has highest precedence (for CI/CD)
    env_value = os.environ.get("PHOENIX_NATIVE_ENABLED", "").lower()
    if env_value in ("1", "true", "yes", "on"):
        return {"use_phoenix_native": True}
    if env_value in ("0", "false", "no", "off"):
        return {"use_phoenix_native": False}

    # YAML config
    phoenix_native_yaml = config.get("phoenix_native", {})
    use_phoenix_native = phoenix_native_yaml.get(
        "use_phoenix_native", DEFAULT_PHOENIX_NATIVE_ENABLED
    )

    return {
        "use_phoenix_native": use_phoenix_native,
    }
