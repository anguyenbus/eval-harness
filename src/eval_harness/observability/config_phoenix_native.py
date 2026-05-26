"""
Phoenix native configuration loading helpers.

This module provides functions for loading Phoenix-native feature flags
from the main eval_config.yaml file, with support for gradual migration.
"""

from __future__ import annotations

from typing import Any, Final

from beartype import beartype
from beartype.typing import Dict

# Constants
DEFAULT_USE_PHOENIX_NATIVE: Final[bool] = False


@beartype
def get_phoenix_native_config(
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Get Phoenix native configuration with safe defaults.

    Args:
        config: Full configuration dictionary from load_config().

    Returns:
        Dictionary with Phoenix native configuration:
            - use_phoenix_native: bool (whether to use Phoenix-native patterns)

    Example:
        >>> config = load_config(Path("eval_config.yaml"))
        >>> phoenix_config = get_phoenix_native_config(config)
        >>> print(phoenix_config["use_phoenix_native"])

    """
    # Get phoenix_native section from config (may not exist)
    phoenix_native_yaml = config.get("phoenix_native", {})

    # Use default if not specified (safe rollback)
    use_phoenix_native = phoenix_native_yaml.get(
        "use_phoenix_native", DEFAULT_USE_PHOENIX_NATIVE
    )

    return {
        "use_phoenix_native": use_phoenix_native,
    }
