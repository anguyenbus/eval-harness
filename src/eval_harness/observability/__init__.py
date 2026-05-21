"""
Observability module for RAG pipeline tracing with Phoenix.

This module provides optional Phoenix (Arize AI) integration for distributed
tracing and observability of RAG evaluation runs. When Phoenix is not installed,
the module gracefully degrades and provides warnings for installation instructions.
"""

from __future__ import annotations

import warnings

__all__: list[str] = []

try:
    from eval_harness.observability.phoenix_adapter import PhoenixAdapter

    __all__ = ["PhoenixAdapter"]
except ImportError:
    # Phoenix not installed - provide helpful warning
    warnings.warn(
        "Phoenix not installed. Install with: `uv pip install eval-harness[phoenix]`",
        ImportWarning,
        stacklevel=2,
    )

# Config helper is always available (no Phoenix dependency)
from eval_harness.observability.config import get_phoenix_config as _get_phoenix_config

__all__.append("get_phoenix_config")

# Re-export for backwards compatibility
get_phoenix_config = _get_phoenix_config
