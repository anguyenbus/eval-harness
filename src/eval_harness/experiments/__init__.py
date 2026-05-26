"""
Phoenix experiment framework integration.

This module provides integration with Phoenix's native experiment API,
allowing DeepEval metrics to be wrapped as Phoenix evaluators and run
through Phoenix's experiment framework for better UI and comparison.

Feature flag: phoenix_native_enabled (default: False)
"""

from __future__ import annotations

from eval_harness.experiments.config import get_phoenix_native_config

__all__ = ["get_phoenix_native_config"]
