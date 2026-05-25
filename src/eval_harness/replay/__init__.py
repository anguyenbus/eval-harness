"""
Phoenix replay evaluation module.

This package provides tools for replay evaluation against generated synthetic
spans, enabling A/B comparison of candidate versus baseline adapters.
"""

from eval_harness.replay.comparison import (
    ComparisonResult,
    paired_comparison,
)
from eval_harness.replay.phoenix_client import PhoenixClient
from eval_harness.replay.tasks import BaselineTask, CandidateTask

__all__ = [
    "PhoenixClient",
    "CandidateTask",
    "BaselineTask",
    "ComparisonResult",
    "paired_comparison",
]
