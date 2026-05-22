"""
Evaluation harness for document parsing and RAG systems.

This package provides:
- Dataset loaders for public benchmarks (OmniDocBench, DP-Bench, LegalBench-RAG)
- Deterministic metrics for parsing and retrieval quality
- Adapter pattern for swapping real/stub implementations
- CLI entry points for running evaluations
- CSV and HTML reporting

Typical usage:
    uv run eval-parsing --dataset dp_bench --parser stub
    uv run eval-rag --dataset legalbench_rag --slice mini
"""

import os

# ====================================================================
# SECURITY: DISABLE THIRD-PARTY TELEMETRY ( executed on package import)
# ====================================================================
# DO NOT REMOVE OR MODIFY. Applies to ALL uses of this package.
#
# This disables DeepEval telemetry (analytics, usage stats) globally.
# Setting it here ensures it's applied before any DeepEval code runs.
#
# Why: Privacy, security, compliance, cost. See .env.example for details.
# Reference: https://docs.confident-ai.com/docs/telemetry-opt-out
# ====================================================================
os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "YES"

__version__ = "0.1.0"
