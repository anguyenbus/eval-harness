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

__version__ = "0.1.0"
