"""
Dataset loaders for public benchmarks.

This module provides loaders for:
- OmniDocBench: Document layout parsing benchmark
- DP-Bench: Document parsing benchmark
- LegalBench-RAG: Legal domain RAG benchmark
- Legal RAG Bench: Legal RAG benchmark with LLM-judge metrics

All loaders use iterator pattern for memory efficiency.
"""

from eval_harness.datasets.dp_bench import load_dp_bench
from eval_harness.datasets.legal_rag_bench import load_legal_rag_bench
from eval_harness.datasets.legalbench_rag import load_legalbench_rag
from eval_harness.datasets.omnidocbench import load_omnidocbench

__all__ = [
    "load_omnidocbench",
    "load_dp_bench",
    "load_legalbench_rag",
    "load_legal_rag_bench",
]
