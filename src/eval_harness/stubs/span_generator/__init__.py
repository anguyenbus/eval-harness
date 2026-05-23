"""
Synthetic span generator for replay evaluation.

This package provides tools to generate synthetic OpenInference-compliant
spans from a stub RAG pipeline for testing evaluation harnesses before
production traffic exists.
"""

from eval_harness.stubs.span_generator.config import (
    GeneratorConfig,
    load_generator_config,
)

__all__ = [
    "GeneratorConfig",
    "load_generator_config",
]
