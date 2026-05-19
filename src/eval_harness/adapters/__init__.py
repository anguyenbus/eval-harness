"""
Adapters for parsers and RAG systems.

The adapter pattern allows swapping between stub implementations and real systems
while maintaining schema validation.
"""

from eval_harness.adapters.schema_validator import (
    SchemaValidationError,
    validate,
)

__all__ = ["SchemaValidationError", "validate"]
