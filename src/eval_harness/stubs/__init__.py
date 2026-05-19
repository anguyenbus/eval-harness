"""
Stub implementations for testing.

Stubs provide minimal, schema-conformant implementations of parsers and RAG systems
for testing and development purposes.
"""

from eval_harness.stubs.stub_ingestion import query as query_stub
from eval_harness.stubs.stub_parser import parse as parse_stub

try:
    from eval_harness.stubs.docling_parser import parse as parse_docling
    __all__ = ["parse_stub", "query_stub", "parse_docling"]
except ImportError:
    __all__ = ["parse_stub", "query_stub"]
