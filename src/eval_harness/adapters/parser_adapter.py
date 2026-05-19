"""
Parser adapter with schema validation.

The adapter pattern allows swapping between stub and real parsers while
ensuring all output conforms to the parser_output schema.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

from eval_harness.adapters.schema_validator import (
    validate as schema_validate,
)

# Type alias for parse callable
ParseCallable = Callable[[Path], dict[str, Any]]


class ParserAdapter:
    """
    Adapter for document parsers with schema validation.

    The adapter wraps a parse function and validates its output against
    the parser_output schema before returning. This ensures that any parser
    (stub or real) produces conformant output.

    Attributes:
        parse_callable: Function with signature (pdf_path: Path) -> dict.

    Example:
        >>> adapter = ParserAdapter()
        >>> output = adapter.parse(Path("document.pdf"))
        >>> # output is guaranteed to validate against parser_output schema

    """

    def __init__(self, parse_callable: ParseCallable | None = None) -> None:
        """
        Initialize parser adapter.

        Args:
            parse_callable: Optional parse function. If None, uses stub parser.
                Must have signature: (pdf_path: Path) -> dict.

        """
        if parse_callable is None:
            # Import stub as default
            from eval_harness.stubs.stub_parser import parse as stub_parse

            self._parse = stub_parse
        else:
            self._parse = parse_callable

    def parse(self, pdf_path: Path) -> dict[str, Any]:
        """
        Parse a PDF file and return validated output.

        Invokes the underlying parse callable, validates the output against
        parser_output.schema.json, and returns the validated dict.

        Args:
            pdf_path: Path to PDF file to parse.

        Returns:
            Validated parser output dictionary conforming to schema.

        Raises:
            SchemaValidationError: If parser output fails schema validation.
            FileNotFoundError: If PDF file doesn't exist (from parser).

        """
        # Invoke parser
        output = self._parse(pdf_path)

        # Validate against schema
        schema_path = Path("contracts/parser_output.schema.json")
        schema_validate(output, schema_path)

        return output
