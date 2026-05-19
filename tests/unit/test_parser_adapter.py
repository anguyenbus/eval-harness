"""Tests for ParserAdapter."""

from pathlib import Path

import pytest

from eval_harness.adapters.parser_adapter import ParserAdapter
from eval_harness.adapters.schema_validator import SchemaValidationError
from eval_harness.stubs.stub_parser import parse as stub_parse


class TestParserAdapter:
    """Test suite for ParserAdapter."""

    def test_adapter_invokes_parse_callable(self, tmp_path):
        """Test that adapter invokes the provided parse callable."""
        dummy_pdf = tmp_path / "test.pdf"
        dummy_pdf.write_bytes(b"%PDF-1.4 dummy")

        adapter = ParserAdapter(parse_callable=stub_parse)
        output = adapter.parse(dummy_pdf)

        assert output is not None
        assert "schema_version" in output

    def test_adapter_validates_output(self, tmp_path):
        """Test that adapter validates output against schema."""
        dummy_pdf = tmp_path / "test.pdf"
        dummy_pdf.write_bytes(b"%PDF-1.4 dummy")

        adapter = ParserAdapter(parse_callable=stub_parse)
        output = adapter.parse(dummy_pdf)

        # Should have validated and returned valid output
        assert output["schema_version"] == "1.0.0"
        assert "elements" in output

    def test_adapter_raises_on_invalid_output(self, tmp_path):
        """Test that adapter raises SchemaValidationError for invalid output."""

        def invalid_parse(pdf_path: Path) -> dict:
            # Returns invalid output missing required fields
            return {"invalid": "output"}

        dummy_pdf = tmp_path / "test.pdf"
        dummy_pdf.write_bytes(b"%PDF-1.4 dummy")

        adapter = ParserAdapter(parse_callable=invalid_parse)

        with pytest.raises(SchemaValidationError):
            adapter.parse(dummy_pdf)

    def test_adapter_default_is_stub(self, tmp_path):
        """Test that adapter defaults to stub parser."""
        dummy_pdf = tmp_path / "test.pdf"
        dummy_pdf.write_bytes(b"%PDF-1.4 dummy")

        # Create adapter with default (should be stub)
        adapter = ParserAdapter()
        output = adapter.parse(dummy_pdf)

        assert output["parser_version"] == "0.0.1-stub"
        assert "STUB_OUTPUT" in [w["code"] for w in output.get("warnings", [])]

    def test_adapter_preserves_pdf_path_info(self, tmp_path):
        """Test that adapter correctly passes PDF path to parser."""
        dummy_pdf = tmp_path / "specific_name.pdf"
        dummy_pdf.write_bytes(b"%PDF-1.4")

        adapter = ParserAdapter(parse_callable=stub_parse)
        output = adapter.parse(dummy_pdf)

        # Stub parser should have used the filename
        assert "specific_name" in output["source"]["filename"]
