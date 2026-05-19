"""Tests for stub parser implementation."""

from pathlib import Path

from eval_harness.adapters.schema_validator import validate
from eval_harness.stubs.stub_parser import parse


class TestStubParser:
    """Test suite for stub parser."""

    def test_stub_output_validates_against_schema(self, tmp_path):
        """Test that stub parser output validates against parser_output schema."""
        # Create a dummy PDF file
        dummy_pdf = tmp_path / "test.pdf"
        dummy_pdf.write_bytes(b"%PDF-1.4 dummy")

        output = parse(dummy_pdf)

        schema_path = Path("contracts/parser_output.schema.json")
        validate(output, schema_path)

    def test_stub_includes_required_fields(self, tmp_path):
        """Test that stub output includes all required fields."""
        dummy_pdf = tmp_path / "test.pdf"
        dummy_pdf.write_bytes(b"%PDF-1.4 dummy")

        output = parse(dummy_pdf)

        # Check required top-level fields
        assert "schema_version" in output
        assert output["schema_version"] == "1.0.0"
        assert "parser_version" in output
        assert output["parser_version"] == "0.0.1-stub"
        assert "source" in output
        assert "pages" in output
        assert "elements" in output
        assert "warnings" in output

    def test_stub_has_stub_output_warning(self, tmp_path):
        """Test that stub output includes STUB_OUTPUT warning code."""
        dummy_pdf = tmp_path / "test.pdf"
        dummy_pdf.write_bytes(b"%PDF-1.4 dummy")

        output = parse(dummy_pdf)

        assert "warnings" in output
        assert len(output["warnings"]) > 0
        warning_codes = [w["code"] for w in output["warnings"]]
        assert "STUB_OUTPUT" in warning_codes

    def test_stub_works_with_different_pdf_paths(self, tmp_path):
        """Test that stub works with different PDF file inputs."""
        pdf1 = tmp_path / "doc1.pdf"
        pdf2 = tmp_path / "doc2.pdf"
        pdf1.write_bytes(b"%PDF-1.4 doc1")
        pdf2.write_bytes(b"%PDF-1.4 doc2")

        output1 = parse(pdf1)
        output2 = parse(pdf2)

        # Should both be valid
        assert output1["schema_version"] == "1.0.0"
        assert output2["schema_version"] == "1.0.0"

        # Source doc_id should reflect filename
        assert "doc1" in output1["source"]["filename"]
        assert "doc2" in output2["source"]["filename"]

    def test_stub_elements_have_valid_structure(self, tmp_path):
        """Test that stub output elements have valid structure."""
        dummy_pdf = tmp_path / "test.pdf"
        dummy_pdf.write_bytes(b"%PDF-1.4 dummy")

        output = parse(dummy_pdf)

        assert len(output["elements"]) >= 2

        # Check heading element
        heading = next(e for e in output["elements"] if e["type"] == "heading")
        assert "element_id" in heading
        assert "char_span" in heading
        assert len(heading["char_span"]) == 2
        assert "text" in heading
        assert heading["level"] == 1

        # Check paragraph element
        paragraph = next(e for e in output["elements"] if e["type"] == "paragraph")
        assert "element_id" in paragraph
        assert "char_span" in paragraph
        assert "text" in paragraph
