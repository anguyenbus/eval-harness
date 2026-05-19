"""Tests for DP-Bench loader."""

from pathlib import Path

from eval_harness.datasets.dp_bench import load_dp_bench


class TestDPBenchLoader:
    """Test suite for DP-Bench dataset loader."""

    def test_loader_reads_reference_json_files(self, tmp_path):
        """Test that loader reads reference.json files."""
        # Create tiny fixture dataset (matching actual structure)
        dataset_dir = tmp_path / "dataset"
        pdfs_dir = dataset_dir / "pdfs"
        pdfs_dir.mkdir(parents=True)

        ref_data = {
            "doc001.pdf": {
                "elements": [
                    {
                        "category": "Paragraph",
                        "content": {"text": "Sample text"},
                        "page": 1,
                    },
                ],
            },
        }

        import json

        (dataset_dir / "reference.json").write_text(json.dumps(ref_data))

        # Create dummy PDF
        (pdfs_dir / "doc001.pdf").write_bytes(b"%PDF-1.4")

        results = list(load_dp_bench(tmp_path))

        assert len(results) == 1
        doc_id, pdf_path, gold_elements = results[0]
        assert doc_id == "doc001"
        assert "elements" in gold_elements

    def test_iterator_pattern_works(self, tmp_path):
        """Test that loader uses iterator pattern."""
        # Create multiple documents
        dataset_dir = tmp_path / "dataset"
        pdfs_dir = dataset_dir / "pdfs"
        pdfs_dir.mkdir(parents=True)

        import json

        ref_data = {}
        for i in range(3):
            doc_name = f"doc{i:03d}.pdf"
            ref_data[doc_name] = {"elements": []}
            (pdfs_dir / doc_name).write_bytes(b"%PDF-1.4")

        (dataset_dir / "reference.json").write_text(json.dumps(ref_data))

        result = load_dp_bench(tmp_path)
        assert hasattr(result, "__iter__")

        count = sum(1 for _ in result)
        assert count == 3

    def test_configurable_root_path(self, tmp_path):
        """Test that loader accepts configurable root path."""
        # Create dataset in subdirectory
        data_dir = tmp_path / "parsing" / "dp_bench"
        dataset_dir = data_dir / "dataset"
        pdfs_dir = dataset_dir / "pdfs"
        pdfs_dir.mkdir(parents=True)

        import json

        ref_data = {"doc001.pdf": {"elements": []}}
        (dataset_dir / "reference.json").write_text(json.dumps(ref_data))
        (pdfs_dir / "doc001.pdf").write_bytes(b"%PDF-1.4")

        results = list(load_dp_bench(data_dir))

        assert len(results) == 1

    def test_returns_correct_tuple_structure(self, tmp_path):
        """Test that loader returns (doc_id, pdf_path, gold_elements) tuples."""
        dataset_dir = tmp_path / "dataset"
        pdfs_dir = dataset_dir / "pdfs"
        pdfs_dir.mkdir(parents=True)

        import json

        ref_data = {
            "testdoc.pdf": {
                "elements": [
                    {
                        "category": "Header",
                        "content": {"text": "Test Title"},
                        "page": 1,
                    },
                    {"category": "Table", "content": {"text": "Table data"}, "page": 1},
                ],
            },
        }

        (dataset_dir / "reference.json").write_text(json.dumps(ref_data))
        (pdfs_dir / "testdoc.pdf").write_bytes(b"%PDF-1.4")

        results = list(load_dp_bench(tmp_path))

        assert len(results) == 1
        doc_id, pdf_path, gold_elements = results[0]

        assert isinstance(doc_id, str)
        assert isinstance(pdf_path, Path)
        assert isinstance(gold_elements, dict)
        assert len(gold_elements["elements"]) == 2
