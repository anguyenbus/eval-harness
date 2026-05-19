"""Tests for OmniDocBench loader."""

from eval_harness.datasets.omnidocbench import load_omnidocbench


class TestOmniDocBenchLoader:
    """Test suite for OmniDocBench dataset loader."""

    def test_loader_yields_english_only_documents(self, tmp_path):
        """Test that loader yields English-only documents."""
        # Create a tiny fixture dataset
        json_data = [
            {
                "page_info": {
                    "page_no": 1,
                    "height": 792,
                    "width": 612,
                    "image_path": "page1.jpg",
                    "page_attribute": {
                        "language": "english",
                        "data_source": "research_report",
                        "fuzzy_scan": False,
                        "watermark": False,
                        "colorful_backgroud": False,
                        "layout": "single_column",
                    },
                },
                "layout_dets": [],
                "extra": {},
            },
            {
                "page_info": {
                    "page_no": 1,
                    "height": 792,
                    "width": 612,
                    "image_path": "page2.jpg",
                    "page_attribute": {
                        "language": "chinese",  # Should be filtered out
                        "data_source": "research_report",
                        "fuzzy_scan": False,
                        "watermark": False,
                        "colorful_backgroud": False,
                        "layout": "single_column",
                    },
                },
                "layout_dets": [],
                "extra": {},
            },
        ]

        json_path = tmp_path / "OmniDocBench.json"
        import json

        json_path.write_text(json.dumps(json_data))

        pages = list(load_omnidocbench(tmp_path))

        # Should only yield English page
        assert len(pages) == 1
        assert pages[0]["page_info"]["page_attribute"]["language"] == "english"

    def test_iterator_pattern_works(self, tmp_path):
        """Test that loader uses iterator pattern for memory efficiency."""
        json_data = [
            {
                "page_info": {
                    "page_no": i,
                    "height": 792,
                    "width": 612,
                    "image_path": f"page{i}.jpg",
                    "page_attribute": {
                        "language": "english",
                        "data_source": "research_report",
                        "fuzzy_scan": False,
                        "watermark": False,
                        "colorful_backgroud": False,
                        "layout": "single_column",
                    },
                },
                "layout_dets": [],
                "extra": {},
            }
            for i in range(5)
        ]

        json_path = tmp_path / "OmniDocBench.json"
        import json

        json_path.write_text(json.dumps(json_data))

        # Should return iterator, not list
        result = load_omnidocbench(tmp_path)
        assert hasattr(result, "__iter__")

        # Can iterate once (generator pattern)
        first = sum(1 for _ in result)
        assert first == 5

        # Can create new iterator to read again
        second = sum(1 for _ in load_omnidocbench(tmp_path))
        assert second == 5

    def test_eval_tags_metadata_attached(self, tmp_path):
        """Test that _eval_tags metadata is attached to each page."""
        json_data = [
            {
                "page_info": {
                    "page_no": 1,
                    "height": 792,
                    "width": 612,
                    "image_path": "page1.jpg",
                    "page_attribute": {
                        "language": "english",
                        "data_source": "research_report",
                        "fuzzy_scan": True,  # Noisy scan
                        "watermark": False,
                        "colorful_backgroud": False,
                        "layout": "single_column",
                    },
                },
                "layout_dets": [],
                "extra": {},
            },
        ]

        json_path = tmp_path / "OmniDocBench.json"
        import json

        json_path.write_text(json.dumps(json_data))

        pages = list(load_omnidocbench(tmp_path))

        assert "_eval_tags" in pages[0]
        assert pages[0]["_eval_tags"]["has_fuzzy_scan"] is True
        assert pages[0]["_eval_tags"]["is_clean"] is False

    def test_filters_relevant_doc_types(self, tmp_path):
        """Test that loader filters to relevant document types only."""
        json_data = [
            {
                "page_info": {
                    "page_no": 1,
                    "height": 792,
                    "width": 612,
                    "image_path": "page1.jpg",
                    "page_attribute": {
                        "language": "english",
                        "data_source": "research_report",  # Relevant
                        "fuzzy_scan": False,
                        "watermark": False,
                        "colorful_backgroud": False,
                        "layout": "single_column",
                    },
                },
                "layout_dets": [],
                "extra": {},
            },
            {
                "page_info": {
                    "page_no": 2,
                    "height": 792,
                    "width": 612,
                    "image_path": "page2.jpg",
                    "page_attribute": {
                        "language": "english",
                        "data_source": "newspaper",  # Not relevant
                        "fuzzy_scan": False,
                        "watermark": False,
                        "colorful_backgroud": False,
                        "layout": "single_column",
                    },
                },
                "layout_dets": [],
                "extra": {},
            },
        ]

        json_path = tmp_path / "OmniDocBench.json"
        import json

        json_path.write_text(json.dumps(json_data))

        pages = list(load_omnidocbench(tmp_path))

        # Should only yield research_report
        assert len(pages) == 1
        assert (
            pages[0]["page_info"]["page_attribute"]["data_source"] == "research_report"
        )
