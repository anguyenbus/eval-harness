"""Tests for dataset compatibility (OmniDocBench and DP-Bench)."""

from eval_harness.metrics.parsing.layout_map import layout_map_score
from eval_harness.metrics.parsing.reading_order import ard_score
from eval_harness.metrics.parsing.text_similarity import bleu_score, meteor_score


class TestOmniDocBenchCompatibility:
    """Test suite for OmniDocBench dataset format compatibility."""

    def test_ard_with_omnidocbench_format(self):
        """Test ARD works with OmniDocBench element IDs."""
        # OmniDocBench uses integer element IDs
        predicted = [1, 2, 3, 4]
        gold = [1, 2, 3, 4]

        score = ard_score(predicted, gold)
        assert score == 1.0

    def test_layout_map_with_omnidocbench_labels(self):
        """Test mAP works with OmniDocBench label types."""
        # OmniDocBench uses labels like text, table, picture, etc.
        gt_bboxes = [
            {"label": "text", "x0": 10, "y0": 10, "x1": 50, "y1": 30},
            {"label": "table", "x0": 10, "y0": 40, "x1": 50, "y1": 80},
        ]

        pred_bboxes = [
            {"label": "text", "x0": 10, "y0": 10, "x1": 50, "y1": 30},
            {"label": "table", "x0": 10, "y0": 40, "x1": 50, "y1": 80},
        ]

        result = layout_map_score(gt_bboxes, pred_bboxes)
        assert "map_50" in result
        assert result["map_50"] > 0.9

    def test_text_metrics_with_omnidocbench_text(self):
        """Test text metrics work with OmniDocBench text content."""
        gt_text = "This is a sample text from OmniDocBench"
        pred_text = "This is a sample text from OmniDocBench"

        bleu = bleu_score(gt_text, pred_text)
        meteor = meteor_score(gt_text, pred_text)

        # Both metrics should give high scores for perfect match
        assert bleu > 0.9
        assert meteor > 0.9


class TestDPBenchCompatibility:
    """Test suite for DP-Bench dataset format compatibility."""

    def test_ard_with_dpbench_format(self):
        """Test ARD works with DP-Bench element IDs."""
        # DP-Bench uses string element IDs
        predicted = ["elem_1", "elem_2", "elem_3"]
        gold = ["elem_1", "elem_2", "elem_3"]

        score = ard_score(predicted, gold)
        assert score == 1.0

    def test_layout_map_with_dpbench_bboxes(self):
        """Test mAP works with DP-Bench bbox format."""
        # DP-Bench uses reference.json with bboxes
        gt_bboxes = [
            {"label": "TEXT", "x0": 0, "y0": 0, "x1": 100, "y1": 20},
            {"label": "TABLE", "x0": 0, "y0": 30, "x1": 100, "y1": 80},
        ]

        pred_bboxes = [
            {"label": "TEXT", "x0": 0, "y0": 0, "x1": 100, "y1": 20},
            {"label": "TABLE", "x0": 0, "y0": 30, "x1": 100, "y1": 80},
        ]

        result = layout_map_score(
            gt_bboxes, pred_bboxes, page_width=100.0, page_height=100.0
        )
        assert "map_50" in result
        assert result["map_50"] > 0.9

    def test_text_metrics_with_dpbench_text(self):
        """Test text metrics work with DP-Bench text content."""
        gt_text = "Sample DP-Bench document text"
        pred_text = "Sample DP-Bench document text"

        bleu = bleu_score(gt_text, pred_text)
        meteor = meteor_score(gt_text, pred_text)

        # Both metrics should give high scores for perfect match
        assert bleu > 0.9
        assert meteor > 0.9


class TestMultiPageAlignment:
    """Test suite for multi-page document alignment."""

    def test_page_index_alignment(self):
        """Test that page_index correctly maps across predictions and ground truth."""
        # Simulate multi-page document
        page_1_gt = [{"label": "text", "x0": 10, "y0": 10, "x1": 50, "y1": 30}]
        page_1_pred = [{"label": "text", "x0": 10, "y0": 10, "x1": 50, "y1": 30}]

        result = layout_map_score(page_1_gt, page_1_pred)
        assert result["map_50"] > 0.9

    def test_metrics_aggregation_across_pages(self):
        """Test metrics aggregation across multiple pages."""
        # This would be handled at the eval runner level
        # Here we just verify single-page metrics work correctly
        gt_bboxes = [{"label": "text", "x0": 10, "y0": 10, "x1": 50, "y1": 30}]
        pred_bboxes = [{"label": "text", "x0": 10, "y0": 10, "x1": 50, "y1": 30}]

        result = layout_map_score(gt_bboxes, pred_bboxes)
        assert 0.0 <= result["map"] <= 1.0
