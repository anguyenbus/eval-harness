"""Integration tests for docling-eval metrics integration."""

from eval_harness.metrics.parsing import (
    ard_score,
    bleu_score,
    char_edit_distance,
    layout_map_score,
    meteor_score,
    word_edit_distance,
)


class TestDoclingEvalIntegration:
    """Integration tests for docling-eval metrics with sample data."""

    def test_full_evaluation_pipeline(self):
        """Test full evaluation pipeline with sample document data."""
        # Sample document data mimicking OmniDocBench/DP-Bench structure
        document_text = (
            "This document contains important information for evaluation purposes."
        )

        # Reading order evaluation
        element_ids = [1, 2, 3, 4, 5]
        reading_order = [1, 2, 3, 4, 5]  # Perfect order
        ard = ard_score(reading_order, element_ids)

        # Layout evaluation
        gt_layout = [
            {"label": "text", "x0": 10, "y0": 10, "x1": 100, "y1": 30},
            {"label": "text", "x0": 10, "y0": 40, "x1": 100, "y1": 60},
        ]
        pred_layout = [
            {"label": "text", "x0": 10, "y0": 10, "x1": 100, "y1": 30},
            {"label": "text", "x0": 10, "y0": 40, "x1": 100, "y1": 60},
        ]
        layout_result = layout_map_score(gt_layout, pred_layout)

        # Text similarity evaluation
        pred_text = (
            "This document contains important information for evaluation purposes."
        )
        bleu = bleu_score(document_text, pred_text)
        meteor = meteor_score(document_text, pred_text)
        char_dist = char_edit_distance(document_text, pred_text)
        word_dist = word_edit_distance(document_text, pred_text)

        # Verify all metrics return valid scores
        assert 0.0 <= ard <= 1.0
        assert 0.0 <= layout_result["map_50"] <= 1.0
        assert 0.0 <= bleu <= 1.0
        assert 0.0 <= meteor <= 1.0
        assert 0.0 <= char_dist <= 1.0
        assert 0.0 <= word_dist <= 1.0

        # Perfect match should give high scores
        assert ard > 0.9
        assert layout_result["map_50"] > 0.9
        assert bleu > 0.9
        assert meteor > 0.9
        assert char_dist > 0.9
        assert word_dist > 0.9

    def test_evaluation_with_imperfect_predictions(self):
        """Test evaluation pipeline with imperfect predictions."""
        # Ground truth
        gt_order = [1, 2, 3, 4]
        gt_layout = [
            {"label": "text", "x0": 10, "y0": 10, "x1": 50, "y1": 30},
            {"label": "table", "x0": 10, "y0": 40, "x1": 50, "y1": 80},
        ]
        gt_text = "The quick brown fox jumps over the lazy dog"

        # Predictions (with some errors)
        pred_order = [1, 3, 2, 4]  # Swapped order
        pred_layout = [
            {"label": "text", "x0": 10, "y0": 10, "x1": 50, "y1": 30},
            {
                "label": "table",
                "x0": 12,
                "y0": 42,
                "x1": 52,
                "y1": 82,
            },  # Slightly offset
        ]
        pred_text = "The quick brown fox jumps over the lazy"  # Missing word

        # Calculate metrics
        ard = ard_score(pred_order, gt_order)
        layout_result = layout_map_score(gt_layout, pred_layout)
        bleu = bleu_score(gt_text, pred_text)
        meteor = meteor_score(gt_text, pred_text)

        # All metrics should return valid scores
        assert 0.0 <= ard <= 1.0
        assert 0.0 <= layout_result["map_50"] <= 1.0
        assert 0.0 <= bleu <= 1.0
        assert 0.0 <= meteor <= 1.0

        # Imperfect predictions should give lower than perfect scores
        assert ard < 1.0 or layout_result["map_50"] < 1.0 or bleu < 1.0

    def test_cross_dataset_compatibility(self):
        """Test that metrics work consistently across different dataset formats."""
        # OmniDocBench-style data
        omnidoc_order = [1, 2, 3]
        omnidoc_layout = [
            {"label": "text", "x0": 0, "y0": 0, "x1": 100, "y1": 20},
        ]

        # DP-Bench-style data
        dpbench_order = ["a", "b", "c"]
        dpbench_layout = [
            {"label": "TEXT", "x0": 0, "y0": 0, "x1": 100, "y1": 20},
        ]

        # Metrics should work for both formats
        omnidoc_ard = ard_score(omnidoc_order, omnidoc_order)
        dpbench_ard = ard_score(dpbench_order, dpbench_order)

        omnidoc_map = layout_map_score(omnidoc_layout, omnidoc_layout)
        dpbench_map = layout_map_score(dpbench_layout, dpbench_layout)

        assert omnidoc_ard == 1.0
        assert dpbench_ard == 1.0
        assert omnidoc_map["map_50"] > 0.9
        assert dpbench_map["map_50"] > 0.9

    def test_edge_case_empty_documents(self):
        """Test metrics with empty/minimal document data."""
        # Empty reading order
        assert ard_score([], []) == 1.0

        # Empty layout
        result = layout_map_score([], [])
        assert result["map"] == 1.0

        # Empty text
        assert bleu_score("", "") == 1.0
        assert meteor_score("", "") == 1.0
        assert char_edit_distance("", "") == 1.0
        assert word_edit_distance("", "") == 1.0

    def test_weighted_ard_integration(self):
        """Test weighted ARD with bbox area weights."""
        from eval_harness.metrics.parsing import ard_weighted_score

        # Document with varying element sizes
        order = [1, 2, 3]
        bboxes = [
            {"x0": 0, "y0": 0, "x1": 10, "y1": 10},  # Small element (area=100)
            {"x0": 0, "y0": 0, "x1": 50, "y1": 50},  # Large element (area=2500)
            {"x0": 0, "y0": 0, "x1": 20, "y1": 20},  # Medium element (area=400)
        ]

        # Perfect order should give perfect score
        score = ard_weighted_score(order, order, bboxes)
        assert score == 1.0

        # Swapped order should give lower score
        # Large element swap should penalize more than small element swap
        swapped_order = [2, 1, 3]  # Swap small with large
        score_swapped = ard_weighted_score(swapped_order, order, bboxes)
        assert score_swapped < 1.0
