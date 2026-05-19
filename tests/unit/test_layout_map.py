"""Tests for mAP (mean Average Precision) layout detection metrics."""

from eval_harness.metrics.parsing.layout_map import (
    layout_map_score,
    normalized,
    scaled,
    to_top_left_origin,
)


class TestLayoutMAPScore:
    """Test suite for layout mAP metrics."""

    def test_map_50_with_mock_data(self):
        """Test mAP@0.5 with mock bounding boxes."""
        # Mock ground truth and predictions with perfect overlap
        gt_bboxes = [
            {"label": "text", "x0": 10, "y0": 10, "x1": 50, "y1": 30, "page_no": 1},
            {"label": "table", "x0": 10, "y0": 40, "x1": 50, "y1": 80, "page_no": 1},
        ]

        pred_bboxes = [
            {"label": "text", "x0": 10, "y0": 10, "x1": 50, "y1": 30, "page_no": 1},
            {"label": "table", "x0": 10, "y0": 40, "x1": 50, "y1": 80, "page_no": 1},
        ]

        result = layout_map_score(gt_bboxes, pred_bboxes)
        assert "map_50" in result
        assert result["map_50"] > 0.9  # Perfect match should be close to 1.0

    def test_map_75_with_mock_data(self):
        """Test mAP@0.75 with mock bounding boxes."""
        gt_bboxes = [
            {"label": "text", "x0": 10, "y0": 10, "x1": 50, "y1": 30, "page_no": 1},
        ]

        pred_bboxes = [
            {"label": "text", "x0": 10, "y0": 10, "x1": 50, "y1": 30, "page_no": 1},
        ]

        result = layout_map_score(gt_bboxes, pred_bboxes)
        assert "map_75" in result
        assert result["map_75"] > 0.9

    def test_empty_inputs(self):
        """Test mAP with empty inputs."""
        result = layout_map_score([], [])
        assert "map" in result
        # Empty predictions should penalize score
        assert result["map"] >= 0.0

    def test_per_class_map(self):
        """Test per-class mAP breakdown."""
        gt_bboxes = [
            {"label": "text", "x0": 10, "y0": 10, "x1": 50, "y1": 30, "page_no": 1},
            {"label": "table", "x0": 10, "y0": 40, "x1": 50, "y1": 80, "page_no": 1},
        ]

        pred_bboxes = [
            {"label": "text", "x0": 10, "y0": 10, "x1": 50, "y1": 30, "page_no": 1},
            {"label": "table", "x0": 10, "y0": 40, "x1": 50, "y1": 80, "page_no": 1},
        ]

        result = layout_map_score(gt_bboxes, pred_bboxes)
        assert "map_per_class" in result
        assert isinstance(result["map_per_class"], dict)

    def test_bbox_conversion(self):
        """Test bbox coordinate conversion utilities."""
        # Test to_top_left_origin
        bbox = {"x0": 10, "y0": 10, "x1": 50, "y1": 30, "page_height": 100}
        converted = to_top_left_origin(bbox, page_height=100)
        assert "x0" in converted
        assert "y0" in converted

        # Test normalized
        bbox_norm = {"x0": 10, "y0": 10, "x1": 50, "y1": 30}
        page_size = {"width": 100, "height": 100}
        normalized_bbox = normalized(bbox_norm, page_size)
        assert 0.0 <= normalized_bbox["x0"] <= 1.0
        assert 0.0 <= normalized_bbox["x1"] <= 1.0

        # Test scaled
        bbox_scaled = {"x0": 0.1, "y0": 0.1, "x1": 0.5, "y1": 0.3}
        scaled_bbox = scaled(bbox_scaled, scale=100.0)
        assert scaled_bbox["x0"] == 10.0
        assert scaled_bbox["x1"] == 50.0

    def test_missing_predictions_penalty(self):
        """Test that missing predictions are penalized (PENALIZE strategy)."""
        gt_bboxes = [
            {"label": "text", "x0": 10, "y0": 10, "x1": 50, "y1": 30, "page_no": 1},
        ]

        # Empty prediction should result in low score
        result = layout_map_score(gt_bboxes, [])
        assert result["map"] < 0.5  # Should penalize heavily

    def test_iou_threshold_range(self):
        """Test mAP@0.5:0.95 (mean across thresholds)."""
        gt_bboxes = [
            {"label": "text", "x0": 10, "y0": 10, "x1": 50, "y1": 30, "page_no": 1},
        ]

        pred_bboxes = [
            {"label": "text", "x0": 10, "y0": 10, "x1": 50, "y1": 30, "page_no": 1},
        ]

        result = layout_map_score(gt_bboxes, pred_bboxes)
        assert "map" in result  # map is the mean across thresholds 0.5:0.95
        assert 0.0 <= result["map"] <= 1.0
