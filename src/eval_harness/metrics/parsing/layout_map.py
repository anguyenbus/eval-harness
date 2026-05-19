"""
mAP (mean Average Precision) layout detection metrics.

This module implements mAP metrics for layout detection evaluation using
torchmetrics. Supports mAP@0.5, mAP@0.75, and mAP@0.5:0.95 with per-class
breakdown for element types (text, table, picture, etc.).

All operations are CPU-only to avoid GPU dependencies.
"""

from __future__ import annotations

import torch
from beartype import beartype
from beartype.typing import Any, Final

# NOTE: torchmetrics import is local to avoid circular imports
# and moved inside function that uses it

# Constants for mAP calculation
_DEVICE: Final[str] = "cpu"
_IOU_TYPE: Final[str] = "bbox"
_DEFAULT_SCALE: Final[float] = 100.0


def to_top_left_origin(
    bbox: dict[str, Any],
    page_height: float,
) -> dict[str, float]:
    """
    Convert bounding box to top-left origin coordinate system.

    NOTE: Docling uses bottom-left origin by default. This conversion
    is necessary for compatibility with torchmetrics which expects
    top-left origin (standard in computer vision).

    Args:
        bbox: Dictionary with x0, y0, x1, y1 keys.
        page_height: Height of the page for coordinate conversion.

    Returns:
        Dictionary with converted coordinates in top-left origin.

    """
    x0 = bbox.get("x0", 0)
    y0 = bbox.get("y0", 0)
    x1 = bbox.get("x1", 0)
    y1 = bbox.get("y1", 0)

    # Convert from bottom-left to top-left origin
    y0_converted = page_height - y1
    y1_converted = page_height - y0

    return {
        "x0": float(x0),
        "y0": float(y0_converted),
        "x1": float(x1),
        "y1": float(y1_converted),
    }


def normalized(
    bbox: dict[str, Any],
    page_size: dict[str, float],
) -> dict[str, float]:
    """
    Normalize bounding box coordinates to [0, 1] range.

    Args:
        bbox: Dictionary with x0, y0, x1, y1 keys.
        page_size: Dictionary with width and height keys.

    Returns:
        Dictionary with normalized coordinates in [0, 1] range.

    """
    page_width = page_size.get("width", 1.0)
    page_height = page_size.get("height", 1.0)

    x0 = bbox.get("x0", 0) / page_width
    y0 = bbox.get("y0", 0) / page_height
    x1 = bbox.get("x1", 0) / page_width
    y1 = bbox.get("y1", 0) / page_height

    return {
        "x0": float(x0),
        "y0": float(y0),
        "x1": float(x1),
        "y1": float(y1),
    }


def scaled(
    bbox: dict[str, Any],
    scale: float = _DEFAULT_SCALE,
) -> dict[str, float]:
    """
    Scale bounding box coordinates by the given factor.

    NOTE: Following docling-eval pattern, bboxes are scaled to 100.0
    for normalization purposes. This ensures consistent evaluation
    across different page sizes.

    Args:
        bbox: Dictionary with x0, y0, x1, y1 keys (normalized [0,1]).
        scale: Scale factor (default 100.0).

    Returns:
        Dictionary with scaled coordinates.

    """
    x0 = bbox.get("x0", 0) * scale
    y0 = bbox.get("y0", 0) * scale
    x1 = bbox.get("x1", 0) * scale
    y1 = bbox.get("y1", 0) * scale

    return {
        "x0": float(x0),
        "y0": float(y0),
        "x1": float(x1),
        "y1": float(y1),
    }


def _extract_labels(
    bboxes: list[dict[str, Any]],
) -> list[str]:
    """
    Extract unique labels from bounding boxes.

    Args:
        bboxes: List of bounding box dictionaries.

    Returns:
        List of unique label strings.

    """
    labels = set()
    for bbox in bboxes:
        label = bbox.get("label", "unknown")
        labels.add(label)
    return sorted(labels)


def _bbox_to_tensor_dict(
    bboxes: list[dict[str, Any]],
    label_map: dict[str, int],
    page_size: dict[str, float],
) -> dict[str, torch.Tensor]:
    """
    Convert bounding boxes to torch tensor format for torchmetrics.

    Args:
        bboxes: List of bounding box dictionaries.
        label_map: Mapping from label strings to integer indices.
        page_size: Dictionary with width and height for normalization.

    Returns:
        Dictionary with torch tensors for boxes, labels, and scores.

    """
    boxes_list = []
    labels_list = []
    scores_list = []

    for bbox in bboxes:
        # Get coordinates (assume already in top-left origin)
        x0 = float(bbox.get("x0", 0))
        y0 = float(bbox.get("y0", 0))
        x1 = float(bbox.get("x1", 0))
        y1 = float(bbox.get("y1", 0))

        # Normalize and scale to 100.0 (docling-eval pattern)
        norm_bbox = normalized({"x0": x0, "y0": y0, "x1": x1, "y1": y1}, page_size)
        scaled_bbox = scaled(norm_bbox, _DEFAULT_SCALE)

        boxes_list.append(
            [scaled_bbox["x0"], scaled_bbox["y0"], scaled_bbox["x1"], scaled_bbox["y1"]]
        )

        # Map label to integer
        label = bbox.get("label", "unknown")
        label_idx = label_map.get(label, 0)
        labels_list.append(label_idx)

        # Use default score of 1.0 for ground truth
        scores_list.append(1.0)

    if not boxes_list:
        return {
            "boxes": torch.empty(0, 4, dtype=torch.float32, device=_DEVICE),
            "labels": torch.empty(0, dtype=torch.long, device=_DEVICE),
            "scores": torch.empty(0, dtype=torch.float32, device=_DEVICE),
        }

    return {
        "boxes": torch.tensor(boxes_list, dtype=torch.float32, device=_DEVICE),
        "labels": torch.tensor(labels_list, dtype=torch.long, device=_DEVICE),
        "scores": torch.tensor(scores_list, dtype=torch.float32, device=_DEVICE),
    }


@beartype
def layout_map_score(
    gt_bboxes: list[dict[str, Any]],
    pred_bboxes: list[dict[str, Any]],
    iou_thresholds: list[float] | None = None,  # noqa: ARG001
    page_width: float = 100.0,
    page_height: float = 100.0,
) -> dict[str, Any]:
    """
    Calculate mAP (mean Average Precision) for layout detection.

    Computes mAP@0.5, mAP@0.75, and mAP@0.5:0.95 (mean across thresholds)
    using torchmetrics. Supports per-class mAP breakdown.

    NOTE: This function follows docling-eval conventions:
        - Bboxes are converted to top-left origin
        - Coordinates are normalized to [0, 1] then scaled to 100.0
        - PENALIZE strategy is used for missing predictions (empty tensors)
        - CPU-only operations (no GPU dependency)

    Args:
        gt_bboxes: Ground truth bounding boxes. Each dict should have:
                   label (str), x0, y0, x1, y1, page_no (int).
        pred_bboxes: Predicted bounding boxes with same structure as gt_bboxes.
        iou_thresholds: List of IoU thresholds (None uses default 0.5:0.95).
        page_width: Page width for normalization (default 100.0).
        page_height: Page height for normalization (default 100.0).

    Returns:
        Dictionary with keys:
            - map: mAP@0.5:0.95 (mean across thresholds)
            - map_50: mAP@0.5
            - map_75: mAP@0.75
            - map_per_class: Dict mapping label names to per-class mAP

    Examples:
        >>> gt = [{"label": "text", "x0": 10, "y0": 10, "x1": 50, "y1": 30}]
        >>> pred = [{"label": "text", "x0": 10, "y0": 10, "x1": 50, "y1": 30}]
        >>> result = layout_map_score(gt, pred)
        >>> result["map_50"] > 0.9
        True

    """
    from torchmetrics.detection.mean_ap import MeanAveragePrecision

    # Handle empty inputs
    if not gt_bboxes and not pred_bboxes:
        return {
            "map": 1.0,
            "map_50": 1.0,
            "map_75": 1.0,
            "map_per_class": {},
        }

    if not gt_bboxes:
        return {
            "map": 0.0,
            "map_50": 0.0,
            "map_75": 0.0,
            "map_per_class": {},
        }

    # Build label mapping
    all_labels = _extract_labels(gt_bboxes + pred_bboxes)
    label_map = {label: idx for idx, label in enumerate(all_labels)}

    # Page size for normalization
    page_size = {"width": page_width, "height": page_height}

    # Convert to torch tensor format
    gt_tensors = _bbox_to_tensor_dict(gt_bboxes, label_map, page_size)
    pred_tensors = _bbox_to_tensor_dict(pred_bboxes, label_map, page_size)

    # Initialize metric with CPU device
    metric = MeanAveragePrecision(
        iou_type=_IOU_TYPE,
        class_metrics=True,
    )

    # Update metric with predictions
    try:
        metric.update([pred_tensors], [gt_tensors])
        result = metric.compute()

        # Extract results
        map_value = float(result.get("map", 0.0))
        map_50 = float(result.get("map_50", 0.0))
        map_75 = float(result.get("map_75", 0.0))

        # Extract per-class mAP if available
        # NOTE: map_per_class can be a 0-d tensor for single class
        map_per_class = {}
        if "map_per_class" in result:
            map_per_class_tensor = result["map_per_class"]
            # Handle 0-d tensor (single class) vs 1-d tensor (multiple classes)
            if hasattr(map_per_class_tensor, "dim") and map_per_class_tensor.dim() == 0:
                # Single class - wrap in list
                map_per_class[all_labels[0]] = float(map_per_class_tensor)
            elif hasattr(map_per_class_tensor, "tolist"):
                # Multiple classes - convert to list
                for label_idx, class_map in enumerate(map_per_class_tensor.tolist()):
                    if label_idx < len(all_labels):
                        label_name = all_labels[label_idx]
                        map_per_class[label_name] = float(class_map)
            else:
                # Fallback for other iterable types
                for label_idx, class_map in enumerate(map_per_class_tensor):
                    if label_idx < len(all_labels):
                        label_name = all_labels[label_idx]
                        map_per_class[label_name] = float(class_map)

    finally:
        metric.reset()

    return {
        "map": map_value,
        "map_50": map_50,
        "map_75": map_75,
        "map_per_class": map_per_class,
    }
