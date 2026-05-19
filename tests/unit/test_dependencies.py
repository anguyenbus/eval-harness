"""Tests for new docling-eval dependencies.

This test module verifies that all new dependencies for docling-eval
metrics integration can be imported successfully.
"""


class TestDependencyImports:
    """Test suite for new dependency imports."""

    def test_torch_cpu_import(self):
        """Test that torch imports work on CPU-only."""
        import torch

        # Verify CPU is available
        assert torch.device("cpu") is not None

        # Verify we can create a simple tensor on CPU
        tensor = torch.tensor([1.0, 2.0, 3.0], device="cpu")
        assert tensor.device.type == "cpu"
        assert len(tensor) == 3

    def test_torchmetrics_import(self):
        """Test that torchmetrics imports work."""
        from torchmetrics.detection.mean_ap import MeanAveragePrecision

        # Verify we can instantiate the metric
        metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True)
        assert metric is not None

    def test_nltk_import_and_download(self):
        """Test that nltk imports work and data can be downloaded."""
        import nltk

        # Verify core nltk modules are available
        assert hasattr(nltk, "word_tokenize")
        assert hasattr(nltk, "edit_distance")
        assert hasattr(nltk, "download")

        # Download required NLTK data (idempotent)
        nltk.download("popular", quiet=True)
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
        nltk.download("wordnet", quiet=True)

    def test_evaluate_library_import(self):
        """Test that HuggingFace evaluate library imports work."""
        import evaluate

        # Verify we can load BLEU metric
        bleu = evaluate.load("bleu")
        assert bleu is not None
        assert hasattr(bleu, "compute")
