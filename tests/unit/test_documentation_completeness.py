"""Tests for documentation completeness of new metrics."""

from eval_harness.metrics.parsing.layout_map import (
    layout_map_score,
    normalized,
    scaled,
    to_top_left_origin,
)
from eval_harness.metrics.parsing.reading_order import (
    ard_score,
    ard_weighted_score,
)
from eval_harness.metrics.parsing.text_similarity import (
    bleu_score,
    char_edit_distance,
    meteor_score,
    word_edit_distance,
)


class TestDocumentationCompleteness:
    """Test suite for documentation completeness of new metrics."""

    def test_all_public_functions_have_docstrings(self):
        """Test that all public metric functions have docstrings."""
        public_functions = [
            # Reading order metrics
            ard_score,
            ard_weighted_score,
            # Layout metrics
            layout_map_score,
            to_top_left_origin,
            normalized,
            scaled,
            # Text similarity metrics
            bleu_score,
            meteor_score,
            char_edit_distance,
            word_edit_distance,
        ]

        for func in public_functions:
            assert func.__doc__ is not None, f"{func.__name__} is missing docstring"
            assert len(func.__doc__) > 50, f"{func.__name__} docstring is too short"

    def test_docstrings_follow_google_style(self):
        """Test that docstrings follow Google style format."""
        test_functions = [
            ard_score,
            layout_map_score,
            bleu_score,
        ]

        for func in test_functions:
            doc = func.__doc__
            assert "Args:" in doc, f"{func.__name__} missing Args section"
            assert "Returns:" in doc, f"{func.__name__} missing Returns section"
            # Check for parameter descriptions
            lines = doc.split("\n")
            args_index = next(i for i, line in enumerate(lines) if "Args:" in line)
            # Should have parameter lines after Args:
            if args_index + 1 < len(lines):
                assert (
                    lines[args_index + 1]
                    .strip()
                    .startswith(
                        (func.__code__.co_varnames[0], "gt", "pred", "gt_bboxes"),
                    )
                ), f"{func.__name__} Args section format incorrect"
