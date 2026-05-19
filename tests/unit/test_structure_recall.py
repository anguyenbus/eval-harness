"""Tests for structure recall metric."""

from eval_harness.metrics.parsing.structure_recall import structure_recall

# 24 element types from OmniDocBench spec
ELEMENT_TYPES = [
    "text_block",
    "title",
    "table",
    "figure",
    "equation_isolated",
    "equation_inline",
    "list",
    "list_item",
    "footnote",
    "caption",
    "header",
    "footer",
    "page_number",
    "code_block",
    "stamp",
    "signature",
    "logo",
    "handwriting",
    "noise",
    "line",
    "separator",
    "region",
    "placeholder",
]


class TestStructureRecall:
    """Test suite for structure recall metric."""

    def test_perfect_recall(self):
        """Test recall of 1.0 when all element types are detected."""
        predicted_elements = [{"type": t} for t in ELEMENT_TYPES[:5]]
        gold_elements = [{"type": t} for t in ELEMENT_TYPES[:5]]

        score = structure_recall(
            predicted_elements,
            gold_elements,
            element_types=ELEMENT_TYPES[:5],
        )
        assert score == 1.0

    def test_no_recall(self):
        """Test recall of 0.0 when no element types are detected."""
        predicted_elements = []
        gold_elements = [{"type": "text_block"}, {"type": "table"}]

        score = structure_recall(
            predicted_elements,
            gold_elements,
            element_types=ELEMENT_TYPES,
        )
        assert score == 0.0

    def test_partial_recall(self):
        """Test recall score for partial detection."""
        # Gold has 3 types, predicted has 2
        gold_elements = [
            {"type": "text_block"},
            {"type": "table"},
            {"type": "figure"},
        ]
        predicted_elements = [
            {"type": "text_block"},
            {"type": "table"},
            # Missing "figure"
        ]

        score = structure_recall(
            predicted_elements,
            gold_elements,
            element_types=ELEMENT_TYPES,
        )
        # Detected 2 out of 3 = 0.667
        assert abs(score - 0.667) < 0.01

    def test_deterministic_behavior(self):
        """Test that same inputs produce same output."""
        gold_elements = [{"type": "text_block"}, {"type": "table"}]
        predicted_elements = [{"type": "text_block"}]

        score1 = structure_recall(
            predicted_elements,
            gold_elements,
            element_types=ELEMENT_TYPES,
        )
        score2 = structure_recall(
            predicted_elements,
            gold_elements,
            element_types=ELEMENT_TYPES,
        )
        assert score1 == score2

    def test_empty_gold(self):
        """Test recall when gold has no elements."""
        gold_elements = []
        predicted_elements = [{"type": "text_block"}]

        score = structure_recall(
            predicted_elements,
            gold_elements,
            element_types=ELEMENT_TYPES,
        )
        # Should be 1.0 since nothing to detect
        assert score == 1.0

    def test_all_24_element_types(self):
        """Test that all 24 element types are handled."""
        gold_elements = [{"type": t} for t in ELEMENT_TYPES]
        predicted_elements = [{"type": t} for t in ELEMENT_TYPES]

        score = structure_recall(
            predicted_elements,
            gold_elements,
            element_types=ELEMENT_TYPES,
        )
        assert score == 1.0

    def test_extra_predicted_types_ignored(self):
        """Test that extra predicted types don't affect recall."""
        gold_elements = [{"type": "text_block"}]
        # Predicted has extra types not in gold
        predicted_elements = [
            {"type": "text_block"},
            {"type": "table"},
            {"type": "figure"},
        ]

        score = structure_recall(
            predicted_elements,
            gold_elements,
            element_types=ELEMENT_TYPES,
        )
        # Should be 1.0 since all gold types are detected
        assert score == 1.0
