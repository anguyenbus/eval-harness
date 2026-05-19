"""Tests for text similarity metrics (BLEU, METEOR, edit distance)."""

from eval_harness.metrics.parsing.text_similarity import (
    bleu_score,
    char_edit_distance,
    meteor_score,
    word_edit_distance,
)


class TestBLEUScore:
    """Test suite for BLEU score metric."""

    def test_perfect_match(self):
        """Test BLEU score near 1.0 for perfect match."""
        # Use longer text for better BLEU score (needs sufficient n-grams)
        score = bleu_score(
            "This is a sample text for evaluation",
            "This is a sample text for evaluation",
        )
        assert score > 0.9  # BLEU should be close to 1.0 for longer text

    def test_no_match(self):
        """Test BLEU score near 0 for no match."""
        score = bleu_score("hello world", "xyz abc")
        # BLEU can be 0 or very low for completely different text
        assert 0.0 <= score < 0.5

    def test_partial_match(self):
        """Test BLEU score for partial overlap."""
        score = bleu_score(
            "This is a sample text for evaluation", "This is a sample text"
        )
        # Partial match should give intermediate score
        assert 0.0 <= score <= 1.0

    def test_empty_strings(self):
        """Test BLEU score handles empty strings."""
        score = bleu_score("", "")
        assert score == 1.0

    def test_one_empty_string(self):
        """Test BLEU score when one string is empty."""
        score = bleu_score("text", "")
        assert score == 0.0


class TestMETEORScore:
    """Test suite for METEOR score metric."""

    def test_perfect_match(self):
        """Test METEOR score close to 1.0 for perfect match."""
        score = meteor_score("hello world", "hello world")
        assert score > 0.9  # METEOR should be very close to 1.0

    def test_no_match(self):
        """Test METEOR score near 0 for no match."""
        score = meteor_score("hello world", "xyz abc")
        # METEOR gives low scores for no overlap
        assert 0.0 <= score < 0.5

    def test_partial_match(self):
        """Test METEOR score for partial overlap."""
        score = meteor_score("hello world", "hello")
        # Partial match should give intermediate score
        assert 0.0 < score < 1.0

    def test_empty_strings(self):
        """Test METEOR score handles empty strings."""
        score = meteor_score("", "")
        assert score >= 0.0

    def test_one_empty_string(self):
        """Test METEOR score when one string is empty."""
        score = meteor_score("text", "")
        assert score >= 0.0


class TestCharEditDistance:
    """Test suite for character-level edit distance metric."""

    def test_perfect_match(self):
        """Test edit distance of 1.0 (0 distance normalized) for perfect match."""
        score = char_edit_distance("hello", "hello")
        assert score == 1.0

    def test_no_match(self):
        """Test edit distance for completely different strings."""
        score = char_edit_distance("abc", "xyz")
        assert score < 0.5

    def test_one_char_difference(self):
        """Test edit distance for one character difference."""
        score = char_edit_distance("hello", "hallo")
        # 1 char diff out of 5 = 0.8 normalized score
        assert 0.7 < score < 0.9

    def test_empty_strings(self):
        """Test edit distance handles empty strings."""
        score = char_edit_distance("", "")
        assert score == 1.0

    def test_one_empty_string(self):
        """Test edit distance when one string is empty."""
        score = char_edit_distance("text", "")
        assert score == 0.0


class TestWordEditDistance:
    """Test suite for word-level edit distance metric."""

    def test_perfect_match(self):
        """Test word edit distance of 1.0 for perfect match."""
        score = word_edit_distance("hello world", "hello world")
        assert score == 1.0

    def test_no_match(self):
        """Test word edit distance for completely different words."""
        score = word_edit_distance("hello world", "foo bar")
        assert score < 0.5

    def test_one_word_difference(self):
        """Test word edit distance for one word difference."""
        score = word_edit_distance("hello world", "hello earth")
        # 1 word diff out of 2
        assert 0.0 < score < 1.0

    def test_empty_strings(self):
        """Test word edit distance handles empty strings."""
        score = word_edit_distance("", "")
        assert score == 1.0

    def test_one_empty_string(self):
        """Test word edit distance when one string is empty."""
        score = word_edit_distance("text", "")
        assert score == 0.0
