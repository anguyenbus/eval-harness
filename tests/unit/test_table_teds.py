"""Tests for Table TEDS metric."""

from eval_harness.metrics.parsing.table_teds import table_teds


class TestTableTEDS:
    """Test suite for Tree Edit Distance Similarity for tables."""

    def test_identical_tables(self):
        """Test TEDS of 1.0 for identical tables."""
        table1 = {
            "rows": 2,
            "cols": 2,
            "cells": [
                {"row": 0, "col": 0, "text": "A1"},
                {"row": 0, "col": 1, "text": "B1"},
                {"row": 1, "col": 0, "text": "A2"},
                {"row": 1, "col": 1, "text": "B2"},
            ],
        }

        score = table_teds(table1, table1)
        assert score == 1.0

    def test_different_structure(self):
        """Test TEDS for tables with different structure."""
        predicted = {
            "rows": 2,
            "cols": 2,
            "cells": [
                {"row": 0, "col": 0, "text": "A1"},
                {"row": 0, "col": 1, "text": "B1"},
                {"row": 1, "col": 0, "text": "A2"},
                {"row": 1, "col": 1, "text": "B2"},
            ],
        }

        gold = {
            "rows": 3,  # Different row count
            "cols": 2,
            "cells": [
                {"row": 0, "col": 0, "text": "A1"},
                {"row": 0, "col": 1, "text": "B1"},
                {"row": 1, "col": 0, "text": "A2"},
                {"row": 1, "col": 1, "text": "B2"},
                {"row": 2, "col": 0, "text": "A3"},
                {"row": 2, "col": 1, "text": "B3"},
            ],
        }

        score = table_teds(predicted, gold)
        # Should be less than 1.0 due to missing row
        assert 0 <= score < 1.0

    def test_same_structure_different_content(self):
        """Test TEDS for tables with same structure but different content."""
        predicted = {
            "rows": 2,
            "cols": 2,
            "cells": [
                {"row": 0, "col": 0, "text": "A1"},
                {"row": 0, "col": 1, "text": "B1"},
                {"row": 1, "col": 0, "text": "A2"},
                {"row": 1, "col": 1, "text": "B2"},
            ],
        }

        gold = {
            "rows": 2,
            "cols": 2,
            "cells": [
                {"row": 0, "col": 0, "text": "X1"},  # Different content
                {"row": 0, "col": 1, "text": "B1"},
                {"row": 1, "col": 0, "text": "A2"},
                {"row": 1, "col": 1, "text": "Y2"},  # Different content
            ],
        }

        score = table_teds(predicted, gold)
        # Should be between 0 and 1
        assert 0 < score < 1.0

    def test_empty_tables(self):
        """Test TEDS for empty tables."""
        table = {"rows": 0, "cols": 0, "cells": []}
        score = table_teds(table, table)
        assert score == 1.0

    def test_deterministic_behavior(self):
        """Test that same inputs produce same output."""
        table1 = {
            "rows": 2,
            "cols": 2,
            "cells": [
                {"row": 0, "col": 0, "text": "A"},
                {"row": 0, "col": 1, "text": "B"},
                {"row": 1, "col": 0, "text": "C"},
                {"row": 1, "col": 1, "text": "D"},
            ],
        }

        score1 = table_teds(table1, table1)
        score2 = table_teds(table1, table1)
        assert score1 == score2

    def test_completely_different_tables(self):
        """Test TEDS near 0 for completely different tables."""
        predicted = {
            "rows": 1,
            "cols": 1,
            "cells": [{"row": 0, "col": 0, "text": "A"}],
        }

        gold = {
            "rows": 3,
            "cols": 3,
            "cells": [
                {"row": i, "col": j, "text": f"{i}{j}"}
                for i in range(3)
                for j in range(3)
            ],
        }

        score = table_teds(predicted, gold)
        # Should be low due to very different structure
        assert 0 <= score <= 0.5
