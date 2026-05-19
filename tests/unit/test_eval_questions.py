"""Tests for eval questions schema validation."""

from pathlib import Path

import pytest

from eval_harness.adapters.schema_validator import validate


class TestEvalQuestions:
    """Test suite for eval questions validation."""

    def test_parsing_json_validates(self, tmp_path):
        """Test that parsing.json validates against schema."""
        # Create a minimal valid parsing eval questions file
        questions_data = {
            "schema_version": "1.0.0",
            "questions": [
                {
                    "question_id": "text_fidelity",
                    "pillar": "parsing",
                    "criterion": "char_level_f1",
                    "description": "Character-level F1 between extracted and gold text",
                    "evaluator": {
                        "kind": "deterministic",
                        "metric": "char_level_f1",
                        "threshold": {"operator": ">=", "value": 0.95},
                    },
                    "required_content": ["parser_output", "gold_parser_elements"],
                    "scope_guardrail": "Normalize whitespace and unicode before comparison",
                    "severity": "major",
                },
                {
                    "question_id": "table_teds",
                    "pillar": "parsing",
                    "criterion": "table_similarity",
                    "description": "Tree Edit Distance Similarity for tables",
                    "evaluator": {
                        "kind": "deterministic",
                        "metric": "table_teds",
                        "threshold": {"operator": ">=", "value": 0.85},
                    },
                    "required_content": ["parser_output", "gold_parser_elements"],
                    "scope_guardrail": "Compare only tables present in both predicted and gold",
                    "severity": "major",
                },
            ],
        }

        schema_path = Path("contracts/eval_questions.schema.json")
        validate(questions_data, schema_path)

    def test_rag_json_validates(self, tmp_path):
        """Test that rag.json validates against schema."""
        questions_data = {
            "schema_version": "1.0.0",
            "questions": [
                {
                    "question_id": "recall_at_5",
                    "pillar": "retrieval",
                    "criterion": "recall_at_k",
                    "description": "Does at least one top-5 chunk overlap gold span?",
                    "evaluator": {
                        "kind": "deterministic",
                        "metric": "recall_at_k",
                        "params": {"k": 5},
                        "threshold": {"operator": ">=", "value": 0.7},
                    },
                    "required_content": ["retrieved_chunks", "gold_spans"],
                    "scope_guardrail": "Do not penalize ranking if at least one relevant in top-k",
                    "severity": "blocker",
                },
                {
                    "question_id": "precision_at_5",
                    "pillar": "retrieval",
                    "criterion": "precision_at_k",
                    "description": "What fraction of top-5 chunks overlap gold spans?",
                    "evaluator": {
                        "kind": "deterministic",
                        "metric": "precision_at_k",
                        "params": {"k": 5},
                    },
                    "required_content": ["retrieved_chunks", "gold_spans"],
                    "scope_guardrail": "Precision < 1.0 is acceptable if recall is high",
                    "severity": "minor",
                },
            ],
        }

        schema_path = Path("contracts/eval_questions.schema.json")
        validate(questions_data, schema_path)

    def test_questions_have_required_fields(self, tmp_path):
        """Test that questions have all required fields."""
        # Missing required field: pillar
        invalid_data = {
            "schema_version": "1.0.0",
            "questions": [
                {
                    "question_id": "invalid_question",
                    "criterion": "some_metric",
                    "description": "Missing pillar field",
                    "evaluator": {
                        "kind": "deterministic",
                        "metric": "char_level_f1",
                    },
                    "required_content": ["parser_output"],
                    "scope_guardrail": "Test",
                },
            ],
        }

        schema_path = Path("contracts/eval_questions.schema.json")

        with pytest.raises(Exception):  # Should fail validation
            validate(invalid_data, schema_path)
