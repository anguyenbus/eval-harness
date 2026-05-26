"""
Tests for DeepEval metrics wrapped as Phoenix evaluators.
"""

import pytest

from eval_harness.experiments.deepeval_evaluators import (
    create_faithfulness_evaluator,
    create_context_precision_evaluator,
    create_context_recall_evaluator,
    create_answer_relevancy_evaluator,
)


def test_create_faithfulness_evaluator_returns_callable():
    """Test that faithfulness evaluator factory returns a callable."""
    evaluator = create_faithfulness_evaluator(judge_model="gpt-4o-mini")
    assert callable(evaluator)
    # Should have name from decorator
    assert hasattr(evaluator, "name") or "faithfulness" in str(evaluator)


def test_create_context_precision_evaluator_returns_callable():
    """Test that context precision evaluator factory returns a callable."""
    evaluator = create_context_precision_evaluator()
    assert callable(evaluator)


def test_create_context_recall_evaluator_returns_callable():
    """Test that context recall evaluator factory returns a callable."""
    evaluator = create_context_recall_evaluator()
    assert callable(evaluator)


def test_create_answer_relevancy_evaluator_returns_callable():
    """Test that answer relevancy evaluator factory returns a callable."""
    evaluator = create_answer_relevancy_evaluator()
    assert callable(evaluator)


@pytest.mark.skipif("os.environ.get('OPENAI_API_KEY') is None")
def test_faithfulness_evaluator_basic_evaluation():
    """Test faithfulness evaluator with basic evaluation (requires OPENAI_API_KEY)."""
    evaluator = create_faithfulness_evaluator()

    result = evaluator(
        output="The contract can be terminated with 30 days notice.",
        retrieval_context=["The contract requires 30 days notice for termination."],
    )

    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0


@pytest.mark.skipif("os.environ.get('OPENAI_API_KEY') is None")
def test_context_precision_evaluator_basic_evaluation():
    """Test context precision evaluator with basic evaluation."""
    evaluator = create_context_precision_evaluator()

    result = evaluator(
        output="The contract can be terminated with 30 days notice.",
        expected="The contract requires 30 days notice for termination.",
        retrieval_context=[
            "The contract requires 30 days notice for termination.",
            "Irrelevant information about pricing.",
        ],
    )

    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0


@pytest.mark.skipif("os.environ.get('OPENAI_API_KEY') is None")
def test_answer_relevancy_evaluator_basic_evaluation():
    """Test answer relevancy evaluator with basic evaluation."""
    evaluator = create_answer_relevancy_evaluator()

    result = evaluator(
        input="What is the termination notice period?",
        output="The contract requires 30 days written notice.",
    )

    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0
