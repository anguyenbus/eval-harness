"""
Tests for Phoenix experiment runner.
"""

from pathlib import Path

import pytest


def test_create_phoenix_client_raises_without_phoenix():
    """Test that create_phoenix_client raises ImportError without Phoenix."""
    # Mock the Client to None to simulate missing Phoenix
    import eval_harness.experiments.runner as runner_module
    from eval_harness.experiments.runner import create_phoenix_client
    original_client = runner_module.Client
    runner_module.Client = None

    try:
        with pytest.raises(ImportError, match="Phoenix client not available"):
            create_phoenix_client()
    finally:
        runner_module.Client = original_client


def test_create_rag_task_returns_callable():
    """Test that create_rag_task returns a callable function."""
    from eval_harness.experiments.runner import create_rag_task

    # Mock RAG adapter - lambda should accept *args to match any signature
    mock_adapter = type("MockAdapter", (), {
        "query": lambda *args: {"answer": {"text": f"Answer to: {args[0] if args else 'unknown'}"}}
    })()

    task = create_rag_task(mock_adapter, Path("/fake"))

    assert callable(task)

    # Test task execution
    result = task({"input": "Test question"})

    assert "answer" in result
    assert "retrieval_context" in result
