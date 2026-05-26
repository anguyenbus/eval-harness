"""
Test evaluation span formatting in Phoenix adapter.
"""

import pytest


def test_format_evaluation_output_basic():
    """Test basic formatting without reasoning."""
    from eval_harness.observability.phoenix_adapter import PhoenixAdapter

    adapter = PhoenixAdapter(enabled=False)

    metrics = {
        "faithfulness": 0.95,
        "context_precision": 0.30,
        "answer_relevancy": 0.88,
    }

    output = adapter._format_evaluation_output(metrics, "PASS", None)

    assert "=== EVALUATION RESULTS ===" in output
    assert "Faithfulness: 0.95 ✓" in output
    assert "Context Precision: 0.30 ✗" in output
    assert "Answer Relevancy: 0.88 ✓" in output
    assert "VERDICT: PASS" in output


def test_format_evaluation_output_with_reasoning():
    """Test formatting with reasoning included."""
    from eval_harness.observability.phoenix_adapter import PhoenixAdapter

    adapter = PhoenixAdapter(enabled=False)

    metrics = {"faithfulness": 0.95, "context_precision": 0.30}
    reasoning = {
        "faithfulness": {
            "reason": "The answer accurately reflects the retrieved context.",
            "verdicts": [
                {"verdict": "yes", "reason": "Claim 1 is supported"},
                {"verdict": "yes", "reason": "Claim 2 is supported"},
            ],
        },
        "context_precision": {
            "reason": "Only 2 out of 5 chunks contain relevant information.",
            "verdicts": [
                {"verdict": "yes", "reason": "Relevant"},
                {"verdict": "no", "reason": "Irrelevant"},
            ],
        },
    }

    output = adapter._format_evaluation_output(metrics, "NEEDS_REVIEW", reasoning)

    assert "FAITHFULNESS: The answer accurately reflects" in output
    assert "(2/2 items passed)" in output
    assert "CONTEXT PRECISION: Only 2 out of 5" in output
    assert "VERDICT: NEEDS_REVIEW" in output


def test_format_evaluation_output_truncates_long_reasoning():
    """Test that long reasoning is truncated."""
    from eval_harness.observability.phoenix_adapter import PhoenixAdapter

    adapter = PhoenixAdapter(enabled=False)

    metrics = {"faithfulness": 0.95}
    reasoning = {
        "faithfulness": {
            "reason": "x" * 300,  # Long reason
        }
    }

    output = adapter._format_evaluation_output(metrics, "PASS", reasoning)

    assert "..." in output  # Should be truncated
    # Find the FAITHFULNESS line and check it's truncated
    lines = output.split("\n")
    faithfulness_line = next(l for l in lines if "FAITHFULNESS:" in l)
    assert len(faithfulness_line) < 300  # Should be shorter than original


def test_add_evaluation_events():
    """Test span events are added correctly."""
    from unittest.mock import MagicMock

    from eval_harness.observability.phoenix_adapter import PhoenixAdapter

    adapter = PhoenixAdapter(enabled=False)

    mock_span = MagicMock()

    reasoning = {
        "faithfulness": {
            "verdicts": [
                {"verdict": "yes", "reason": "Supported by context"},
                {"verdict": "no", "reason": "Not found in context"},
            ],
            "claims": [
                {"claim_text": "Contract can be terminated", "is_verified": True},
                {"claim_text": "Notice period is 30 days", "is_verified": False},
            ],
        }
    }

    adapter._add_evaluation_events(mock_span, reasoning)

    # Should add 2 verdict events + 2 claim events = 4 events
    assert mock_span.add_event.call_count == 4

    # Check first verdict event
    first_call = mock_span.add_event.call_args_list[0]
    assert first_call[1]["name"] == "faithfulness_verdict_1"
    assert first_call[1]["attributes"]["verdict"] == "yes"
    assert "Supported by context" in first_call[1]["attributes"]["reason"]

    # Check first claim event
    third_call = mock_span.add_event.call_args_list[2]
    assert third_call[1]["name"] == "faithfulness_claim_1"
    assert "Contract can be terminated" in third_call[1]["attributes"]["claim_text"]
    assert third_call[1]["attributes"]["verified"] == "True"
