"""
Tests for replay task wrappers.
"""

from unittest.mock import MagicMock

import pytest

from eval_harness.replay.tasks import BaselineTask, CandidateTask


class TestCandidateTask:
    """Tests for CandidateTask wrapper."""

    def test_candidate_task_initialization(self) -> None:
        """Test CandidateTask initialization."""
        adapter = MagicMock()
        task = CandidateTask(adapter, "test_candidate")

        assert task._name == "test_candidate"

    def test_candidate_task_call_success(self) -> None:
        """Test CandidateTask with successful adapter call."""
        adapter = MagicMock(return_value={"result": "success"})
        task = CandidateTask(adapter, "test_candidate")

        result = task({"input": "data"})

        assert result["success"] is True
        assert result["output"] == {"result": "success"}
        assert result["error"] is None
        assert result["task_name"] == "test_candidate"

    def test_candidate_task_call_failure(self) -> None:
        """Test CandidateTask with adapter exception."""
        adapter = MagicMock(side_effect=ValueError("Test error"))
        task = CandidateTask(adapter, "test_candidate")

        result = task({"input": "data"})

        assert result["success"] is False
        assert result["output"] is None
        assert result["error"] == "Test error"
        assert result["task_name"] == "test_candidate"

    def test_candidate_task_passes_input(self) -> None:
        """Test that input data is passed to adapter."""
        adapter = MagicMock()
        task = CandidateTask(adapter, "test_candidate")

        input_data = {"question": "test question"}
        task(input_data)

        adapter.assert_called_once_with(input_data)


class TestBaselineTask:
    """Tests for BaselineTask wrapper."""

    def test_baseline_task_initialization(self) -> None:
        """Test BaselineTask initialization."""
        adapter = MagicMock()
        task = BaselineTask(adapter, "test_baseline")

        assert task._name == "test_baseline"

    def test_baseline_task_call_success(self) -> None:
        """Test BaselineTask with successful adapter call."""
        adapter = MagicMock(return_value={"result": "success"})
        task = BaselineTask(adapter, "test_baseline")

        result = task({"input": "data"})

        assert result["success"] is True
        assert result["output"] == {"result": "success"}
        assert result["error"] is None
        assert result["task_name"] == "test_baseline"

    def test_baseline_task_call_failure(self) -> None:
        """Test BaselineTask with adapter exception."""
        adapter = MagicMock(side_effect=RuntimeError("Test error"))
        task = BaselineTask(adapter, "test_baseline")

        result = task({"input": "data"})

        assert result["success"] is False
        assert result["output"] is None
        assert result["error"] == "Test error"
        assert result["task_name"] == "test_baseline"

    def test_baseline_task_passes_input(self) -> None:
        """Test that input data is passed to adapter."""
        adapter = MagicMock()
        task = BaselineTask(adapter, "test_baseline")

        input_data = {"question": "test question"}
        task(input_data)

        adapter.assert_called_once_with(input_data)
