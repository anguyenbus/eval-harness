"""
Tests for Phoenix Native DeepEval Integration.

Tests dataset creation, evaluator functions, and parameter mapping.
"""

from unittest.mock import MagicMock, patch

import pytest

from eval_harness.experiments.deepeval_evaluators import (
    create_answer_relevancy_evaluator,
    create_context_precision_evaluator,
    create_context_recall_evaluator,
    create_faithfulness_evaluator,
)


# Undecorated version of create_phoenix_dataset for testing
def _create_phoenix_dataset_no_beartype(
    client,
    corpus_dir,
    slice_name: str = "nano",
    dataset_name: str | None = None,
):
    """Undecorated version for testing - bypasses beartype."""
    from eval_harness.datasets.legal_rag_bench import load_legal_rag_bench

    DEFAULT_DATASET_NAME = "legal-rag-bench"

    dataset = load_legal_rag_bench(cache_dir=corpus_dir, slice=slice_name)

    inputs = []
    outputs = []
    metadata_list = []

    for query_id, query_text, relevant_passage_id, gold_answer in dataset:
        inputs.append({"input": query_text})
        outputs.append({"expected": gold_answer})
        metadata_list.append({
            "query_id": query_id,
            "relevant_passage_id": relevant_passage_id,
        })

    name = dataset_name or f"{DEFAULT_DATASET_NAME}-{slice_name}"

    try:
        existing = client.datasets.get_dataset(dataset=name)
        if inputs:
            client.datasets.add_examples_to_dataset(
                dataset=name,
                inputs=inputs,
                outputs=outputs,
                metadata=metadata_list,
                input_keys=["input"],
                output_keys=["expected"],
            )
        return existing
    except Exception:
        return client.datasets.create_dataset(
            name=name,
            inputs=inputs,
            outputs=outputs,
            metadata=metadata_list,
            input_keys=["input"],
            output_keys=["expected"],
            dataset_description=f"Legal RAG Bench {slice_name} slice",
        )


@pytest.mark.unit
class TestEvaluatorParameterHandling:
    """Test evaluators handle both string and dict parameter formats."""

    @pytest.fixture
    def mock_task_output(self):
        """Mock task output dict."""
        return {
            "answer": "This is a test answer.",
            "retrieval_context": ["Context 1", "Context 2"],
        }

    def test_faithfulness_evaluator_with_dict_params(self, mock_task_output):
        """Faithfulness evaluator extracts values from dict parameters."""
        # Mock DeepEval imports to avoid actual LLM calls
        with patch("deepeval.metrics.FaithfulnessMetric") as mock_metric_class, \
             patch("deepeval.test_case.LLMTestCase") as mock_test_case_class:

            mock_metric = MagicMock()
            mock_metric.score = 0.85
            mock_metric_class.return_value = mock_metric

            mock_test_case = MagicMock()
            mock_test_case_class.return_value = mock_test_case

            evaluator = create_faithfulness_evaluator(judge_model="gpt-4o-mini")

            # Call with dict parameters (Phoenix format)
            evaluator(
                output=mock_task_output,
                input={"input": "Test question?"},
                expected={"expected": "Test expected answer."},
            )

            # Verify the test case was created with extracted strings
            mock_test_case_class.assert_called_once()
            call_kwargs = mock_test_case_class.call_args[1]
            assert call_kwargs["input"] == "Test question?"
            assert call_kwargs["actual_output"] == "This is a test answer."
            assert call_kwargs["expected_output"] == "Test expected answer."
            assert call_kwargs["retrieval_context"] == ["Context 1", "Context 2"]

    def test_faithfulness_evaluator_with_string_params(self, mock_task_output):
        """Faithfulness evaluator works with string parameters."""
        with patch("deepeval.metrics.FaithfulnessMetric") as mock_metric_class, \
             patch("deepeval.test_case.LLMTestCase") as mock_test_case_class:

            mock_metric = MagicMock()
            mock_metric.score = 0.85
            mock_metric_class.return_value = mock_metric

            mock_test_case = MagicMock()
            mock_test_case_class.return_value = mock_test_case

            evaluator = create_faithfulness_evaluator(judge_model="gpt-4o-mini")

            # Call with string parameters (ideal format)
            evaluator(
                output=mock_task_output,
                input="Test question?",
                expected="Test expected answer.",
            )

            mock_test_case_class.assert_called_once()
            call_kwargs = mock_test_case_class.call_args[1]
            assert call_kwargs["input"] == "Test question?"
            assert call_kwargs["actual_output"] == "This is a test answer."

    def test_context_precision_evaluator_with_output_key_dict(self, mock_task_output):
        """Context precision evaluator handles {'output': ...} dict format."""
        with patch("deepeval.metrics.ContextualPrecisionMetric") as mock_metric_class, \
             patch("deepeval.test_case.LLMTestCase") as mock_test_case_class:

            mock_metric = MagicMock()
            mock_metric.score = 0.75
            mock_metric_class.return_value = mock_metric

            mock_test_case = MagicMock()
            mock_test_case_class.return_value = mock_test_case

            evaluator = create_context_precision_evaluator(judge_model="gpt-4o-mini")

            # Call with 'output' key (old Phoenix format)
            evaluator(
                output=mock_task_output,
                input={"input": "Question?"},
                expected={"output": "Expected answer."},
            )

            mock_test_case_class.assert_called_once()
            call_kwargs = mock_test_case_class.call_args[1]
            assert call_kwargs["input"] == "Question?"
            assert call_kwargs["expected_output"] == "Expected answer."

    def test_context_recall_evaluator_with_expected_key_dict(self, mock_task_output):
        """Context recall evaluator handles {'expected': ...} dict format."""
        with patch("deepeval.metrics.ContextualRecallMetric") as mock_metric_class, \
             patch("deepeval.test_case.LLMTestCase") as mock_test_case_class:

            mock_metric = MagicMock()
            mock_metric.score = 0.80
            mock_metric_class.return_value = mock_metric

            mock_test_case = MagicMock()
            mock_test_case_class.return_value = mock_test_case

            evaluator = create_context_recall_evaluator(judge_model="gpt-4o-mini")

            # Call with 'expected' key
            evaluator(
                output=mock_task_output,
                input={"input": "Question?"},
                expected={"expected": "Expected answer."},
            )

            mock_test_case_class.assert_called_once()
            call_kwargs = mock_test_case_class.call_args[1]
            assert call_kwargs["input"] == "Question?"
            assert call_kwargs["expected_output"] == "Expected answer."

    def test_answer_relevancy_evaluator_with_dict_input(self, mock_task_output):
        """Answer relevancy evaluator handles dict input parameter."""
        with patch("deepeval.metrics.AnswerRelevancyMetric") as mock_metric_class, \
             patch("deepeval.test_case.LLMTestCase") as mock_test_case_class, \
             patch("eval_harness.adapters.embeddings.get_embedder") as mock_embedder:

            mock_embedder.return_value = MagicMock()

            mock_metric = MagicMock()
            mock_metric.score = 0.90
            mock_metric_class.return_value = mock_metric

            mock_test_case = MagicMock()
            mock_test_case_class.return_value = mock_test_case

            evaluator = create_answer_relevancy_evaluator(judge_model="gpt-4o-mini")

            # Call with dict input
            evaluator(
                output=mock_task_output,
                input={"input": "Test question?"},
            )

            mock_test_case_class.assert_called_once()
            call_kwargs = mock_test_case_class.call_args[1]
            assert call_kwargs["input"] == "Test question?"
            assert call_kwargs["actual_output"] == "This is a test answer."

    def test_evaluator_handles_none_output(self):
        """Evaluator handles None output gracefully."""
        with patch("deepeval.metrics.FaithfulnessMetric") as mock_metric_class, \
             patch("deepeval.test_case.LLMTestCase") as mock_test_case_class:

            mock_metric = MagicMock()
            mock_metric.score = 0.0
            mock_metric_class.return_value = mock_metric

            mock_test_case = MagicMock()
            mock_test_case_class.return_value = mock_test_case

            evaluator = create_faithfulness_evaluator(judge_model="gpt-4o-mini")

            # Call with None output
            evaluator(
                output=None,
                input="Question?",
                expected="Answer.",
            )

            mock_test_case_class.assert_called_once()
            call_kwargs = mock_test_case_class.call_args[1]
            assert call_kwargs["actual_output"] == ""
            assert call_kwargs["retrieval_context"] == []

    def test_evaluator_handles_missing_keys_in_dict(self, mock_task_output):
        """Evaluator handles dicts with missing keys gracefully."""
        with patch("deepeval.metrics.FaithfulnessMetric") as mock_metric_class, \
             patch("deepeval.test_case.LLMTestCase") as mock_test_case_class:

            mock_metric = MagicMock()
            mock_metric.score = 0.0
            mock_metric_class.return_value = mock_metric

            mock_test_case = MagicMock()
            mock_test_case_class.return_value = mock_test_case

            evaluator = create_faithfulness_evaluator(judge_model="gpt-4o-mini")

            # Call with dict that has wrong keys
            evaluator(
                output=mock_task_output,
                input={"wrong_key": "Question?"},
                expected={"wrong_key": "Answer."},
            )

            mock_test_case_class.assert_called_once()
            call_kwargs = mock_test_case_class.call_args[1]
            # Should fall back to empty strings when keys are missing
            assert call_kwargs["input"] == ""
            assert call_kwargs["expected_output"] == ""

    def test_context_recall_evaluator_no_embedder_param(self, mock_task_output):
        """Context recall evaluator creates metric without embedder param."""
        with patch("deepeval.metrics.ContextualRecallMetric") as mock_metric_class, \
             patch("deepeval.test_case.LLMTestCase") as mock_test_case_class:

            mock_metric = MagicMock()
            mock_metric.score = 0.80
            mock_metric_class.return_value = mock_metric

            mock_test_case = MagicMock()
            mock_test_case_class.return_value = mock_test_case

            evaluator = create_context_recall_evaluator(judge_model="gpt-4o-mini")

            # Call evaluator to trigger metric creation
            evaluator(
                output=mock_task_output,
                input={"input": "Question?"},
                expected={"expected": "Expected answer."},
            )

            # Verify ContextualRecallMetric was created without embedder
            mock_metric_class.assert_called_once()
            call_kwargs = mock_metric_class.call_args[1]
            assert "embedder" not in call_kwargs
            assert call_kwargs["model"] == "gpt-4o-mini"
            assert call_kwargs["include_reason"] is True


@pytest.mark.unit
class TestDatasetCreation:
    """Test Phoenix dataset creation format."""

    def test_dataset_format_uses_correct_keys(self):
        """Dataset uses 'input' and 'expected' keys."""
        from pathlib import Path

        patch_path = "eval_harness.datasets.legal_rag_bench.load_legal_rag_bench"
        with patch(patch_path) as mock_load:
            mock_client = MagicMock()
            mock_dataset = MagicMock()
            mock_client.datasets.create_dataset.return_value = mock_dataset
            mock_client.datasets.get_dataset.side_effect = Exception("Not found")

            # Mock dataset: (query_id, query_text, relevant_passage_id, gold_answer)
            mock_load.return_value = [
                (1, "Test question?", "passage-1", "Test answer."),
                (2, "Another question?", "passage-2", "Another answer."),
            ]

            _create_phoenix_dataset_no_beartype(
                client=mock_client,
                corpus_dir=Path("/fake/path"),
                slice_name="nano",
            )

            # Verify create_dataset was called with correct format
            mock_client.datasets.create_dataset.assert_called_once()
            call_kwargs = mock_client.datasets.create_dataset.call_args[1]

            # Check inputs format
            inputs = call_kwargs["inputs"]
            assert len(inputs) == 2
            assert inputs[0] == {"input": "Test question?"}
            assert inputs[1] == {"input": "Another question?"}

            # Check outputs format
            outputs = call_kwargs["outputs"]
            assert len(outputs) == 2
            assert outputs[0] == {"expected": "Test answer."}
            assert outputs[1] == {"expected": "Another answer."}

            # Check metadata format
            metadata = call_kwargs["metadata"]
            assert len(metadata) == 2
            assert metadata[0]["query_id"] == 1
            assert metadata[0]["relevant_passage_id"] == "passage-1"

            # Check input_keys and output_keys are specified
            assert call_kwargs["input_keys"] == ["input"]
            assert call_kwargs["output_keys"] == ["expected"]

    def test_dataset_adds_to_existing(self):
        """Dataset adds examples to existing dataset."""
        from pathlib import Path

        patch_path = "eval_harness.datasets.legal_rag_bench.load_legal_rag_bench"
        with patch(patch_path) as mock_load:
            mock_client = MagicMock()
            mock_existing = MagicMock()
            mock_client.datasets.get_dataset.return_value = mock_existing

            mock_load.return_value = [
                (1, "Question?", "passage-1", "Answer."),
            ]

            _create_phoenix_dataset_no_beartype(
                client=mock_client,
                corpus_dir=Path("/fake/path"),
                slice_name="nano",
            )

            # Should add to existing, not create new
            mock_client.datasets.add_examples_to_dataset.assert_called_once()
            mock_client.datasets.create_dataset.assert_not_called()

            # Verify add_examples_to_dataset was called with correct keys
            call_kwargs = mock_client.datasets.add_examples_to_dataset.call_args[1]
            assert call_kwargs["input_keys"] == ["input"]
            assert call_kwargs["output_keys"] == ["expected"]


@pytest.mark.unit
class TestTaskOutputFormat:
    """Test task function returns correct format for evaluators."""

    def test_task_returns_dict_with_answer_and_context(self):
        """Task returns dict with 'answer' and 'retrieval_context' keys."""
        from eval_harness.experiments.runner import create_rag_task

        mock_adapter = MagicMock()
        mock_adapter.query.return_value = {
            "answer": {"text": "Generated answer."},
            "retrieved_chunks": [
                {"text": "Context 1"},
                {"text": "Context 2"},
            ],
        }

        from pathlib import Path
        task = create_rag_task(mock_adapter, Path("/fake/corpus"))

        # Test with dict format (Phoenix default)
        result = task({"input": "Test question?"})
        assert isinstance(result, dict)
        assert "answer" in result
        assert "retrieval_context" in result
        assert result["answer"] == "Generated answer."
        assert result["retrieval_context"] == ["Context 1", "Context 2"]

        # Verify adapter was called with string, not dict
        mock_adapter.query.assert_called_once()
        question_arg = mock_adapter.query.call_args[0][0]
        assert isinstance(question_arg, str)
        assert question_arg == "Test question?"

    def test_task_handles_nested_dict_input(self):
        """Task handles nested dict format {'input': {'input': 'question'}}."""
        from eval_harness.experiments.runner import create_rag_task

        mock_adapter = MagicMock()
        mock_adapter.query.return_value = {
            "answer": {"text": "Answer."},
            "retrieved_chunks": [{"text": "Context"}],
        }

        from pathlib import Path
        task = create_rag_task(mock_adapter, Path("/fake/corpus"))

        # Test with nested dict format
        result = task({"input": {"input": "Nested question?"}})

        assert result["answer"] == "Answer."
        mock_adapter.query.assert_called_once()
        question_arg = mock_adapter.query.call_args[0][0]
        assert question_arg == "Nested question?"

    def test_task_handles_string_input(self):
        """Task handles direct string input (edge case)."""
        from eval_harness.experiments.runner import create_rag_task

        mock_adapter = MagicMock()
        mock_adapter.query.return_value = {
            "answer": {"text": "Answer."},
            "retrieved_chunks": [{"text": "Context"}],
        }

        from pathlib import Path
        task = create_rag_task(mock_adapter, Path("/fake/corpus"))

        # Test with string format
        result = task("Direct question?")

        assert result["answer"] == "Answer."
        mock_adapter.query.assert_called_once()
        question_arg = mock_adapter.query.call_args[0][0]
        assert question_arg == "Direct question?"

    def test_task_handles_empty_dict_input(self):
        """Task handles empty dict gracefully."""
        from eval_harness.experiments.runner import create_rag_task

        mock_adapter = MagicMock()
        mock_adapter.query.return_value = {
            "answer": {"text": "Answer."},
            "retrieved_chunks": [],
        }

        from pathlib import Path
        task = create_rag_task(mock_adapter, Path("/fake/corpus"))

        # Test with empty dict
        result = task({})

        assert result["answer"] == "Answer."
        mock_adapter.query.assert_called_once()
        question_arg = mock_adapter.query.call_args[0][0]
        assert question_arg == ""

    def test_task_output_is_evaluator_compatible(self):
        """Task output format is compatible with evaluator expectations."""
        from eval_harness.experiments.deepeval_evaluators import (
            create_faithfulness_evaluator,
        )
        from eval_harness.experiments.runner import create_rag_task

        mock_adapter = MagicMock()
        mock_adapter.query.return_value = {
            "answer": {"text": "Answer text."},
            "retrieved_chunks": [{"text": "Context"}],
        }

        from pathlib import Path
        task = create_rag_task(mock_adapter, Path("/fake/corpus"))
        task_output = task({"input": "Question?"})

        # Create evaluator and verify it can handle the task output
        with patch("deepeval.metrics.FaithfulnessMetric") as mock_metric_class, \
             patch("deepeval.test_case.LLMTestCase") as mock_test_case_class:

            mock_metric = MagicMock()
            mock_metric.score = 1.0
            mock_metric_class.return_value = mock_metric

            mock_test_case = MagicMock()
            mock_test_case_class.return_value = mock_test_case

            evaluator = create_faithfulness_evaluator(judge_model="gpt-4o-mini")

            # Should not raise an error
            evaluator(
                output=task_output,
                input="Question?",
                expected="Expected answer.",
            )

            # Verify the evaluator correctly extracted from task output
            mock_test_case_class.assert_called_once()
            call_kwargs = mock_test_case_class.call_args[1]
            assert call_kwargs["actual_output"] == "Answer text."
            assert call_kwargs["retrieval_context"] == ["Context"]
