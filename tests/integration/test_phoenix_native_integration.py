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
            mock_metric.success = True
            mock_metric.reason = "The answer is fully supported by context."
            mock_metric.threshold = 0.5
            mock_metric.evaluation_model = "gpt-4o-mini"
            mock_metric_class.return_value = mock_metric

            mock_test_case = MagicMock()
            mock_test_case_class.return_value = mock_test_case

            evaluator = create_faithfulness_evaluator(judge_model="gpt-4o-mini")

            # Call with dict parameters (Phoenix format)
            result = evaluator(
                output=mock_task_output,
                input={"input": "Test question?"},
                expected={"expected": "Test expected answer."},
            )

            # Verify the test case was created with extracted strings
            mock_test_case_class.assert_called_once()
            call_kwargs = mock_test_case_class.call_args[1]
            assert call_kwargs["input"] == "Test question?"
            assert call_kwargs["actual_output"] == "This is a test answer."
            assert call_kwargs["retrieval_context"] == ["Context 1", "Context 2"]

            # Verify result contains explanation and metadata
            assert result["score"] == 0.85
            assert result["label"] == "faithful"
            assert result["explanation"] == "The answer is fully supported by context."
            # Check metadata fields
            metadata = result["metadata"]
            assert metadata["threshold"] == 0.5
            assert metadata["success"] is True
            assert metadata["model"] == "gpt-4o-mini"
            # Successful evaluation should not include verdicts (failed-only pattern)
            assert "verdicts" not in metadata

    def test_faithfulness_evaluator_with_string_params(self, mock_task_output):
        """Faithfulness evaluator works with string parameters."""
        with patch("deepeval.metrics.FaithfulnessMetric") as mock_metric_class, \
             patch("deepeval.test_case.LLMTestCase") as mock_test_case_class:

            mock_metric = MagicMock()
            mock_metric.score = 0.85
            mock_metric.success = True
            mock_metric.reason = "Good answer."
            mock_metric.threshold = 0.5
            mock_metric_class.return_value = mock_metric

            mock_test_case = MagicMock()
            mock_test_case_class.return_value = mock_test_case

            evaluator = create_faithfulness_evaluator(judge_model="gpt-4o-mini")

            # Call with string parameters (ideal format)
            result = evaluator(
                output=mock_task_output,
                input="Test question?",
                expected="Test expected answer.",
            )

            mock_test_case_class.assert_called_once()
            call_kwargs = mock_test_case_class.call_args[1]
            assert call_kwargs["input"] == "Test question?"
            assert call_kwargs["actual_output"] == "This is a test answer."
            assert result["score"] == 0.85
            assert "explanation" in result

    def test_context_precision_evaluator_with_output_key_dict(self, mock_task_output):
        """Context precision evaluator handles {'output': ...} dict format."""
        with patch("deepeval.metrics.ContextualPrecisionMetric") as mock_metric_class, \
             patch("deepeval.test_case.LLMTestCase") as mock_test_case_class:

            mock_metric = MagicMock()
            mock_metric.score = 0.75
            mock_metric.success = True
            mock_metric.reason = "Good precision."
            mock_metric.threshold = 0.5
            mock_metric_class.return_value = mock_metric

            mock_test_case = MagicMock()
            mock_test_case_class.return_value = mock_test_case

            evaluator = create_context_precision_evaluator(judge_model="gpt-4o-mini")

            # Call with 'output' key (old Phoenix format)
            result = evaluator(
                output=mock_task_output,
                input={"input": "Question?"},
                expected={"output": "Expected answer."},
            )

            mock_test_case_class.assert_called_once()
            call_kwargs = mock_test_case_class.call_args[1]
            assert call_kwargs["input"] == "Question?"
            assert call_kwargs["expected_output"] == "Expected answer."
            assert result["score"] == 0.75
            assert "explanation" in result

    def test_context_recall_evaluator_with_expected_key_dict(self, mock_task_output):
        """Context recall evaluator handles {'expected': ...} dict format."""
        with patch("deepeval.metrics.ContextualRecallMetric") as mock_metric_class, \
             patch("deepeval.test_case.LLMTestCase") as mock_test_case_class:

            mock_metric = MagicMock()
            mock_metric.score = 0.80
            mock_metric.success = True
            mock_metric.reason = "Good recall."
            mock_metric.threshold = 0.5
            mock_metric_class.return_value = mock_metric

            mock_test_case = MagicMock()
            mock_test_case_class.return_value = mock_test_case

            evaluator = create_context_recall_evaluator(judge_model="gpt-4o-mini")

            # Call with 'expected' key
            result = evaluator(
                output=mock_task_output,
                input={"input": "Question?"},
                expected={"expected": "Expected answer."},
            )

            mock_test_case_class.assert_called_once()
            call_kwargs = mock_test_case_class.call_args[1]
            assert call_kwargs["input"] == "Question?"
            assert call_kwargs["expected_output"] == "Expected answer."
            assert result["score"] == 0.80
            assert "explanation" in result

    def test_answer_relevancy_evaluator_with_dict_input(self, mock_task_output):
        """Answer relevancy evaluator handles dict input parameter."""
        with patch("deepeval.metrics.AnswerRelevancyMetric") as mock_metric_class, \
             patch("deepeval.test_case.LLMTestCase") as mock_test_case_class, \
             patch("eval_harness.adapters.embeddings.get_embedder") as mock_embedder:

            mock_embedder.return_value = MagicMock()

            mock_metric = MagicMock()
            mock_metric.score = 0.90
            mock_metric.success = True
            mock_metric.reason = "Relevant answer."
            mock_metric.threshold = 0.5
            mock_metric_class.return_value = mock_metric

            mock_test_case = MagicMock()
            mock_test_case_class.return_value = mock_test_case

            evaluator = create_answer_relevancy_evaluator(judge_model="gpt-4o-mini")

            # Call with dict input
            result = evaluator(
                output=mock_task_output,
                input={"input": "Test question?"},
            )

            mock_test_case_class.assert_called_once()
            call_kwargs = mock_test_case_class.call_args[1]
            assert call_kwargs["input"] == "Test question?"
            assert call_kwargs["actual_output"] == "This is a test answer."
            assert result["score"] == 0.90
            assert "explanation" in result

    def test_evaluator_handles_none_output(self):
        """Evaluator handles None output gracefully."""
        with patch("deepeval.metrics.FaithfulnessMetric") as mock_metric_class, \
             patch("deepeval.test_case.LLMTestCase") as mock_test_case_class:

            mock_metric = MagicMock()
            mock_metric.score = 0.0
            mock_metric.success = False
            mock_metric.reason = ""
            mock_metric.threshold = 0.5
            mock_metric_class.return_value = mock_metric

            mock_test_case = MagicMock()
            mock_test_case_class.return_value = mock_test_case

            evaluator = create_faithfulness_evaluator(judge_model="gpt-4o-mini")

            # Call with None output - returns skipped result
            result = evaluator(
                output=None,
                input="Question?",
                expected="Answer.",
            )

            # Should return skipped result since retrieval_context is empty
            assert result["score"] == 0.0
            assert result["label"] == "skipped"
            assert "Missing retrieval_context" in result["explanation"]

    def test_evaluator_handles_missing_keys_in_dict(self, mock_task_output):
        """Evaluator handles dicts with missing keys gracefully."""
        with patch("deepeval.metrics.FaithfulnessMetric") as mock_metric_class, \
             patch("deepeval.test_case.LLMTestCase") as mock_test_case_class:

            mock_metric = MagicMock()
            mock_metric.score = 0.85
            mock_metric.success = True
            mock_metric.reason = "Good answer."
            mock_metric.threshold = 0.5
            mock_metric_class.return_value = mock_metric

            mock_test_case = MagicMock()
            mock_test_case_class.return_value = mock_test_case

            evaluator = create_faithfulness_evaluator(judge_model="gpt-4o-mini")

            # Call with dict that has wrong keys but valid output
            result = evaluator(
                output=mock_task_output,
                input={"wrong_key": "Question?"},
                expected={"wrong_key": "Answer."},
            )

            mock_test_case_class.assert_called_once()
            call_kwargs = mock_test_case_class.call_args[1]
            # Should fall back to empty strings when keys are missing
            assert call_kwargs["input"] == ""
            # But result should still have score since retrieval_context is present
            assert result["score"] == 0.85

    def test_context_recall_evaluator_no_embedder_param(self, mock_task_output):
        """Context recall evaluator creates metric without embedder param."""
        with patch("deepeval.metrics.ContextualRecallMetric") as mock_metric_class, \
             patch("deepeval.test_case.LLMTestCase") as mock_test_case_class:

            mock_metric = MagicMock()
            mock_metric.score = 0.80
            mock_metric.success = True
            mock_metric.reason = "Good recall."
            mock_metric.threshold = 0.5
            mock_metric_class.return_value = mock_metric

            mock_test_case = MagicMock()
            mock_test_case_class.return_value = mock_test_case

            evaluator = create_context_recall_evaluator(judge_model="gpt-4o-mini")

            # Call evaluator to trigger metric creation
            result = evaluator(
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
            assert result["score"] == 0.80

    def test_failed_evaluator_includes_verdicts_in_metadata(self, mock_task_output):
        """Failed evaluations include verdicts in metadata for debugging."""
        with patch("deepeval.metrics.FaithfulnessMetric") as mock_metric_class, \
             patch("deepeval.test_case.LLMTestCase") as mock_test_case_class:

            # Create mock verdict objects that return dicts from model_dump()
            mock_verdict_1 = MagicMock()
            mock_verdict_1.verdict = "no"
            mock_verdict_1.reason = "Claim not supported by context"
            mock_verdict_1.model_dump.return_value = {
                "verdict": "no",
                "reason": "Claim not supported by context",
            }

            mock_verdict_2 = MagicMock()
            mock_verdict_2.verdict = "yes"
            mock_verdict_2.model_dump.return_value = {"verdict": "yes"}

            mock_metric = MagicMock()
            mock_metric.score = 0.3  # Failed (below threshold)
            mock_metric.success = False
            mock_metric.reason = "Some claims are not supported."
            mock_metric.threshold = 0.5
            mock_metric.evaluation_model = "gpt-4o-mini"
            mock_metric.verdicts = [mock_verdict_1, mock_verdict_2]
            mock_metric_class.return_value = mock_metric

            mock_test_case = MagicMock()
            mock_test_case_class.return_value = mock_test_case

            evaluator = create_faithfulness_evaluator(judge_model="gpt-4o-mini")

            result = evaluator(
                output=mock_task_output,
                input="Question?",
                expected="Answer.",
            )

            # Failed evaluation should include verdicts
            assert result["score"] == 0.3
            assert result["label"] == "unfaithful"
            metadata = result["metadata"]
            assert "verdicts" in metadata
            assert len(metadata["verdicts"]) == 2
            # Verify verdicts were serialized using model_dump
            assert metadata["verdicts"][0]["verdict"] == "no"

    def test_passed_evaluator_excludes_verdicts_from_metadata(self, mock_task_output):
        """Passed evaluations exclude verdicts from metadata (failed-only pattern)."""
        with patch("deepeval.metrics.FaithfulnessMetric") as mock_metric_class, \
             patch("deepeval.test_case.LLMTestCase") as mock_test_case_class:

            # Even though verdicts exist, they shouldn't be included for passing scores
            mock_verdict = MagicMock()
            mock_verdict.verdict = "yes"
            mock_verdict.model_dump.return_value = {"verdict": "yes"}

            mock_metric = MagicMock()
            mock_metric.score = 0.9  # Passed (above threshold)
            mock_metric.success = True
            mock_metric.reason = "All claims supported."
            mock_metric.threshold = 0.5
            mock_metric.evaluation_model = "gpt-4o-mini"
            mock_metric.verdicts = [mock_verdict]
            mock_metric_class.return_value = mock_metric

            mock_test_case = MagicMock()
            mock_test_case_class.return_value = mock_test_case

            evaluator = create_faithfulness_evaluator(judge_model="gpt-4o-mini")

            result = evaluator(
                output=mock_task_output,
                input="Question?",
                expected="Answer.",
            )

            # Passing evaluation should NOT include verdicts
            assert result["score"] == 0.9
            assert result["label"] == "faithful"
            metadata = result["metadata"]
            assert "verdicts" not in metadata


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
            mock_metric.success = True
            mock_metric.reason = "Excellent answer."
            mock_metric.threshold = 0.5
            mock_metric_class.return_value = mock_metric

            mock_test_case = MagicMock()
            mock_test_case_class.return_value = mock_test_case

            evaluator = create_faithfulness_evaluator(judge_model="gpt-4o-mini")

            # Should not raise an error
            result = evaluator(
                output=task_output,
                input="Question?",
                expected="Expected answer.",
            )

            # Verify the evaluator correctly extracted from task output
            mock_test_case_class.assert_called_once()
            call_kwargs = mock_test_case_class.call_args[1]
            assert call_kwargs["actual_output"] == "Answer text."
            assert call_kwargs["retrieval_context"] == ["Context"]
            # Verify result has explanation
            assert result["score"] == 1.0
            assert result["explanation"] == "Excellent answer."
