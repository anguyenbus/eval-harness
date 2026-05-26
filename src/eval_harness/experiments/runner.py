"""
Phoenix experiment runner for RAG evaluation.

Orchestrates running RAG experiments with Phoenix's native experiment API,
including dataset management, task execution, and result retrieval.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final

from beartype import beartype
from beartype.typing import Callable

from eval_harness.experiments.datasets import create_phoenix_dataset
from eval_harness.experiments.deepeval_evaluators import (
    create_answer_relevancy_evaluator,
    create_context_precision_evaluator,
    create_context_recall_evaluator,
    create_faithfulness_evaluator,
)

try:
    from phoenix.client import Client
    from phoenix.client.resources.experiments.types import (
        RanExperiment as PhoenixRanExperiment,
    )
except ImportError:
    Client = None  # type: ignore[assignment]
    PhoenixRanExperiment = None  # type: ignore[assignment]

# Type alias for experiment results (can be dict or Phoenix object)
RanExperiment = PhoenixRanExperiment | dict[str, Any] | None

# Constants
DEFAULT_EXPERIMENT_NAME: Final[str] = "rag-evaluation"


@beartype
def create_phoenix_client(
    endpoint: str = "http://localhost:6006",
) -> Client:
    """
    Create a Phoenix client instance.

    Args:
        endpoint: Phoenix server endpoint (UI URL, converted to base_url).

    Returns:
        Phoenix Client instance.

    Raises:
        ImportError: If Phoenix client is not available.

    """
    if Client is None:
        raise ImportError(
            "Phoenix client not available. "
            "Install with: pip install arize-phoenix"
        )

    # Phoenix Client uses base_url parameter
    # The endpoint format is typically http://localhost:6006 for UI
    # The API base_url is http://localhost:6006 (same for Phoenix)
    return Client(base_url=endpoint)


@beartype
def create_rag_task(
    rag_adapter: Any,
    corpus_dir: Path,
) -> Callable:
    """
    Create a RAG task function for Phoenix experiments.

    The task function takes a dataset example and returns the RAG output.

    Args:
        rag_adapter: RAG adapter instance.
        corpus_dir: Path to RAG corpus.

    Returns:
        Task function compatible with Phoenix run_experiment().

    """
    def rag_task(example: dict[str, Any] | str) -> dict[str, Any]:
        """
        Execute RAG query for a single example.

        Args:
            example: Dataset example - can be dict with 'input' key, nested dict,
                    or string directly depending on Phoenix format.

        Returns:
            Dict with 'answer' and 'retrieval_context' for Phoenix evaluators.

        """
        # Extract question from various Phoenix formats
        if isinstance(example, str):
            question = example
        elif isinstance(example, dict):
            # Try different possible structures
            val = example.get("input", "")
            if isinstance(val, dict):
                # Nested dict like {"input": {"input": "question"}}
                question = val.get("input", "")
            elif isinstance(val, str):
                question = val
            else:
                question = str(example.get("input", ""))
        else:
            question = str(example)

        # Query RAG system
        result = rag_adapter.query(question, corpus_dir)

        # Extract output in Phoenix format
        output = result.get("answer", {}).get("text", "")
        retrieved_chunks = result.get("retrieved_chunks", [])
        retrieval_context = [chunk.get("text", "") for chunk in retrieved_chunks]

        # Return tuple: Phoenix maps first element to 'output' by default
        # But we need to return a dict for proper mapping
        return {
            "answer": output,
            "retrieval_context": retrieval_context,
        }

    return rag_task


@beartype
def run_phoenix_experiment(
    rag_adapter: Any,
    corpus_dir: Path,
    endpoint: str = "http://localhost:6006",
    slice_name: str = "nano",
    experiment_name: str | None = None,
    judge_model: str = "gpt-4o-mini",
) -> RanExperiment:
    """
    Run a RAG evaluation experiment using Phoenix's native experiment API.

    Args:
        rag_adapter: RAG adapter instance.
        corpus_dir: Path to RAG corpus.
        endpoint: Phoenix server endpoint.
        slice_name: Dataset slice ("nano" or "full").
        experiment_name: Name for the experiment.
        judge_model: LLM judge model for DeepEval metrics.

    Returns:
        RanExperiment instance with results.

    Raises:
        ImportError: If Phoenix client is not available.

    """
    client = create_phoenix_client(endpoint)

    # Create or get dataset
    dataset = create_phoenix_dataset(
        client=client,
        corpus_dir=corpus_dir,
        slice_name=slice_name,
    )

    # Create task function
    task = create_rag_task(rag_adapter, corpus_dir)

    # Create evaluators
    evaluators = [
        create_faithfulness_evaluator(judge_model=judge_model),
        create_context_precision_evaluator(judge_model=judge_model),
        create_context_recall_evaluator(judge_model=judge_model),
        create_answer_relevancy_evaluator(judge_model=judge_model),
    ]

    # Run experiment
    name = experiment_name or f"{DEFAULT_EXPERIMENT_NAME}-{slice_name}"

    experiment = client.experiments.run_experiment(
        dataset=dataset,
        task=task,
        evaluators=evaluators,
        experiment_name=name,
        experiment_description=f"RAG evaluation on {slice_name} slice",
    )

    return experiment


@beartype
def export_experiment_results(
    experiment: RanExperiment | dict[str, Any],
    output_dir: Path,
    dataset_examples: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Export experiment results to CSV and JSON.

    Maintains backward compatibility with existing eval-rag CSV format.

    Args:
        experiment: RanExperiment instance from Phoenix (dict-like or object).
        output_dir: Output directory for results.
        dataset_examples: Optional original dataset examples for mapping.

    Returns:
        Dictionary with paths to exported files.

    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Handle both dict and object types for experiment
    if isinstance(experiment, dict):
        base_name = experiment.get("experiment_name", "experiment")
    else:
        base_name = getattr(experiment, "experiment_name", "experiment")

    # Build results DataFrame from experiment data
    results_df = _get_experiment_dataframe(experiment, dataset_examples)

    # Export to CSV
    csv_path = output_dir / f"{base_name}_results.csv"
    results_df.to_csv(csv_path, index=False)

    # Export summary to JSON
    json_path = output_dir / f"{base_name}_summary.json"
    summary = _calculate_experiment_summary(experiment, results_df)

    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    return {
        "csv_path": str(csv_path),
        "json_path": str(json_path),
    }


@beartype
def _get_experiment_dataframe(
    experiment: RanExperiment | dict[str, Any],
    dataset_examples: list[dict[str, Any]] | None = None,
) -> Any:
    """
    Extract experiment results as pandas DataFrame.

    Maps Phoenix experiment results to eval-rag CSV format.

    Args:
        experiment: RanExperiment instance (dict-like or object).
        dataset_examples: Optional dataset examples for metadata.

    Returns:
        pandas DataFrame with columns matching eval-rag format.

    """
    import pandas as pd

    rows = []

    # Map dataset examples by their ID/index
    example_map = {}
    if dataset_examples:
        for i, example in enumerate(dataset_examples):
            example_map[i] = example

    # Extract task_runs and evaluation_runs (handle both dict and object)
    if isinstance(experiment, dict):
        task_runs = experiment.get("task_runs", [])
        eval_runs = experiment.get("evaluation_runs", [])
    else:
        task_runs = getattr(experiment, "task_runs", [])
        eval_runs = getattr(experiment, "evaluation_runs", [])

    # Helper to extract values from both dict and object types
    def _get_attr(obj: Any, key: str, default: Any = None) -> Any:
        """Get attribute from both dict-like and object types."""
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    # Create a map of experiment_run_id to evaluations
    eval_map: dict[str, dict[str, Any]] = {}
    for eval_run in eval_runs:
        run_id = _get_attr(eval_run, "experiment_run_id")
        result = _get_attr(eval_run, "result")

        if result and run_id:
            # Result might be a single evaluation or a list
            if isinstance(result, list):
                for r in result:
                    name = _get_attr(r, "name", "")
                    score = _get_attr(r, "score")
                    if name and score is not None:
                        if run_id not in eval_map:
                            eval_map[run_id] = {}
                        eval_map[run_id][name] = score
            else:
                name = _get_attr(result, "name", "")
                score = _get_attr(result, "score")
                if name and score is not None:
                    if run_id not in eval_map:
                        eval_map[run_id] = {}
                    eval_map[run_id][name] = score

    # Build rows
    for i, task_run in enumerate(task_runs):
        run_id = _get_attr(task_run, "id")
        output = _get_attr(task_run, "output", {})
        error = _get_attr(task_run, "error", "")

        # Get input and expected from task_run (Phoenix provides these as dicts)
        input_dict = _get_attr(task_run, "input", {})
        expected_dict = _get_attr(task_run, "expected_output", {})
        metadata = _get_attr(task_run, "metadata", {})

        # Extract values from dicts
        if isinstance(input_dict, dict):
            input_val = input_dict.get("input", "")
        else:
            input_val = str(input_dict)

        if isinstance(expected_dict, dict):
            expected_val = expected_dict.get("expected", "")
        else:
            expected_val = str(expected_dict)

        # Extract generated answer from task output (new format:
        # {"answer": ..., "retrieval_context": [...]})
        if isinstance(output, dict):
            generated_answer = output.get("answer", "")
        else:
            generated_answer = str(output)

        row = {
            "query_id": _get_attr(metadata, "query_id", i),
            "question": input_val,
            "gold_answer": expected_val,
            "generated_answer": generated_answer,
            "relevant_passage_retrieved": "",
            "faithfulness_score": eval_map.get(run_id, {}).get("faithfulness", 0.0),
            "context_precision_score": eval_map.get(run_id, {}).get(
                "context_precision", 0.0
            ),
            "context_recall_score": eval_map.get(run_id, {}).get(
                "context_recall", 0.0
            ),
            "answer_relevancy_score": eval_map.get(run_id, {}).get(
                "answer_relevancy", 0.0
            ),
            "judge_verdict": "",
            "total_ms": 0,
            "error": error if error else "",
        }

        rows.append(row)

    return pd.DataFrame(rows)


@beartype
def _calculate_experiment_summary(
    experiment: RanExperiment | dict[str, Any],
    results_df: Any,
) -> dict[str, Any]:
    """Calculate summary statistics from experiment results."""
    metrics = [
        "faithfulness_score",
        "context_precision_score",
        "context_recall_score",
        "answer_relevancy_score",
    ]
    averages = {}

    if not results_df.empty:
        # Filter out error rows
        valid_df = results_df[results_df["error"].isna() | (results_df["error"] == "")]
        if len(valid_df) > 0:
            for metric in metrics:
                if metric in valid_df.columns:
                    avg = valid_df[metric].mean()
                    averages[metric] = round(float(avg), 4)

    # Extract experiment metadata (handle both dict and object)
    if isinstance(experiment, dict):
        experiment_name = experiment.get("experiment_name", "")
        experiment_id = experiment.get("experiment_id", "")
        dataset_id = experiment.get("dataset_id", "")
    else:
        experiment_name = getattr(experiment, "experiment_name", "")
        experiment_id = getattr(experiment, "experiment_id", "")
        dataset_id = getattr(experiment, "dataset_id", "")

    error_count = (
        int(results_df["error"].ne("").sum())
        if "error" in results_df.columns
        else 0
    )

    return {
        "experiment_name": experiment_name,
        "experiment_id": experiment_id,
        "dataset_id": dataset_id,
        "metrics_avg": averages,
        "total_processed": len(results_df),
        "errors": error_count,
    }
