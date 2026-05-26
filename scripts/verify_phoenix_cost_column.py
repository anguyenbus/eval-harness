#!/usr/bin/env python3
"""
Verify Phoenix cost column name and schema.

This is a one-off manual script to verify the exact column name Phoenix uses
for cost data. Run with a tiny dataset (2-3 examples) to inspect the schema.

Usage:
    python scripts/verify_phoenix_cost_column.py

After running, update the verification comment in src/eval_harness/experiments/cost_summary.py
with the actual Phoenix version and date.

NOTE: This is a manual debugging tool, NOT a pytest test.
"""

from pathlib import Path
from typing import Any


def _create_tiny_dataset(client: Any) -> Any:
    """Create a tiny dataset with 2-3 examples for verification."""
    dataset_name = "cost-verification-tiny"

    # Try to get existing dataset
    try:
        existing = client.datasets.list_datasets()
        for ds in existing:
            if hasattr(ds, "name") and ds.name == dataset_name:
                return ds
    except Exception:
        pass

    # Create new tiny dataset
    examples = [
        {
            "input": {"input": "What is RAG?"},
            "output": {"expected": "RAG stands for Retrieval-Augmented Generation."},
            "metadata": {"query_id": "verify_1"},
        },
        {
            "input": {"input": "How does vector search work?"},
            "output": {"expected": "Vector search finds similar embeddings."},
            "metadata": {"query_id": "verify_2"},
        },
    ]

    dataset = client.datasets.create_dataset(
        dataset_name=dataset_name,
        inputs=[e["input"] for e in examples],
        outputs=[e["output"] for e in examples],
        metadata=[e["metadata"] for e in examples],
    )

    return dataset


def _create_simple_task() -> callable:
    """Create a simple task that returns static output."""
    def simple_task(example: Any) -> dict[str, Any]:
        question = (
            example.get("input", "")
            if isinstance(example, dict)
            else str(example)
        )
        return {
            "answer": f"Answer to: {question}",
            "retrieval_context": ["Context 1", "Context 2"],
        }
    return simple_task


def _create_simple_evaluators() -> list:
    """Create minimal evaluators for cost verification."""
    try:
        from phoenix.evals import create_evaluator

        @create_evaluator(name="dummy_evaluator")
        def dummy_evaluator(input: Any, output: Any, expected: Any = None) -> dict:
            """Dummy evaluator that returns a static score."""
            return {
                "score": 0.85,
                "label": "pass",
                "explanation": "Dummy evaluation for cost verification",
                "metadata": {"evaluation_cost": 0.001},
            }

        return [dummy_evaluator]
    except ImportError:
        print("[ERROR] phoenix.evals.create_evaluator not available")
        return []


def verify_cost_column() -> None:
    """Run tiny experiment and print cost column information."""
    try:
        from phoenix.client import Client
    except ImportError as err:
        print(f"[ERROR] Phoenix client not available: {err}")
        print("Install with: pip install arize-phoenix")
        return

    print("[INFO] Connecting to Phoenix at http://localhost:6006")
    client = Client(base_url="http://localhost:6006")

    print("[INFO] Creating tiny dataset (2-3 examples)")
    dataset = _create_tiny_dataset(client)

    print("[INFO] Creating simple task and evaluators")
    task = _create_simple_task()
    evaluators = _create_simple_evaluators()

    print("[INFO] Running experiment...")
    try:
        experiment = client.experiments.run_experiment(
            dataset=dataset,
            task=task,
            evaluators=evaluators,
            experiment_name="cost-column-verification",
        )
    except Exception as err:
        print(f"[ERROR] Experiment run failed: {err}")
        print("[INFO] This script requires Phoenix to be running at http://localhost:6006")
        return

    print("\n" + "=" * 60)
    print("EXPERIMENT RESULTS")
    print("=" * 60)

    # Print basic experiment info
    if hasattr(experiment, "experiment_id"):
        print(f"Experiment ID: {experiment.experiment_id}")
    if isinstance(experiment, dict):
        print(f"Experiment ID: {experiment.get('experiment_id', 'N/A')}")

    # Get dataframe
    print("\n[INFO] Getting experiment as dataframe...")
    try:
        df = experiment.as_dataframe() if hasattr(experiment, "as_dataframe") else None

        if df is None:
            print("[WARN] experiment.as_dataframe() not available, trying dict method")
            # Try to get from dict representation
            if isinstance(experiment, dict):
                import pandas as pd
                task_runs = experiment.get("task_runs", [])
                if task_runs:
                    rows = []
                    for run in task_runs:
                        if isinstance(run, dict):
                            rows.append(run)
                        else:
                            rows.append(vars(run))
                    df = pd.DataFrame(rows)
                else:
                    print("[ERROR] No task_runs found in experiment")
                    return
            else:
                print("[ERROR] Cannot extract dataframe from experiment")
                return

        print(f"\nDataFrame shape: {df.shape}")
        print(f"DataFrame columns ({len(df.columns)}):")
        for col in df.columns.tolist():
            print(f"  - {col}")

        print(f"\nDataFrame dtypes:")
        for col, dtype in df.dtypes.items():
            print(f"  - {col}: {dtype}")

        # Find cost-related columns
        print(f"\n[INFO] Searching for cost-related columns...")
        cost_columns = [col for col in df.columns if "cost" in col.lower()]
        if cost_columns:
            print(f"Cost-related columns found: {cost_columns}")
        else:
            print("[WARN] No columns with 'cost' in name found")

        # Check for common Phoenix cost column names
        possible_names = ["cost", "total_cost", "usage_cost", "llm_cost"]
        found_cost_col = None
        for name in possible_names:
            if name in df.columns:
                found_cost_col = name
                print(f"[INFO] Found cost column: '{name}'")
                break

        if found_cost_col:
            print(f"\n[INFO] Column '{found_cost_col}' statistics:")
            print(df[found_cost_col].describe())

            print(f"\n[INFO] First 3 values from '{found_cost_col}':")
            print(df[found_cost_col].head(3).tolist())
        else:
            print("[WARN] No known cost column found. Check columns list above.")

        # Print first row as example
        print(f"\n[INFO] First row data:")
        print(df.iloc[0].to_dict())

    except Exception as err:
        print(f"[ERROR] Failed to get dataframe: {err}")
        import traceback
        traceback.print_exc()
        return

    # Get Phoenix version
    try:
        import arize_phoenix
        phoenix_version = arize_phoenix.__version__
        print(f"\n[INFO] Phoenix version: {phoenix_version}")
        print("\n" + "=" * 60)
        print("VERIFICATION COMPLETE")
        print("=" * 60)
        print(f"\nUpdate APP_COST_COLUMN constant in cost_summary.py with:")
        print(f"# Verified against arize-phoenix=={phoenix_version} on [CURRENT_DATE]")
        print(f"# Re-run scripts/verify_phoenix_cost_column.py if upgrading Phoenix.")
        if found_cost_col:
            print(f"APP_COST_COLUMN: Final[str] = \"{found_cost_col}\"")
        else:
            print("[WARN] Could not determine cost column name automatically")
            print("Review the columns list above and set APP_COST_COLUMN manually")
    except Exception:
        print("\n[WARN] Could not determine Phoenix version")


if __name__ == "__main__":
    verify_cost_column()
