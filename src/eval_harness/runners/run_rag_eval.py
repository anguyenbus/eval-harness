"""
CLI runner for RAG evaluation on Legal RAG Bench.

Usage:
    uv run eval-rag --slice full --rag stub-local

NOTE: The stub-local RAG option uses a ChromaDB-based reference implementation
for demonstration purposes. It is not intended for production use.

DeepEval LLM-judge metrics (Faithfulness, ContextualPrecision, ContextualRecall,
AnswerRelevancy) are enabled by default and require OPENAI_API_KEY to be set.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from beartype import beartype
from dotenv import load_dotenv

from eval_harness.adapters.rag_adapter import RagAdapter
from eval_harness.config import load_config

# ====================================================================
# SECURITY: DISABLE THIRD-PARTY TELEMETRY
# ====================================================================
# DO NOT REMOVE OR MODIFY. See deepeval_config.py for full explanation.
# This ensures telemetry is disabled even if this module is imported directly.
os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "YES"

# Load environment variables from .env file
load_dotenv()


def _process_query(
    query_id: str,
    query_text: str,
    relevant_passage_id: str,
    gold_answer_text: str,
    corpus_dir: Path,
    rag_adapter: RagAdapter,
    evaluator: Any,
    writer: Any,
    csv_file: Any,
    phoenix_adapter: Any,
    trace_id: str | None,
    framework_version: str | None = None,
    judge_model: str | None = None,
) -> tuple[bool, int, dict[str, Any]]:  # (success, trace_count_increment, reasoning)
    """
    Process a single query and write result to CSV.

    Returns:
        (success, trace_increment): Whether query succeeded and trace count increment.

    """
    try:
        # Query RAG system
        output = rag_adapter.query(query_text, corpus_dir)

        # Extract fields from output
        retrieved_chunks = output.get("retrieved_chunks", [])
        timings = output.get("timings_ms", {})
        generated_answer = output.get("answer", {}).get("text", "")

        # Record retrieval span in Phoenix (if enabled)
        if phoenix_adapter and trace_id:
            phoenix_adapter.start_retrieval_span(
                trace_id=trace_id,
                query_text=query_text,
                chunks=retrieved_chunks,
                k=len(retrieved_chunks),
                timing_ms=timings.get("retrieval", 0),
            )

            # Record generation span
            model = output.get("system_version", {}).get("generator_model", "unknown")
            phoenix_adapter.start_generation_span(
                trace_id=trace_id,
                model=model,
                prompt=query_text,
                tokens=0,
                timing_ms=timings.get("generation", 0),
            )

        # Check if relevant passage was retrieved
        relevant_passage_retrieved = any(
            chunk.get("doc_id") == relevant_passage_id for chunk in retrieved_chunks
        )

        # Compute metrics with full reasoning (DeepEval)
        import time as time_module

        metric_start = time_module.time()
        metric_result = evaluator.compute_metrics_with_reasoning(
            output, gold_answer_text
        )
        metric_end = time_module.time()

        metric_scores = metric_result["scores"]
        metric_reasoning = metric_result["reasoning"]
        metric_scores["metric_computation_time_ms"] = (metric_end - metric_start) * 1000

        # Determine verdict
        faithfulness = metric_scores.get("faithfulness", 0.0)
        verdict = "PASS" if faithfulness > 0.7 else "NEEDS_REVIEW"

        # Record evaluation span in Phoenix (if enabled)
        if phoenix_adapter and trace_id:
            phoenix_adapter.start_evaluation_span(
                trace_id, metric_scores, verdict=verdict, reasoning=metric_reasoning
            )

        # Prepare result row
        result = {
            "query_id": query_id,
            "question": query_text,
            "gold_answer": gold_answer_text,
            "generated_answer": generated_answer,
            "relevant_passage_retrieved": relevant_passage_retrieved,
            "faithfulness_score": faithfulness,
            "context_precision_score": metric_scores.get("context_precision", 0.0),
            "context_recall_score": metric_scores.get("context_recall", 0.0),
            "answer_relevancy_score": metric_scores.get("answer_relevancy", 0.0),
            "judge_verdict": verdict,
            "total_ms": timings.get("total", 0),
            "error": "",
        }

        # Add metadata columns if provided
        if framework_version is not None:
            result["framework_version"] = framework_version
        if judge_model is not None:
            result["llm_judge_model"] = judge_model
        if "metric_computation_time_ms" in metric_scores:
            result["metric_computation_time_ms"] = metric_scores[
                "metric_computation_time_ms"
            ]

        # Write result
        writer.writerow(result)
        csv_file.flush()
        return True, 1 if phoenix_adapter else 0, metric_reasoning

    except Exception as e:
        error_result = {
            "query_id": query_id,
            "question": query_text,
            "gold_answer": gold_answer_text,
            "generated_answer": "",
            "relevant_passage_retrieved": False,
            "faithfulness_score": 0.0,
            "context_precision_score": 0.0,
            "context_recall_score": 0.0,
            "answer_relevancy_score": 0.0,
            "judge_verdict": "ERROR",
            "total_ms": 0,
            "error": str(e),
        }

        # Add metadata columns with empty values if error
        if framework_version is not None:
            error_result["framework_version"] = ""
        if judge_model is not None:
            error_result["llm_judge_model"] = ""
        error_result["metric_computation_time_ms"] = 0

        writer.writerow(error_result)
        csv_file.flush()
        return False, 0, {}


def load_dataset(slice_name: str, config: dict) -> Any:
    """
    Load Legal RAG Bench dataset by slice.

    Args:
        slice_name: Slice of dataset ('nano' or 'full').
        config: Configuration dictionary.

    Returns:
        Iterator over dataset items.

    """
    from eval_harness.datasets import load_legal_rag_bench

    dataset_config = config["datasets"].get("legal_rag_bench", {})
    cache_dir = Path(dataset_config.get("cache_path", "data/rag/legal_rag_bench"))

    return load_legal_rag_bench(cache_dir=cache_dir, slice=slice_name)


def get_rag(
    rag_name: str,
    force_reingest: bool = False,
    top_k: int = 5,
    embedder: Any = None,
) -> RagAdapter:
    """
    Get RAG adapter by name.

    NOTE: The 'stub-local' option uses a reference ChromaDB implementation
    for demonstration purposes. It is not intended for production use.

    Args:
        rag_name: Name of RAG system ('stub-local' uses ChromaDB-backed system).
        force_reingest: Force re-ingestion of corpus.
        top_k: Number of chunks to retrieve.
        embedder: Optional shared embedder instance.

    Returns:
        RagAdapter instance.

    """
    if rag_name == "stub-local":
        # Use ChromaDB-backed query system (reference stub implementation)
        from eval_harness.stubs.rag.chromadb_query import query as chromadb_query

        # Wrap in adapter with config
        def chromadb_wrapper(
            question: str, corpus_dir: Path, embedder: Any = None
        ) -> dict[str, Any]:
            return chromadb_query(
                question=question,
                corpus_dir=corpus_dir,
                top_k=top_k,
                force_reingest=force_reingest,
                embedder=embedder,
            )

        return RagAdapter(query_callable=chromadb_wrapper, embedder=embedder)
    else:
        # For future: import custom RAG module
        print(f"WARNING: Custom RAG '{rag_name}' not implemented, using stub-local")
        print("NOTE: stub-local implementation is for demonstration only")
        from eval_harness.stubs.rag.chromadb_query import query as chromadb_query

        def chromadb_wrapper(
            question: str, corpus_dir: Path, embedder: Any = None
        ) -> dict[str, Any]:
            return chromadb_query(
                question=question,
                corpus_dir=corpus_dir,
                top_k=top_k,
                force_reingest=force_reingest,
                embedder=embedder,
            )

        return RagAdapter(query_callable=chromadb_wrapper, embedder=embedder)


@beartype
def _run_phoenix_native(
    args: Any,
    config: dict[str, Any],
    phoenix_config: dict[str, Any],
) -> None:
    """
    Run evaluation using Phoenix Native experiment API.

    Args:
        args: Parsed CLI arguments.
        config: Loaded configuration dictionary.
        phoenix_config: Phoenix configuration dictionary.

    """
    from pathlib import Path

    from eval_harness.adapters.embeddings import get_embedder
    from eval_harness.experiments.runner import (
        export_experiment_results,
        run_phoenix_experiment,
    )

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("results") / "eval_rag" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get dataset config
    dataset_config = config["datasets"].get("legal_rag_bench", {})
    corpus_dir = Path(dataset_config.get("path", "data/rag/legal_rag_bench"))

    # Create shared embedder
    embeddings_config = dataset_config.get("embeddings", {})
    embedder_provider = embeddings_config.get("provider", "huggingface")
    embedder_model = embeddings_config.get(
        "model", "sentence-transformers/all-MiniLM-L6-v2"
    )

    embedder = get_embedder(provider=embedder_provider, model=embedder_model)
    print(f"Shared embedder: {embedder_provider}/{embedder_model}")

    # Get RAG adapter
    rag_adapter = get_rag(
        args.rag,
        force_reingest=args.force_reingest,
        top_k=args.top_k,
        embedder=embedder,
    )

    # Get judge model config
    from eval_harness.metrics.deepeval_config import get_deepeval_config

    deepeval_config = get_deepeval_config(config)
    judge_model = deepeval_config["judge_model"]

    print("Running Phoenix Native experiment...")
    print(f"  Phoenix endpoint: {phoenix_config['endpoint']}")
    print(f"  Dataset slice: {args.slice}")
    print(f"  Judge model: {judge_model}")

    # Run experiment
    experiment = run_phoenix_experiment(
        rag_adapter=rag_adapter,
        corpus_dir=corpus_dir,
        endpoint=phoenix_config["endpoint"],
        slice_name=args.slice,
        experiment_name=f"rag-eval-{args.slice}-{timestamp}",
        judge_model=judge_model,
    )

    print(f"Experiment completed: {experiment.get('experiment_name', 'unknown')}")
    print(f"View results at: {phoenix_config['endpoint']}/datasets")

    # Export results using Phoenix native methods
    export_result = export_experiment_results(
        experiment=experiment,
        output_dir=output_dir,
    )
    print(f"Results exported to: {export_result['csv_path']}")
    print(f"Summary exported to: {export_result['json_path']}")

    sys.exit(0)


def main() -> None:
    """Run RAG evaluation on Legal RAG Bench dataset."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate RAG systems on Legal RAG Bench with DeepEval metrics"
    )
    parser.add_argument(
        "--slice",
        choices=["nano", "full"],
        default="full",
        help="Dataset slice (nano=10 queries, full=100 queries)",
    )
    parser.add_argument(
        "--rag",
        required=True,
        choices=["stub-local"],
        help=(
            "RAG system to use. Options: stub-local "
            "(ChromaDB-backed reference implementation)"
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("eval_config.yaml"),
        help="Path to eval_config.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for CSV results (default: results/eval_rag/TIMESTAMP)",
    )
    parser.add_argument(
        "--force-reingest",
        action="store_true",
        help="Force re-ingestion of corpus into ChromaDB",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve (default: 5)",
    )

    # Phoenix integration flags
    parser.add_argument(
        "--enable-phoenix",
        action="store_true",
        help="Enable Phoenix observability for RAG pipeline tracing",
    )
    parser.add_argument(
        "--phoenix-endpoint",
        type=str,
        default=None,
        help="Phoenix server endpoint (e.g., http://localhost:6006 or https://phoenix.example.com)",
    )

    args = parser.parse_args()

    # Load configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    # Initialize Phoenix if enabled
    phoenix_adapter = None
    phoenix_config = None

    if args.enable_phoenix:
        try:
            from eval_harness.observability.config import get_phoenix_config

            phoenix_config = get_phoenix_config(
                config,
                cli_enabled=args.enable_phoenix,
                cli_endpoint=args.phoenix_endpoint,
            )

            phoenix_mode = phoenix_config.get("mode", "spans")

            # Check for native mode
            if phoenix_mode == "native":
                print("Phoenix Native mode enabled (experiment API)")
                return _run_phoenix_native(
                    args=args,
                    config=config,
                    phoenix_config=phoenix_config,
                )

            # Spans mode (original)
            from eval_harness.observability.phoenix_adapter import PhoenixAdapter

            phoenix_adapter = PhoenixAdapter(
                endpoint=phoenix_config["endpoint"],
                project_name="eval-harness",
                enabled=phoenix_config["enabled"],
                export_path=Path(phoenix_config["export_path"]),
            )

            if phoenix_adapter.is_connected():
                print(f"Phoenix UI available at: {phoenix_config['endpoint']}")
            else:
                export = phoenix_config["export_path"]
                print(f"Phoenix unreachable. Traces will be buffered to: {export}")
        except ImportError:
            print("WARNING: Phoenix not installed.")
            print("Install with: `uv pip install eval-harness[phoenix]`")
            print("Continuing evaluation without Phoenix observability...")
        except ValueError as e:
            print(f"ERROR: Invalid Phoenix configuration: {e}")
            sys.exit(1)

    # Create output directory

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir is None:
        args.output_dir = Path("results") / "eval_rag" / timestamp
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load Legal RAG Bench dataset
    dataset_config = config["datasets"].get("legal_rag_bench", {})
    print(f"Loading Legal RAG Bench dataset ({args.slice} slice)")
    dataset = load_dataset(args.slice, config)

    corpus_dir = Path(dataset_config.get("path", "data/rag/legal_rag_bench"))

    # Create shared embedder (used by both RAG retrieval and DeepEval)
    try:
        from eval_harness.adapters.embeddings import get_embedder

        embeddings_config = dataset_config.get("embeddings", {})
        embedder_provider = embeddings_config.get("provider", "huggingface")
        embedder_model = embeddings_config.get(
            "model", "sentence-transformers/all-MiniLM-L6-v2"
        )

        embedder = get_embedder(provider=embedder_provider, model=embedder_model)
        print(f"Shared embedder: {embedder_provider}/{embedder_model}")
    except Exception as e:
        print(f"ERROR: Could not initialize embedder: {e}")
        sys.exit(1)

    # Initialize DeepEval evaluator (always enabled)
    try:
        from eval_harness.adapters.deepeval_adapter import DeepEvalEvaluator
        from eval_harness.metrics.deepeval_config import get_deepeval_config

        deepeval_config = get_deepeval_config(config)
        judge_model = deepeval_config["judge_model"]
        llm_provider = deepeval_config["judge_model_provider"]
        temperature = deepeval_config["temperature"]
        max_concurrent = deepeval_config["max_concurrent"]

        evaluator = DeepEvalEvaluator(
            llm_provider=llm_provider,
            judge_model=judge_model,
            temperature=temperature,
            max_concurrent=max_concurrent,
            embedder=embedder,
        )

        # Get framework version for metadata
        try:
            import deepeval

            framework_version = deepeval.__version__
        except (ImportError, AttributeError):
            framework_version = "unknown"

        print(f"DeepEval evaluation enabled with {llm_provider}/{judge_model}")
        print(f"Framework version: {framework_version}")
        print(f"Max concurrent evaluations: {max_concurrent}")
    except Exception as e:
        print(f"ERROR: Could not initialize DeepEval evaluator: {e}")
        print("DeepEval metrics are required. Please set OPENAI_API_KEY.")
        sys.exit(1)

    # Get RAG system
    print(f"Using RAG system: {args.rag}")
    if args.rag == "stub-local":
        print("NOTE: Using reference stub-local implementation (demonstration only)")
    if args.force_reingest:
        print("Force reingest enabled")
    print(f"Top-k retrieval: {args.top_k}")

    rag_adapter = get_rag(
        args.rag,
        force_reingest=args.force_reingest,
        top_k=args.top_k,
        embedder=embedder,
    )

    # Setup output file for incremental writes
    output_file = args.output_dir / f"legal_rag_bench_{args.slice}_results.csv"
    file_exists = output_file.exists()

    # Define CSV columns - maintain backward compatibility with RAGAS schema
    # plus additional metadata columns for DeepEval
    fieldnames = [
        "query_id",
        "question",
        "gold_answer",
        "generated_answer",
        "relevant_passage_retrieved",
        "faithfulness_score",
        "context_precision_score",
        "context_recall_score",
        "answer_relevancy_score",
        "judge_verdict",
        "total_ms",
        "error",
        # Additional metadata columns
        "framework_version",
        "metric_computation_time_ms",
        "llm_judge_model",
    ]

    # Open CSV for incremental appending
    csv_file = open(output_file, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    # Write header if new file
    if not file_exists:
        writer.writeheader()

    processed = 0
    errors = 0
    trace_count = 0
    all_reasoning: list[dict[str, Any]] = []

    # Convert dataset to list to get count
    dataset_list = list(dataset)
    num_questions = len(dataset_list)

    # Use Phoenix eval_run span to group all queries
    if phoenix_adapter:
        eval_metadata = {
            "slice": args.slice,
            "top_k": args.top_k,
            "rag_system": args.rag,
            "evaluation_framework": "deepeval",
            "framework_version": framework_version,
            "judge_model": judge_model,
        }
        with phoenix_adapter.eval_run_span(
            f"legal-rag-bench-{args.slice}",
            num_questions=num_questions,
            metadata=eval_metadata,
        ):
            for (
                query_id,
                query_text,
                relevant_passage_id,
                gold_answer_text,
            ) in dataset_list:
                print(f"Processing query {query_id}...")
                with phoenix_adapter.rag_query_span(query_text) as trace_id:
                    success, trace_inc, reasoning = _process_query(
                        query_id,
                        query_text,
                        relevant_passage_id,
                        gold_answer_text,
                        corpus_dir,
                        rag_adapter,
                        evaluator,
                        writer,
                        csv_file,
                        phoenix_adapter,
                        trace_id,
                        framework_version=framework_version,
                        judge_model=judge_model,
                    )
                    if success:
                        processed += 1
                    else:
                        errors += 1
                    trace_count += trace_inc
                    # Collect reasoning for details.json
                    if reasoning:
                        all_reasoning.append(
                            {
                                "query_id": query_id,
                                "question": query_text,
                                "reasoning": reasoning,
                            }
                        )
    else:
        for query_id, query_text, relevant_passage_id, gold_answer_text in dataset_list:
            print(f"Processing query {query_id}...")
            success, trace_inc, reasoning = _process_query(
                query_id,
                query_text,
                relevant_passage_id,
                gold_answer_text,
                corpus_dir,
                rag_adapter,
                evaluator,
                writer,
                csv_file,
                None,
                None,
                framework_version=framework_version,
                judge_model=judge_model,
            )
            if success:
                processed += 1
            else:
                errors += 1
            trace_count += trace_inc
            # Collect reasoning for details.json
            if reasoning:
                all_reasoning.append(
                    {
                        "query_id": query_id,
                        "question": query_text,
                        "reasoning": reasoning,
                    }
                )

    csv_file.close()

    # Export Phoenix traces
    if phoenix_adapter:
        export_result = phoenix_adapter.export_traces()
        print(f"Phoenix traces exported: {export_result['trace_count']} traces")
        if export_result["mode"] == "parquet" and export_result["path"]:
            print(f"Traces buffered to Parquet: {export_result['path']}")

    # Calculate metric averages - reload CSV
    df = pd.read_csv(output_file)
    metrics = [
        "relevant_passage_retrieved",
        "faithfulness_score",
        "context_precision_score",
        "context_recall_score",
        "answer_relevancy_score",
        "total_ms",
    ]

    averages = {}
    if not df.empty:
        # Filter out error rows
        valid_df = df[df["error"].isna() | (df["error"] == "")]
        if len(valid_df) > 0:
            print("\nMetric averages:")
            for metric in metrics:
                if metric in valid_df.columns:
                    avg = valid_df[metric].mean()
                    averages[metric] = round(float(avg), 4)
                    print(f"  {metric}: {avg:.4f}")

    # Write JSON summary with same base filename
    json_file = output_file.with_suffix(".json")
    summary = {
        "dataset": "legal_rag_bench",
        "slice": args.slice,
        "timestamp": timestamp,
        "csv_file": str(output_file.name),
        "metrics_avg": averages,
        "total_processed": processed,
        "errors": errors,
        "top_k": args.top_k,
        "evaluation_framework": "deepeval",
        "framework_version": framework_version,
        "judge_model": judge_model,
    }

    # Add Phoenix info to summary if enabled
    if phoenix_adapter:
        summary["phoenix"] = {
            "enabled": True,
            "trace_count": trace_count,
            "endpoint": phoenix_adapter._endpoint,
        }

    with open(json_file, "w") as f:
        json.dump(summary, f, indent=2)

    # Write details.json with full reasoning
    if all_reasoning:
        details_file = output_file.stem + "_details.json"
        details_path = args.output_dir / details_file
        details = {
            "dataset": "legal_rag_bench",
            "slice": args.slice,
            "timestamp": timestamp,
            "total_queries": len(all_reasoning),
            "evaluation_framework": "deepeval",
            "framework_version": framework_version,
            "judge_model": judge_model,
            "queries": all_reasoning,
        }
        with open(details_path, "w") as f:
            json.dump(details, f, indent=2)
        print(f"Details with reasoning written to: {details_path}")

    print(f"\nResults written to: {output_file}")
    print(f"Total queries processed: {processed}")
    print(f"Errors: {errors}")
    print(f"Summary written to: {json_file}")

    sys.exit(0)


if __name__ == "__main__":
    main()
