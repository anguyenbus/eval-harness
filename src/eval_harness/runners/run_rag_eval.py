"""
CLI runner for RAG evaluation on Legal RAG Bench.

Usage:
    uv run eval-rag --slice full

NOTE: The stub RAG option uses a ChromaDB-based reference implementation for demonstration purposes.
It is not intended for production use.

RAGAS LLM-judge metrics (Faithfulness, ContextPrecision, ContextRecall, AnswerRelevancy)
are enabled by default and require OPENAI_API_KEY to be set.
"""
from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from eval_harness.adapters.rag_adapter import RagAdapter

# Load environment variables from .env file
load_dotenv()


from eval_harness.config import load_config


def load_dataset(slice_name: str, config: dict):
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


def get_rag(rag_name: str, force_reingest: bool = False, top_k: int = 5) -> RagAdapter:
    """
    Get RAG adapter by name.

    NOTE: The 'stub' option uses a reference ChromaDB implementation
    for demonstration purposes. It is not intended for production use.

    Args:
        rag_name: Name of RAG system ('stub' uses ChromaDB-backed system).
        force_reingest: Force re-ingestion of corpus.
        top_k: Number of chunks to retrieve.

    Returns:
        RagAdapter instance.

    """
    if rag_name == "stub":
        # Use ChromaDB-backed query system (reference stub implementation)
        from eval_harness.stubs.rag.chromadb_query import query as chromadb_query

        # Wrap in adapter with config
        def chromadb_wrapper(question: str, corpus_dir: Path) -> dict[str, Any]:
            return chromadb_query(
                question=question,
                corpus_dir=corpus_dir,
                top_k=top_k,
                force_reingest=force_reingest,
            )

        return RagAdapter(query_callable=chromadb_wrapper)
    else:
        # For future: import custom RAG module
        print(f"WARNING: Custom RAG '{rag_name}' not implemented, using stub")
        print("NOTE: Stub implementation is for demonstration only")
        from eval_harness.stubs.rag.chromadb_query import query as chromadb_query

        def chromadb_wrapper(question: str, corpus_dir: Path) -> dict[str, Any]:
            return chromadb_query(
                question=question,
                corpus_dir=corpus_dir,
                top_k=top_k,
                force_reingest=force_reingest,
            )

        return RagAdapter(query_callable=chromadb_wrapper)


def main() -> None:
    """Main entry point for eval-rag CLI."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate RAG systems on Legal RAG Bench with RAGAS metrics"
    )
    parser.add_argument(
        "--slice",
        choices=["nano", "full"],
        default="full",
        help="Dataset slice (nano=10 queries, full=100 queries)",
    )
    parser.add_argument(
        "--rag",
        default="stub",
        help="RAG system to use (NOTE: 'stub' is a reference implementation for demonstration only, not for production)",
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
        default=Path("results"),
        help="Output directory for CSV results",
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

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load Legal RAG Bench dataset
    dataset_config = config["datasets"].get("legal_rag_bench", {})
    print(f"Loading Legal RAG Bench dataset ({args.slice} slice)")
    dataset = load_dataset(args.slice, config)

    corpus_dir = Path(dataset_config.get("path", "data/rag/legal_rag_bench"))

    # Initialize RAGAS evaluator (always enabled)
    try:
        from eval_harness.adapters.ragas_adapter import RagasEvaluator

        ragas_config = dataset_config.get("ragas", {})
        judge_model = ragas_config.get("judge_model", "gpt-4o")
        llm_provider = ragas_config.get("judge_model_provider", "openai")

        ragas_evaluator = RagasEvaluator(
            llm_provider=llm_provider,
            judge_model=judge_model,
        )
        print(f"RAGAS evaluation enabled with {llm_provider}/{judge_model}")
    except Exception as e:
        print(f"ERROR: Could not initialize RAGAS evaluator: {e}")
        print("RAGAS metrics are required. Please set OPENAI_API_KEY.")
        sys.exit(1)

    # Get RAG system
    print(f"Using RAG system: {args.rag}")
    if args.rag == "stub":
        print("NOTE: Using reference stub implementation (demonstration only)")
    if args.force_reingest:
        print(f"Force reingest enabled")
    print(f"Top-k retrieval: {args.top_k}")

    rag_adapter = get_rag(
        args.rag,
        force_reingest=args.force_reingest,
        top_k=args.top_k,
    )

    # Setup output file for incremental writes with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = args.output_dir / f"legal_rag_bench_{args.slice}_results_{timestamp}.csv"
    file_exists = output_file.exists()

    # Define CSV columns - RAGAS metrics + Legal RAG Bench specific only
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
    ]

    # Open CSV for incremental appending
    csv_file = open(output_file, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    # Write header if new file
    if not file_exists:
        writer.writeheader()

    processed = 0
    errors = 0

    for query_id, query_text, relevant_passage_id, gold_answer_text in dataset:
        print(f"Processing query {query_id}...")

        try:
            # Query RAG system
            output = rag_adapter.query(query_text, corpus_dir)

            # Extract fields from output
            retrieved_chunks = output.get("retrieved_chunks", [])
            timings = output.get("timings_ms", {})
            generated_answer = output.get("answer", {}).get("text", "")

            # Check if relevant passage was retrieved
            relevant_passage_retrieved = any(
                chunk.get("doc_id") == relevant_passage_id
                for chunk in retrieved_chunks
            )

            # Compute RAGAS metrics (always enabled)
            ragas_scores = ragas_evaluator.compute_metrics(output, gold_answer_text)

            # Prepare result row
            result = {
                "query_id": query_id,
                "question": query_text,
                "gold_answer": gold_answer_text,
                "generated_answer": generated_answer,
                "relevant_passage_retrieved": relevant_passage_retrieved,
                "faithfulness_score": ragas_scores.get("faithfulness", 0.0),
                "context_precision_score": ragas_scores.get("context_precision", 0.0),
                "context_recall_score": ragas_scores.get("context_recall", 0.0),
                "answer_relevancy_score": ragas_scores.get("answer_relevancy", 0.0),
                "judge_verdict": "PASS" if ragas_scores.get("faithfulness", 0.0) > 0.7 else "NEEDS_REVIEW",
                "total_ms": timings.get("total", 0),
                "error": "",
            }

            # Write result
            writer.writerow(result)
            csv_file.flush()
            processed += 1

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
            writer.writerow(error_result)
            csv_file.flush()
            errors += 1

    csv_file.close()

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
    }
    with open(json_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults written to: {output_file}")
    print(f"Total queries processed: {processed}")
    print(f"Errors: {errors}")
    print(f"Summary written to: {json_file}")

    sys.exit(0)


if __name__ == "__main__":
    main()
