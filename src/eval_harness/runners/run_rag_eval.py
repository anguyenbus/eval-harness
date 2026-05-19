"""
CLI runner for RAG evaluation.

Usage:
    uv run eval-rag --dataset legalbench_rag --slice mini

NOTE: The stub RAG option uses a ChromaDB-based reference implementation for demonstration purposes.
It is not intended for production use.
"""
from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from eval_harness.adapters.rag_adapter import RagAdapter


def _calculate_recall_at_k(
    gold_spans: list[list[int]], retrieved_chunks: list[dict[str, Any]]
) -> dict[str, float]:
    """
    Calculate recall metrics based on gold span overlap with retrieved chunks.

    Args:
        gold_spans: List of [start, end) gold spans from dataset.
        retrieved_chunks: List of retrieved chunk dicts with char_span.

    Returns:
        Dict with recall_at_k, precision_at_k, num_relevant_chunks.
    """
    if not gold_spans or not retrieved_chunks:
        return {"recall_at_k": 0.0, "precision_at_k": 0.0, "num_relevant": 0}

    relevant_count = 0
    k = len(retrieved_chunks)

    for chunk in retrieved_chunks:
        chunk_span = chunk.get("char_span", [])
        if not chunk_span or len(chunk_span) != 2:
            continue

        chunk_start, chunk_end = chunk_span

        # Check if this chunk overlaps with any gold span
        for gold_start, gold_end in gold_spans:
            # Overlap exists if intervals intersect
            overlap_start = max(chunk_start, gold_start)
            overlap_end = min(chunk_end, gold_end)

            if overlap_start < overlap_end:
                relevant_count += 1
                break  # Count chunk once even if it overlaps multiple gold spans

    recall_at_k = 1.0 if relevant_count > 0 else 0.0
    precision_at_k = relevant_count / k if k > 0 else 0.0

    return {
        "recall_at_k": recall_at_k,
        "precision_at_k": precision_at_k,
        "num_relevant": relevant_count,
    }


def _token_f1(gold: str, predicted: str) -> dict[str, float]:
    """
    Calculate token-level F1 score between gold and predicted answers.

    Args:
        gold: Reference answer text.
        predicted: Generated answer text.

    Returns:
        Dict with f1, precision, recall scores.
    """
    if not gold and not predicted:
        return {"f1": 1.0, "precision": 1.0, "recall": 1.0}
    if not gold or not predicted:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0}

    gold_tokens = set(gold.lower().split())
    pred_tokens = set(predicted.lower().split())

    if not gold_tokens or not pred_tokens:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0}

    common = gold_tokens & pred_tokens
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)

    if precision + recall == 0:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0}

    f1 = 2 * precision * recall / (precision + recall)

    return {"f1": f1, "precision": precision, "recall": recall}
from eval_harness.config import load_config


def load_dataset(dataset_name: str, slice_name: str, config: dict):
    """
    Load dataset by name and slice.

    Args:
        dataset_name: Name of dataset (currently only 'legalbench_rag').
        slice_name: Slice of dataset ('mini' or 'full').
        config: Configuration dictionary.

    Returns:
        Iterator over dataset items.

    """
    if dataset_name == "legalbench_rag":
        from eval_harness.datasets import load_legalbench_rag

        root = Path(config["datasets"]["legalbench_rag"]["path"])
        return load_legalbench_rag(root, slice=slice_name)

    else:
        print(f"ERROR: Unknown dataset: {dataset_name}")
        print("Supported datasets: legalbench_rag")
        sys.exit(1)


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
        description="Evaluate RAG systems on public benchmarks"
    )
    parser.add_argument(
        "--dataset",
        choices=["legalbench_rag"],
        default="legalbench_rag",
        help="Dataset to evaluate on",
    )
    parser.add_argument(
        "--slice",
        choices=["nano", "mini", "full"],
        default="mini",
        help="Dataset slice (nano has 48 queries, mini has 776, full has 6,889)",
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

    # Load dataset
    print(f"Loading dataset: {args.dataset} ({args.slice} slice)")
    dataset = load_dataset(args.dataset, args.slice, config)

    # Get corpus directory from config
    corpus_dir = Path(config["datasets"]["legalbench_rag"]["path"])

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
    output_file = args.output_dir / f"{args.dataset}_{args.slice}_results_{timestamp}.csv"
    file_exists = output_file.exists()

    # Define all CSV columns
    fieldnames = [
        "query_id",
        "question",
        "gold_answer",
        "generated_answer",
        "answer_supported",
        "recall_at_k",
        "precision_at_k",
        "num_relevant_chunks",
        "num_citations",
        "citation_precision",
        "f1_score",
        "exact_match",
        "retrieval_ms",
        "generation_ms",
        "total_ms",
        "top_k",
        "retrieved_chunk_ids",
        "retrieval_scores",
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

    for query_id, query_text, gold_spans, gold_answer_text in dataset:
        print(f"Processing query {query_id}...")

        try:
            # Query RAG system
            output = rag_adapter.query(query_text, corpus_dir)

            # Extract fields from output
            answer = output.get("answer", {})
            retrieved_chunks = output.get("retrieved_chunks", [])
            timings = output.get("timings_ms", {})

            generated_answer = answer.get("text", "")
            answer_supported = answer.get("answer_supported", False)
            citations = answer.get("citations", [])

            # Calculate retrieval metrics
            recall_metrics = _calculate_recall_at_k(gold_spans, retrieved_chunks)

            # Calculate answer quality metrics
            f1_metrics = _token_f1(gold_answer_text, generated_answer)
            exact_match = 1.0 if gold_answer_text.strip().lower() == generated_answer.strip().lower() else 0.0

            # Extract chunk IDs and scores
            chunk_ids = [c.get("chunk_id", "") for c in retrieved_chunks]
            chunk_scores = [c.get("score", 0.0) for c in retrieved_chunks]

            # Calculate citation precision
            valid_citations = 0
            for citation in citations:
                cited_ids = citation.get("chunk_ids", [])
                if any(cid in chunk_ids for cid in cited_ids):
                    valid_citations += 1

            citation_precision = valid_citations / len(citations) if citations else 1.0

            # Write comprehensive result
            writer.writerow(
                {
                    "query_id": query_id,
                    "question": query_text,
                    "gold_answer": gold_answer_text,
                    "generated_answer": generated_answer,
                    "answer_supported": answer_supported,
                    "recall_at_k": recall_metrics["recall_at_k"],
                    "precision_at_k": recall_metrics["precision_at_k"],
                    "num_relevant_chunks": recall_metrics["num_relevant"],
                    "num_citations": len(citations),
                    "citation_precision": citation_precision,
                    "f1_score": f1_metrics["f1"],
                    "exact_match": exact_match,
                    "retrieval_ms": timings.get("retrieval", 0),
                    "generation_ms": timings.get("generation", 0),
                    "total_ms": timings.get("total", 0),
                    "top_k": args.top_k,
                    "retrieved_chunk_ids": "|".join(chunk_ids),
                    "retrieval_scores": "|".join(f"{s:.4f}" for s in chunk_scores),
                    "error": "",
                }
            )
            csv_file.flush()
            processed += 1

        except Exception as e:
            writer.writerow(
                {
                    "query_id": query_id,
                    "question": query_text,
                    "gold_answer": gold_answer_text,
                    "generated_answer": "",
                    "answer_supported": False,
                    "recall_at_k": 0.0,
                    "precision_at_k": 0.0,
                    "num_relevant_chunks": 0,
                    "num_citations": 0,
                    "citation_precision": 0.0,
                    "f1_score": 0.0,
                    "exact_match": 0.0,
                    "retrieval_ms": 0,
                    "generation_ms": 0,
                    "total_ms": 0,
                    "top_k": args.top_k,
                    "retrieved_chunk_ids": "",
                    "retrieval_scores": "",
                    "error": str(e),
                }
            )
            csv_file.flush()
            errors += 1

    csv_file.close()

    # Calculate metric averages - reload CSV
    df = pd.read_csv(output_file)
    metrics = [
        "recall_at_k",
        "precision_at_k",
        "citation_precision",
        "f1_score",
        "exact_match",
        "answer_supported",
        "retrieval_ms",
        "generation_ms",
        "total_ms",
    ]

    averages = {}
    if not df.empty:
        # Filter out error rows (error column is NaN for no errors, or has error message)
        valid_df = df[df["error"].isna() | (df["error"] == "")]
        if len(valid_df) > 0:
            print("\nMetric averages:")
            for metric in metrics:
                avg = valid_df[metric].mean()
                averages[metric] = round(float(avg), 4)
                print(f"  {metric}: {avg:.4f}")

    # Write JSON summary with same base filename
    json_file = output_file.with_suffix(".json")
    summary = {
        "dataset": args.dataset,
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
