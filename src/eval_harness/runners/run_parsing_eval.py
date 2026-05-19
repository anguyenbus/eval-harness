"""
CLI runner for parsing evaluation.

Usage:
    uv run eval-parsing --dataset omnidocbench --parser docling
    uv run eval-parsing --dataset omnidocbench --parser stub
    uv run eval-parsing --dataset dp_bench --parser stub
"""

import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from eval_harness.adapters.parser_adapter import ParserAdapter
from eval_harness.config import load_config
from eval_harness.metrics.parsing.markdown_converter import parser_output_to_markdown
from eval_harness.metrics.parsing.mhs import evaluate_heading_level
from eval_harness.metrics.parsing.nid import evaluate_reading_order as evaluate_nid
from eval_harness.metrics.parsing.reading_order import ard_score
from eval_harness.metrics.parsing.table_teds import evaluate_table
from eval_harness.metrics.parsing.text_similarity import bleu_score, meteor_score


def load_dataset(dataset_name: str, config: dict):
    """
    Load dataset by name.

    Args:
        dataset_name: Name of dataset ('omnidocbench' or 'dp_bench').
        config: Configuration dictionary.

    Returns:
        Iterator over dataset items.

    """
    if dataset_name == "omnidocbench":
        from eval_harness.datasets import load_omnidocbench

        root = Path(config["datasets"]["omnidocbench"]["path"])
        return load_omnidocbench(root)

    elif dataset_name == "dp_bench":
        from eval_harness.datasets import load_dp_bench

        root = Path(config["datasets"]["dp_bench"]["path"])
        return load_dp_bench(root)

    else:
        print(f"ERROR: Unknown dataset: {dataset_name}")
        print("Supported datasets: omnidocbench, dp_bench")
        sys.exit(1)


def get_parser(parser_name: str) -> tuple[ParserAdapter, str]:
    """
    Get parser adapter by name.

    Args:
        parser_name: Name of parser ('stub', 'docling', or path to parser module).

    Returns:
        Tuple of (ParserAdapter instance, parser_type).

    """
    if parser_name == "stub":
        from eval_harness.stubs.stub_parser import parse as parse_func
        return ParserAdapter(parse_func), "stub"

    elif parser_name == "fast":
        from eval_harness.stubs.digital_pdf_parser import parse as parse_func
        return ParserAdapter(parse_func), "fast"

    elif parser_name == "docling":
        try:
            from eval_harness.stubs.docling_parser import parse as parse_func
            return ParserAdapter(parse_func), "docling"
        except ImportError as e:
            print(f"ERROR: {e}")
            print("Install docling with: uv add docling")
            sys.exit(1)

    else:
        # For future: import custom parser module
        print(f"WARNING: Custom parser '{parser_name}' not implemented, using stub")
        from eval_harness.stubs.stub_parser import parse as parse_func
        return ParserAdapter(parse_func), "stub"


def _extract_gold_text_from_omnidocbench(page: dict) -> str:
    """
    Extract gold text from OmniDocBench page annotations.

    Args:
        page: OmniDocBench page dict with layout_dets.

    Returns:
        Concatenated text from all layout_dets in reading order.

    """
    layout_dets = page.get("layout_dets", [])

    # Sort by order field, handling None values
    def sort_key(x):
        order = x.get("order")
        if order is None:
            return float("inf")
        return order

    sorted_dets = sorted(layout_dets, key=sort_key)

    # Extract text, maintaining order
    texts = []
    for det in sorted_dets:
        text = det.get("text", "")
        if text:
            texts.append(text)

    return " ".join(texts)


def _get_pdf_path_for_page(page: dict, dataset_root: Path) -> Path:
    """
    Get image/PDF path for an OmniDocBench page.

    OmniDocBench uses PNG images, not PDFs.

    Args:
        page: OmniDocBench page dict.
        dataset_root: Root path of dataset.

    Returns:
        Path to the image file.

    """
    # OmniDocBench structure: root/images/{filename}
    page_info = page.get("page_info", {})
    image_name = page_info.get("image_path", "")

    # Try images directory
    image_path = dataset_root / "images" / image_name

    return image_path


def main() -> None:
    """Run the parsing evaluation CLI."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate document parsing on public benchmarks"
    )
    parser.add_argument(
        "--dataset",
        choices=["omnidocbench", "dp_bench"],
        required=True,
        help="Dataset to evaluate on",
    )
    parser.add_argument(
        "--parser",
        default="stub",
        choices=["stub", "fast", "docling"],
        help=(
            "Parser to use (fast=pypdf for digital PDFs, "
            "docling=full parsing with OCR)"
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
        default=Path("results"),
        help="Output directory for CSV results",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of items to process (for testing)",
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
    print(f"Loading dataset: {args.dataset}")
    dataset = load_dataset(args.dataset, config)

    # Get parser
    print(f"Using parser: {args.parser}")
    parser_adapter, parser_type = get_parser(args.parser)

    # Setup output file for incremental writes with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{args.dataset}_{parser_type}_results_{timestamp}.csv"
    output_file = args.output_dir / filename
    file_exists = output_file.exists()

    # Define all CSV columns
    fieldnames = [
        "query_id",
        "error",
        "nid", "nid_s", "teds", "teds_s", "mhs", "mhs_s", "ard", "bleu", "meteor",
    ]

    # Open CSV for incremental appending
    csv_file = open(output_file, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    # Write header if new file
    if not file_exists:
        writer.writeheader()

    processed = 0
    errors = 0
    examined = 0  # Count all items looked at (for OCR limit)

    for idx, item in enumerate(dataset):
        if args.limit and examined >= args.limit:
            break
        examined += 1

        query_id = f"{args.dataset}_{idx}"

        try:
            if args.dataset == "omnidocbench":
                # Extract gold text from OmniDocBench annotations
                gold_text = _extract_gold_text_from_omnidocbench(item)

                if not gold_text:
                    # Skip pages without text (count toward limit to avoid infinite OCR)
                    continue

                # Get PDF path
                dataset_root = Path(config["datasets"]["omnidocbench"]["path"])
                pdf_path = _get_pdf_path_for_page(item, dataset_root)

                if not pdf_path.exists():
                    writer.writerow(
                        {
                            "query_id": query_id,
                            "error": f"PDF not found: {pdf_path}",
                            "nid": 0.0,
                            "nid_s": 0.0,
                            "teds": 0.0,
                            "teds_s": 0.0,
                            "mhs": 0.0,
                            "mhs_s": 0.0,
                            "ard": 0.0,
                            "bleu": 0.0,
                            "meteor": 0.0,
                        }
                    )
                    csv_file.flush()
                    errors += 1
                    continue

                print(f"Processing page {idx + 1}...")

                # Parse document
                output = parser_adapter.parse(pdf_path)

                # Convert parser output to markdown for comparison
                pred_markdown = parser_output_to_markdown(output)

                # For OmniDocBench, gold_text is just concatenated text
                gt_markdown = gold_text

                # Calculate all metrics
                nid, nid_s = evaluate_nid(gt_markdown, pred_markdown)
                teds, teds_s = evaluate_table(gt_markdown, pred_markdown)
                mhs, mhs_s = evaluate_heading_level(gt_markdown, pred_markdown)
                # ARD uses token lists
                ard = ard_score(gt_markdown.split(), pred_markdown.split())
                bleu = bleu_score(gt_markdown, pred_markdown)
                meteor = meteor_score(gt_markdown, pred_markdown)

                # Convert None to 0.0
                def safe_float(x):
                    return round(x, 4) if x is not None else 0.0

                writer.writerow({
                    "query_id": query_id,
                    "error": "",
                    "nid": safe_float(nid),
                    "nid_s": safe_float(nid_s),
                    "teds": safe_float(teds),
                    "teds_s": safe_float(teds_s),
                    "mhs": safe_float(mhs),
                    "mhs_s": safe_float(mhs_s),
                    "ard": safe_float(ard),
                    "bleu": safe_float(bleu),
                    "meteor": safe_float(meteor),
                })
                csv_file.flush()
                processed += 1

            elif args.dataset == "dp_bench":
                doc_id, pdf_path, gold_elements = item
                print(f"Processing document {doc_id}...")

                # Parse document
                output = parser_adapter.parse(pdf_path)

                # Convert parser output to markdown for comparison
                pred_markdown = parser_output_to_markdown(output)

                # Build ground truth markdown from DP-Bench elements
                gt_lines = []
                for elem in gold_elements.get("elements", []):
                    category = elem.get("category", "")
                    text = elem.get("content", {}).get("text", "")

                    if not text:
                        continue

                    if category == "Header":
                        gt_lines.append(f"# {text}")
                    elif category == "Paragraph":
                        gt_lines.append(text)
                    elif category == "Table":
                        gt_lines.append(f"[TABLE: {text}]")
                    elif category == "List":
                        gt_lines.append(f"- {text}")
                    else:
                        gt_lines.append(text)
                    gt_lines.append("")  # Blank line between elements

                gt_markdown = "\n".join(gt_lines)

                # Calculate all metrics
                nid, nid_s = evaluate_nid(gt_markdown, pred_markdown)
                teds, teds_s = evaluate_table(gt_markdown, pred_markdown)
                mhs, mhs_s = evaluate_heading_level(gt_markdown, pred_markdown)
                ard = ard_score(gt_markdown.split(), pred_markdown.split())
                bleu = bleu_score(gt_markdown, pred_markdown)
                meteor = meteor_score(gt_markdown, pred_markdown)

                # Convert None to 0.0
                def safe_float(x):
                    return round(x, 4) if x is not None else 0.0

                writer.writerow({
                    "query_id": doc_id,
                    "error": "",
                    "nid": safe_float(nid),
                    "nid_s": safe_float(nid_s),
                    "teds": safe_float(teds),
                    "teds_s": safe_float(teds_s),
                    "mhs": safe_float(mhs),
                    "mhs_s": safe_float(mhs_s),
                    "ard": safe_float(ard),
                    "bleu": safe_float(bleu),
                    "meteor": safe_float(meteor),
                })
                csv_file.flush()
                processed += 1

        except Exception as e:
            writer.writerow({
                "query_id": query_id,
                "error": str(e),
                "nid": 0.0, "nid_s": 0.0, "teds": 0.0, "teds_s": 0.0,
                "mhs": 0.0, "mhs_s": 0.0, "ard": 0.0, "bleu": 0.0, "meteor": 0.0,
            })
            csv_file.flush()
            errors += 1

    # Close CSV file
    csv_file.close()

    # Print summary
    print(f"\nResults written to: {output_file}")
    print(f"Total items processed: {processed}")
    print(f"Errors: {errors}")

    # Calculate metric averages (excluding error rows) - reload CSV
    df = pd.read_csv(output_file)
    metrics = [
        "nid",
        "nid_s",
        "teds",
        "teds_s",
        "mhs",
        "mhs_s",
        "ard",
        "bleu",
        "meteor",
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
        "parser": parser_type,
        "timestamp": timestamp,
        "csv_file": str(output_file.name),
        "metrics_avg": averages,
        "total_processed": processed,
        "errors": errors,
    }
    with open(json_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to: {json_file}")

    sys.exit(0)


if __name__ == "__main__":
    main()
