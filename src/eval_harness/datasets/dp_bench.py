"""
DP-Bench dataset loader.

This module loads the DP-Bench (Document Parsing Benchmark) dataset,
which provides ground truth layout annotations for document parsing evaluation.
"""

import json
from collections.abc import Iterator
from pathlib import Path


def load_dp_bench(root: Path) -> Iterator[tuple[str, Path, dict]]:
    """
    Load DP-Bench dataset and yield (doc_id, pdf_path, gold_elements) tuples.

    The DP-Bench dataset structure from HuggingFace:
        root/
          dataset/
            pdfs/
              01030000000001.pdf
              01030000000002.pdf
              ...
            reference.json  # Single JSON with all annotations

    Args:
        root: Path to DP-Bench root directory.
            Default would be: data/parsing/dp_bench/

    Yields:
        tuple: (doc_id, pdf_path, gold_elements) where:
            - doc_id: Document identifier (PDF filename without extension)
            - pdf_path: Path to the PDF file
            - gold_elements: Ground truth from reference.json

    Raises:
        FileNotFoundError: If root directory or reference.json doesn't exist.

    """
    dataset_dir = root / "dataset"
    reference_path = dataset_dir / "reference.json"
    pdfs_dir = dataset_dir / "pdfs"

    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"DP-Bench dataset directory not found: {dataset_dir}"
        )

    if not reference_path.exists():
        raise FileNotFoundError(
            f"DP-Bench reference.json not found: {reference_path}"
        )

    if not pdfs_dir.exists():
        raise FileNotFoundError(
            f"DP-Bench pdfs directory not found: {pdfs_dir}"
        )

    # Load reference annotations
    with open(reference_path) as f:
        reference = json.load(f)

    # Iterate over each document in reference
    for pdf_filename, gold_elements in reference.items():
        doc_id = pdf_filename.replace(".pdf", "")
        pdf_path = pdfs_dir / pdf_filename

        if not pdf_path.exists():
            # Skip if PDF file missing
            continue

        yield doc_id, pdf_path, gold_elements
