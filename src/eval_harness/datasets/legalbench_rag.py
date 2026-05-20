"""
LegalBench-RAG dataset loader.

This module loads the LegalBench-RAG dataset for retrieval evaluation.
Supports both mini (776 queries) and full (6,889 queries) slices.

The actual LegalBench-RAG structure:
    root/
      data/
        benchmarks/
          cuad.json
          contractnli.json
          maud.json
          privacy_qa.json
        corpus/
          cuad/
          contractnli/
          maud/
          privacy_qa/
"""

import json
from collections.abc import Iterator
from pathlib import Path

# Map directory names to benchmark file names
BENCHMARK_FILES = {
    "cuad": "cuad.json",
    "contract_nli": "contractnli.json",
    "maud": "maud.json",
    "privacyqa": "privacy_qa.json",
}


def load_legalbench_rag(
    root: Path,
    slice: str = "mini",
) -> Iterator[tuple[str, str, list, str]]:
    """
    Load LegalBench-RAG dataset and yield query tuples.

    Args:
        root: Path to LegalBench root directory (e.g., data/rag/legalbench/).
        slice: Either "nano" (50 queries), "mini" (776 queries across 4 sub-corpora),
            or "full" (6,889 queries). Default: "mini".

    Yields:
        tuple: (query_id, query_text, gold_spans, gold_answer_text) where:
            - query_id: Unique query identifier
            - query_text: The question text
            - gold_spans: List of [start, end] character spans in source doc
            - gold_answer_text: The reference answer text

    Raises:
        FileNotFoundError: If root directory doesn't exist.
        ValueError: If slice is not "nano", "mini" or "full".

    """
    if not root.exists():
        raise FileNotFoundError(f"LegalBench-RAG directory not found: {root}")

    if slice not in ("nano", "mini", "full"):
        raise ValueError(f"slice must be 'nano', 'mini' or 'full', got: {slice}")

    # Actual structure: root/data/benchmarks/*.json
    benchmarks_dir = root / "data" / "benchmarks"

    if not benchmarks_dir.exists():
        raise FileNotFoundError(f"Benchmarks directory not found: {benchmarks_dir}")

    query_count = 0

    for corpus_name, benchmark_file in BENCHMARK_FILES.items():
        benchmark_path = benchmarks_dir / benchmark_file

        if not benchmark_path.exists():
            continue

        with open(benchmark_path) as f:
            data = json.load(f)

        for idx, test in enumerate(data.get("tests", [])):
            query_text = test.get("query", "")
            snippets = test.get("snippets", [])

            # Collect all gold spans and answers from snippets
            gold_spans = []
            gold_answers = []

            for snippet in snippets:
                span = snippet.get("span", [])
                if span and len(span) == 2:
                    gold_spans.append(span)

                answer = snippet.get("answer", "")
                if answer:
                    gold_answers.append(answer)

            # Create query ID
            query_id = f"{corpus_name}_{idx}"

            # Combine all answers with separator
            gold_answer_text = " [SEP] ".join(gold_answers) if gold_answers else ""

            yield (query_id, query_text, gold_spans, gold_answer_text)
            query_count += 1

            # Slice limits: nano=12 per corpus (48 total), mini=194 per corpus (776 total)
            queries_per_corpus = {"nano": 12, "mini": 194}.get(slice, None)

            if queries_per_corpus and query_count >= queries_per_corpus * (
                list(BENCHMARK_FILES.keys()).index(corpus_name) + 1
            ):
                break
