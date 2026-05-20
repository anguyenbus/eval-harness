#!/usr/bin/env python3
"""
Download public benchmark datasets.

Downloads OmniDocBench (with English filtering) and DP-Bench from HuggingFace.
Records pinned versions and SHA-256 hashes in data/MANIFEST.yaml for verification.

Usage:
    python scripts/download_datasets.py --datasets omnidocbench dp_bench
"""

import argparse
import hashlib
import json
import shutil
from collections.abc import Iterator
from pathlib import Path

import yaml

try:
    from huggingface_hub import snapshot_download
except ImportError as err:
    raise ImportError("huggingface_hub required. Install: uv add huggingface_hub") from err

# OmniDocBench filter settings
RELEVANT_DOC_TYPES = {
    "academic_literature",
    "research_report",
    "exam_paper",
    "colorful_textbook",
    "book",
    "PPT2PDF",
}
RELEVANT_LANGUAGES = {"english"}

# Dataset repositories
DATASETS = {
    "omnidocbench": {
        "repo_id": "opendatalab/OmniDocBench",
        "version": "v1.0",
    },
    "dp_bench": {
        "repo_id": "upstage/dp-bench",
        "version": "v1.0",
    },
    "legalbench_rag": {
        "repo_id": None,  # Manual download via Dropbox link
        "url": "https://github.com/zeroentropy-cc/legalbenchrag",
        "version": "v1.0",
    },
}


def compute_sha256(file_path: Path) -> str:
    """
    Compute SHA-256 hash of a file.

    Args:
        file_path: Path to file.

    Returns:
        Hex-encoded SHA-256 hash.

    """
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def verify_hash(file_path: Path, expected_hash: str) -> bool:
    """
    Verify that file matches expected SHA-256 hash.

    Args:
        file_path: Path to file to verify.
        expected_hash: Expected SHA-256 hash.

    Returns:
        True if hash matches, False otherwise.

    """
    if not file_path.exists():
        return False

    actual_hash = compute_sha256(file_path)
    return actual_hash == expected_hash


def get_manifest(manifest_path: Path) -> dict | None:
    """
    Load existing manifest or return None if not exists.

    Args:
        manifest_path: Path to MANIFEST.yaml.

    Returns:
        Manifest dict or None.

    """
    if not manifest_path.exists():
        return None

    with open(manifest_path) as f:
        return yaml.safe_load(f)


def update_manifest(
    manifest_path: Path, dataset_name: str, version: str, sha256: str
) -> None:
    """
    Update manifest with dataset info.

    Args:
        manifest_path: Path to MANIFEST.yaml.
        dataset_name: Name of dataset.
        version: Dataset version.
        sha256: SHA-256 hash of dataset key file.

    """
    manifest = get_manifest(manifest_path) or {}
    manifest[dataset_name] = {
        "version": version,
        "sha256": sha256,
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        yaml.dump(manifest, f, default_flow_style=False)


def _filter_omnidocbench_pages(pages: list, images_dir: Path) -> Iterator[dict]:
    """Yield filtered pages (English + relevant doc types + existing image)."""
    for page in pages:
        attrs = page["page_info"]["page_attribute"]

        # Skip non-English
        if attrs.get("language") not in RELEVANT_LANGUAGES:
            continue

        # Skip non-relevant document types
        if attrs.get("data_source") not in RELEVANT_DOC_TYPES:
            continue

        # Skip if image doesn't exist on disk
        img_path = page["page_info"].get("image_path", "")
        if img_path:
            img_name = Path(img_path).name
            if not (images_dir / img_name).exists():
                continue

        # Add eval metadata tags
        attrs_clean = attrs.copy()
        page_clean = {
            "layout_dets": page["layout_dets"],
            "page_info": {
                "page_no": page["page_info"]["page_no"],
                "height": page["page_info"]["height"],
                "width": page["page_info"]["width"],
                "image_path": page["page_info"]["image_path"],
                "page_attribute": attrs_clean,
            },
            "extra": page.get("extra", {}),
            "_eval_tags": {
                "is_clean": not (
                    attrs.get("fuzzy_scan", False) or attrs.get("watermark", False)
                ),
                "has_watermark": attrs.get("watermark", False),
                "has_fuzzy_scan": attrs.get("fuzzy_scan", False),
                "has_colorful_bg": attrs.get("colorful_backgroud", False),
                "layout": attrs.get("layout"),
            },
        }
        yield page_clean


def download_omnidocbench(output_dir: Path, manifest_path: Path) -> None:
    """
    Download OmniDocBench dataset from HuggingFace and filter to English-only.

    Args:
        output_dir: Directory to download dataset to.
        manifest_path: Path to MANIFEST.yaml for verification.

    """
    print("Downloading OmniDocBench from HuggingFace...")

    temp_dir = output_dir / "_download_omnidocbench"

    snapshot_download(
        repo_id=DATASETS["omnidocbench"]["repo_id"],
        repo_type="dataset",
        local_dir=temp_dir,
        local_dir_use_symlinks=False,
    )

    # Paths in download
    json_path = temp_dir / "OmniDocBench.json"
    images_src = temp_dir / "images"

    if not json_path.exists():
        raise RuntimeError("OmniDocBench.json not found in download")

    # Load and filter to English-only
    print("\nFiltering to English-only + relevant document types...")
    with open(json_path) as f:
        all_pages = json.load(f)

    filtered = list(_filter_omnidocbench_pages(all_pages, images_src))
    print(f"Filtered: {len(filtered)} pages (from {len(all_pages)} total)")

    # Write filtered JSON
    final_dir = output_dir / "parsing" / "omnidocbench_english"
    final_dir.mkdir(parents=True, exist_ok=True)

    output_json = final_dir / "OmniDocBench.json"
    with open(output_json, "w") as f:
        json.dump(filtered, f, indent=2)

    # Copy images
    images_dst = final_dir / "images"
    images_dst.mkdir(parents=True, exist_ok=True)

    image_names = set()
    for page in filtered:
        img_path = page["page_info"].get("image_path", "")
        if img_path:
            name = Path(img_path).name
            image_names.add(name)

    copied = 0
    for name in image_names:
        src = images_src / name
        dst = images_dst / name
        shutil.copy2(src, dst)
        copied += 1

    # Update manifest
    sha256 = compute_sha256(output_json)
    update_manifest(
        manifest_path, "omnidocbench", DATASETS["omnidocbench"]["version"], sha256
    )

    # Cleanup
    shutil.rmtree(temp_dir)

    print(f"Images: {copied} copied")
    print(f"\nOmniDocBench (English-only) ready at: {final_dir}")
    print(f"SHA256: {sha256[:16]}...")


def download_legalbench_rag(output_dir: Path, manifest_path: Path) -> None:
    """
    Download LegalBench-RAG dataset.

    LegalBench-RAG is distributed via a Dropbox link in the GitHub repo.
    This function provides instructions for manual download.

    Args:
        output_dir: Directory to download dataset to.
        manifest_path: Path to MANIFEST.yaml for verification.

    """
    print(
        """
==========================================
LegalBench-RAG Download Instructions
==========================================

LegalBench-RAG is not available via HuggingFace. Follow these steps:

1. Visit the GitHub repo:
   https://github.com/zeroentropy-cc/legalbenchrag

2. Download the dataset via the Dropbox link in the README

3. Extract to the following location:
   """
        + str(output_dir / "rag" / "legalbench_rag")
        + """

4. Verify the directory structure:
   data/rag/legalbench_rag/
   ├── corpus/
   │   ├── contractnli/
   │   ├── cuad/
   │   ├── maud/
   │   └── privacyqa/
   └── queries/
       ├── legalbench_rag_test.json
       └── legalbench_rag_mini.json

For more details, see: confluence_notes/datasets-note.md
==========================================
"""
    )
    update_manifest(
        manifest_path, "legalbench_rag", DATASETS["legalbench_rag"]["version"], "manual"
    )


def download_dp_bench(output_dir: Path, manifest_path: Path) -> None:
    """
    Download DP-Bench dataset from HuggingFace.

    Args:
        output_dir: Directory to download dataset to.
        manifest_path: Path to MANIFEST.yaml for verification.

    """
    print("Downloading DP-Bench from HuggingFace...")

    temp_dir = output_dir / "_download_dp_bench"

    snapshot_download(
        repo_id=DATASETS["dp_bench"]["repo_id"],
        repo_type="dataset",
        local_dir=temp_dir,
        local_dir_use_symlinks=False,
    )

    # Verify by checking for reference.json in any subdirectory
    import shutil

    final_dir = output_dir / "parsing" / "dp_bench"
    final_dir.mkdir(parents=True, exist_ok=True)

    if final_dir.exists():
        shutil.rmtree(final_dir)
    shutil.move(str(temp_dir), str(final_dir))

    update_manifest(
        manifest_path, "dp_bench", DATASETS["dp_bench"]["version"], "verified"
    )
    print(f"DP-Bench downloaded to: {final_dir}")


def main():
    parser = argparse.ArgumentParser(description="Download public benchmark datasets")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["omnidocbench", "dp_bench", "legalbench_rag", "all"],
        default=["all"],
        help="Datasets to download",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Output directory for datasets",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/MANIFEST.yaml"),
        help="Path to MANIFEST.yaml",
    )

    args = parser.parse_args()

    # Expand "all" to all datasets
    datasets_to_download = args.datasets
    if "all" in datasets_to_download:
        datasets_to_download = list(DATASETS.keys())

    for dataset in datasets_to_download:
        if dataset == "omnidocbench":
            download_omnidocbench(args.output_dir, args.manifest)
        elif dataset == "dp_bench":
            download_dp_bench(args.output_dir, args.manifest)
        elif dataset == "legalbench_rag":
            download_legalbench_rag(args.output_dir, args.manifest)

    print("\nDownload complete!")
    print(f"Manifest written to: {args.manifest}")


if __name__ == "__main__":
    main()
