#!/usr/bin/env python3
"""
Download and filter OmniDocBench to English-only, RFI-relevant document types.

End-to-end script:
1. Downloads OmniDocBench from HuggingFace (if not already downloaded)
2. Filters to English-only with relevant document types
3. Outputs filtered JSON and images

Usage:
    python filter_omnidocbench.py --output-dir data/parsing/omnidocbench_english
"""

import argparse
import json
import shutil
from collections.abc import Iterator
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
except ImportError as err:
    raise ImportError("huggingface_hub required. Install: pip install huggingface_hub") from err

# HuggingFace dataset info
HF_REPO = "opendatalab/OmniDocBench"

# Document types relevant to RFI workflows
RELEVANT_DOC_TYPES = {
    "academic_literature",
    "research_report",
    "exam_paper",
    "colorful_textbook",
    "book",
    "PPT2PDF",
}

# Languages we keep
RELEVANT_LANGUAGES = {"english"}


def download_omnidocbench(output_dir: Path) -> Path:
    """Download OmniDocBench from HuggingFace."""
    print(f"Downloading {HF_REPO} from HuggingFace...")

    # Create temp download directory
    temp_dir = output_dir / "_download"
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Download using huggingface_hub
    snapshot_download(
        repo_id=HF_REPO,
        repo_type="dataset",
        local_dir=temp_dir,
        local_dir_use_symlinks=False,
    )

    print(f"Downloaded to: {temp_dir}")
    return temp_dir


def load_omnidocbench(json_path: Path) -> list[dict]:
    """Load full OmniDocBench JSON."""
    with open(json_path) as f:
        return json.load(f)


def filter_pages(pages: list[dict], images_dir: Path) -> Iterator[dict]:
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


def copy_images(pages: list[dict], src_dir: Path, dst_dir: Path) -> int:
    """Copy images referenced in filtered pages."""
    dst_dir.mkdir(parents=True, exist_ok=True)

    image_names = set()
    for page in pages:
        img_path = page["page_info"].get("image_path", "")
        if img_path:
            name = Path(img_path).name
            image_names.add(name)

    copied = 0
    for name in image_names:
        src = src_dir / name
        dst = dst_dir / name
        shutil.copy2(src, dst)
        copied += 1

    return copied


def main():
    parser = argparse.ArgumentParser(
        description="Download and filter OmniDocBench to English-only data"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Output directory for filtered data",
    )
    parser.add_argument(
        "--keep-download",
        action="store_true",
        help="Keep the full download instead of deleting it",
    )
    args = parser.parse_args()

    # Step 1: Download full dataset
    temp_dir = download_omnidocbench(args.output_dir)

    # Paths in download
    json_path = temp_dir / "OmniDocBench.json"
    images_src = temp_dir / "images"

    # Step 2: Load and filter
    print("\nFiltering to English-only...")
    pages = load_omnidocbench(json_path)
    filtered = list(filter_pages(pages, images_src))

    # Step 3: Write filtered JSON
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_json = args.output_dir / "OmniDocBench.json"
    with open(output_json, "w") as f:
        json.dump(filtered, f, indent=2)

    print(
        f"Filtered: {len(filtered)} pages with existing images (from {len(pages)} total)"
    )
    print(f"JSON output: {output_json}")

    # Step 4: Copy images
    images_dst = args.output_dir / "images"
    copied = copy_images(filtered, images_src, images_dst)
    print(f"Images: {copied} copied")
    print(f"Images output: {images_dst}")

    # Step 5: Cleanup
    if not args.keep_download:
        print(f"\nCleaning up download directory: {temp_dir}")
        shutil.rmtree(temp_dir)
    else:
        print(f"\nKeeping download directory: {temp_dir}")

    print(f"\nDone! Filtered dataset ready at: {args.output_dir}")


if __name__ == "__main__":
    main()
