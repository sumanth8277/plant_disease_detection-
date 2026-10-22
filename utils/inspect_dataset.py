"""
Plant Disease Dataset Inspection Utility (utils/inspect_dataset.py)
Audits raw and processed plant disease foliar image datasets.
Detects unreadable images via Pillow, counts class distributions, discovers exact duplicates,
and calculates class imbalance metrics.
Outputs machine-readable and readable summaries into reports/training/.

Usage:
    python utils/inspect_dataset.py --dataset-root dataset/raw
"""

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from PIL import Image, UnidentifiedImageError

# Ensure UTF-8 output on Windows consoles if supported
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure project root is on sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import Config

logger = logging.getLogger(__name__)

PLANTVILLAGE_INFO = {
    "name": "PlantVillage Foliar Dataset",
    "description": (
        "Open-access foliar dataset containing 54,000+ curated images of healthy and diseased crops."
    ),
    "sources": [
        "https://github.com/spMohanty/PlantVillage-Dataset",
        "https://www.kaggle.com/datasets/emmarex/plantdisease",
        "https://data.mendeley.com/datasets/tywbtsjrjv/1",
    ],
}


def compute_file_hash(filepath: Path) -> str:
    """Calculates SHA-256 digest of image bytes to identify duplicates."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def inspect_dataset_directory(
    dataset_root: Optional[Union[str, Path]] = None,
    output_report_path: Optional[Union[str, Path]] = None,
    min_samples_threshold: int = 15,
) -> Dict[str, Any]:
    """
    Recursively inspects a class-organized plant disease dataset.

    Args:
        dataset_root: Root directory containing class subfolders (e.g. dataset/raw).
        output_report_path: Destination JSON path under reports/training/.
        min_samples_threshold: Warning threshold for small classes.

    Returns:
        Structured audit report dict.
    """
    if dataset_root is None:
        dataset_root = Config.BASE_DIR / "dataset" / "raw"
    else:
        dataset_root = Path(dataset_root)

    report: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "dataset_root": str(dataset_root.resolve()),
        "status": "not_found",
        "dataset_provenance": PLANTVILLAGE_INFO,
        "total_classes": 0,
        "total_valid_images": 0,
        "classes": {},
        "extension_counts": {},
        "corrupt_files": [],
        "duplicate_groups": [],
        "warnings": [],
        "class_imbalance": {},
    }

    if not dataset_root.exists():
        report["warnings"].append(
            f"Dataset root directory '{dataset_root}' does not exist. "
            "Please download the PlantVillage dataset and place class folders in dataset/raw/."
        )
        _save_report(report, output_report_path)
        return report

    if not dataset_root.is_dir():
        report["warnings"].append(f"Specified path '{dataset_root}' is not a directory.")
        _save_report(report, output_report_path)
        return report

    # Find class folders (subdirectories containing images)
    class_dirs = [d for d in dataset_root.iterdir() if d.is_dir() and not d.name.startswith(".")]

    if not class_dirs:
        report["status"] = "empty"
        report["warnings"].append(
            f"Directory '{dataset_root}' is empty or contains no class subfolders. "
            "Expected structure: dataset/raw/<Crop___Disease_Class>/<images>."
        )
        _save_report(report, output_report_path)
        return report

    report["status"] = "inspected"
    report["total_classes"] = len(class_dirs)

    hash_to_files: Dict[str, List[str]] = {}
    class_counts: Dict[str, int] = {}

    for cdir in sorted(class_dirs, key=lambda x: x.name):
        class_name = cdir.name
        class_images = []
        corrupt_in_class = []

        image_files = [
            f for f in cdir.iterdir()
            if f.is_file() and not f.name.startswith(".")
        ]

        valid_count = 0
        for fpath in image_files:
            ext = fpath.suffix.lower()
            report["extension_counts"][ext] = report["extension_counts"].get(ext, 0) + 1

            if ext not in (".jpg", ".jpeg", ".png"):
                report["warnings"].append(
                    f"Non-standard image format '{ext}' encountered in '{class_name}/{fpath.name}'."
                )

            # Pillow integrity validation
            try:
                with Image.open(fpath) as img:
                    img.verify()
                valid_count += 1
                class_images.append(fpath.name)

                fhash = compute_file_hash(fpath)
                hash_to_files.setdefault(fhash, []).append(f"{class_name}/{fpath.name}")

            except (UnidentifiedImageError, OSError, SyntaxError) as err:
                corrupt_entry = {
                    "file": f"{class_name}/{fpath.name}",
                    "error": str(err),
                }
                report["corrupt_files"].append(corrupt_entry)
                corrupt_in_class.append(fpath.name)

        class_counts[class_name] = valid_count
        report["classes"][class_name] = {
            "valid_images": valid_count,
            "corrupt_images": len(corrupt_in_class),
            "sample_files": class_images[:5],
        }

        if valid_count < min_samples_threshold:
            report["warnings"].append(
                f"Class '{class_name}' has only {valid_count} valid image(s) "
                f"(below threshold of {min_samples_threshold} for 70/15/15 splitting)."
            )

    report["total_valid_images"] = sum(class_counts.values())

    # Filter duplicate groups
    dup_groups = [
        {"hash": h, "files": paths}
        for h, paths in hash_to_files.items()
        if len(paths) > 1
    ]
    report["duplicate_groups"] = dup_groups
    if dup_groups:
        report["warnings"].append(
            f"Detected {len(dup_groups)} duplicate image group(s) across dataset."
        )

    # Imbalance metrics
    if class_counts:
        counts = list(class_counts.values())
        min_c = min(counts)
        max_c = max(counts)
        mean_c = sum(counts) / len(counts)
        report["class_imbalance"] = {
            "min_class_size": min_c,
            "max_class_size": max_c,
            "mean_class_size": round(mean_c, 2),
            "imbalance_ratio": round(max_c / max(min_c, 1), 2),
        }

    _save_report(report, output_report_path)
    return report


def _save_report(report: Dict[str, Any], output_path: Optional[Union[str, Path]]) -> None:
    """Writes the inspection dictionary to reports/training/."""
    if output_path is None:
        target_dir = Config.BASE_DIR / "reports" / "training"
        target_dir.mkdir(parents=True, exist_ok=True)
        output_path = target_dir / "dataset_inspection_report.json"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Inspect PlantVillage foliar disease dataset for image integrity, class counts, and duplicates."
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=str(Config.BASE_DIR / "dataset" / "raw"),
        help="Root directory of the raw class-organized images (default: dataset/raw)",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=15,
        help="Minimum samples per class threshold (default: 15)",
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default=str(Config.BASE_DIR / "reports" / "training" / "dataset_inspection_report.json"),
        help="Path for saving JSON inspection report",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("PlantVillage Foliar Dataset Inspection Tool (Phase 3)")
    print("=" * 70)
    print(f"Target Directory: {args.dataset_root}")
    print(f"Output Report:    {args.output_report}")
    print("-" * 70)

    report = inspect_dataset_directory(
        dataset_root=args.dataset_root,
        output_report_path=args.output_report,
        min_samples_threshold=args.min_samples,
    )

    status = report.get("status")
    print(f"Status: {status.upper()}")

    if status == "not_found":
        print("\n[NOTICE] The dataset directory does not exist on this machine.")
        print("Please acquire the PlantVillage dataset from legitimate open sources:")
        for url in PLANTVILLAGE_INFO["sources"]:
            print(f"  * {url}")
        print("\nExpected directory organization:")
        print("  dataset/raw/")
        print("    Tomato___Early_blight/")
        print("    Tomato___Late_blight/")
        print("    Tomato___healthy/")
    elif status == "empty":
        print("\n[WARNING] Dataset directory exists but contains no class subfolders.")
    else:
        print(f"\n[OK] Found {report['total_classes']} classes with {report['total_valid_images']} valid images.")
        print("\nClass Distribution Summary:")
        for cname, details in report["classes"].items():
            print(f"  * {cname:<35}: {details['valid_images']} valid ({details['corrupt_images']} corrupt)")

        if report.get("duplicate_groups"):
            print(f"\n[WARNING] Detected {len(report['duplicate_groups'])} exact duplicate file groups.")

        imbalance = report.get("class_imbalance", {})
        if imbalance:
            print(f"\nClass Imbalance Ratio: {imbalance.get('imbalance_ratio', 1.0)}x")
            print(f"  Min Class Size:  {imbalance.get('min_class_size')}")
            print(f"  Max Class Size:  {imbalance.get('max_class_size')}")
            print(f"  Mean Class Size: {imbalance.get('mean_class_size')}")

    if report.get("warnings"):
        print("\nInspection Warnings:")
        for w in report["warnings"]:
            print(f"  [!] {w}")

    print("\n" + "=" * 70)
    print(f"Report written to: {args.output_report}")
    print("=" * 70)


if __name__ == "__main__":
    from typing import Union
    main()
