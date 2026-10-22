"""
PlantVillage Dataset Inspection Utility (utils/dataset_inspector.py)
Inspects class-organized foliar disease image directories.
Checks for missing folders, image extensions, corrupt files, duplicates (SHA-256),
and class imbalance, generating a structured JSON report under reports/.
"""

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)

PLANTVILLAGE_SOURCE_INFO = {
    "name": "PlantVillage Dataset",
    "description": (
        "Open-access repository containing over 54,000 expertly curated leaf images "
        "across healthy and diseased crops (e.g., Tomato, Potato, Pepper, Corn, Apple)."
    ),
    "repository_urls": [
        "https://github.com/spMohanty/PlantVillage-Dataset",
        "https://www.kaggle.com/datasets/emmarex/plantdisease",
        "https://data.mendeley.com/datasets/tywbtsjrjv/1",
    ],
}


def compute_file_hash(filepath: Path) -> str:
    """Calculates SHA-256 hash of a file's contents for exact duplicate detection."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def inspect_dataset(
    dataset_path: Path | str,
    output_report_path: Optional[Path | str] = None,
    min_samples_threshold: int = 15,
) -> Dict[str, Any]:
    """
    Analyzes a class-organized plant disease dataset.

    Args:
        dataset_path: Path to dataset root (e.g., dataset/PlantVillage).
        output_report_path: Optional path to write JSON report (default reports/dataset_inspection_report.json).
        min_samples_threshold: Minimum recommended images per class for 70/15/15 splitting.

    Returns:
        Structured dictionary containing dataset metrics.
    """
    dataset_path = Path(dataset_path)

    report: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "dataset_path": str(dataset_path.resolve()),
        "status": "not_found",
        "plantvillage_reference": PLANTVILLAGE_SOURCE_INFO,
        "total_classes": 0,
        "total_images": 0,
        "classes": {},
        "extension_counts": {},
        "corrupt_files": [],
        "duplicate_groups": [],
        "warnings": [],
        "class_imbalance": {},
    }

    if not dataset_path.exists():
        report["warnings"].append(
            f"Dataset directory '{dataset_path}' does not exist. "
            "Please download the PlantVillage dataset from the referenced sources."
        )
        _save_report(report, output_report_path)
        return report

    if not dataset_path.is_dir():
        report["warnings"].append(f"Specified path '{dataset_path}' is not a directory.")
        _save_report(report, output_report_path)
        return report

    # Find class folders (subdirectories containing images)
    class_dirs = [d for d in dataset_path.iterdir() if d.is_dir() and not d.name.startswith(".")]

    if not class_dirs:
        report["status"] = "empty"
        report["warnings"].append(
            f"Directory '{dataset_path}' is empty or contains no class subfolders. "
            "Expected structure: dataset/PlantVillage/<Crop___Disease_Class>/<images>."
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

            # Pillow integrity check
            try:
                with Image.open(fpath) as img:
                    img.verify()
                valid_count += 1
                class_images.append(fpath.name)

                # Compute hash for duplicate tracking
                file_hash = compute_file_hash(fpath)
                hash_to_files.setdefault(file_hash, []).append(f"{class_name}/{fpath.name}")

            except (UnidentifiedImageError, OSError, SyntaxError) as err:
                corrupt_entry = {
                    "file": str(fpath.relative_to(dataset_path)),
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

        # Check threshold
        if valid_count < min_samples_threshold:
            report["warnings"].append(
                f"Class '{class_name}' contains only {valid_count} valid images (below recommended threshold of {min_samples_threshold})."
            )

    report["total_images"] = sum(class_counts.values())

    # Filter duplicate groups (hashes with > 1 file)
    duplicate_groups = [
        {"hash": h, "files": paths}
        for h, paths in hash_to_files.items()
        if len(paths) > 1
    ]
    report["duplicate_groups"] = duplicate_groups
    if duplicate_groups:
        report["warnings"].append(
            f"Detected {len(duplicate_groups)} exact duplicate file group(s) across dataset."
        )

    # Class imbalance calculation
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


def _save_report(report: Dict[str, Any], output_path: Optional[Path | str]) -> None:
    """Saves inspection dictionary as a JSON file under reports/."""
    if output_path is None:
        from config import Config
        reports_dir = Path(Config.REPORTS_DIR)
        reports_dir.mkdir(parents=True, exist_ok=True)
        output_path = reports_dir / "dataset_inspection_report.json"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info("Dataset inspection report saved to %s", output_path)
    except Exception as err:
        logger.error("Failed to write inspection report: %s", err)
