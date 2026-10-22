"""
Dataset Splitting Utility (utils/dataset_splitter.py)
Prepares reproducible, stratified, leak-free train/val/test splits for Phase 3 CNN training.
Keeps source dataset untouched and generates machine-readable split manifests under reports/.
"""

import hashlib
import json
import logging
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def compute_file_hash(filepath: Path) -> str:
    """Computes SHA-256 hash to prevent identical image leakage across splits."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def prepare_dataset_split(
    dataset_path: Path | str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
    output_dir: Optional[Path | str] = None,
    dry_run: bool = True,
    output_report_path: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """
    Performs reproducible stratified partitioning of class-organized foliar images.

    Args:
        dataset_path: Path to dataset source (e.g. dataset/PlantVillage).
        train_ratio: Proportion of images allocated to training (default 0.70).
        val_ratio: Proportion allocated to validation (default 0.15).
        test_ratio: Proportion allocated to test (default 0.15).
        random_seed: Deterministic RNG seed for reproducibility (default 42).
        output_dir: Target directory if performing physical split (e.g. dataset/split).
        dry_run: If True, computes manifest and summary without copying or moving files.
        output_report_path: Path for saving summary JSON report.

    Returns:
        Structured dictionary summarizing split counts and manifest.
    """
    dataset_path = Path(dataset_path)

    # Validate ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-4:
        raise ValueError(
            f"Split ratios must sum to 1.0 (got train={train_ratio}, val={val_ratio}, test={test_ratio}, sum={total_ratio})"
        )

    summary: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "dataset_path": str(dataset_path.resolve()),
        "ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        "random_seed": random_seed,
        "dry_run": dry_run,
        "classes_processed": 0,
        "total_images": 0,
        "subsets": {"train": 0, "val": 0, "test": 0},
        "per_class_summary": {},
        "warnings": [],
        "manifest": {"train": [], "val": [], "test": []},
    }

    if not dataset_path.exists() or not dataset_path.is_dir():
        summary["warnings"].append(f"Dataset directory '{dataset_path}' does not exist.")
        _save_split_summary(summary, output_report_path)
        return summary

    class_dirs = [d for d in dataset_path.iterdir() if d.is_dir() and not d.name.startswith(".")]
    if not class_dirs:
        summary["warnings"].append(f"No class folders found in '{dataset_path}'.")
        _save_split_summary(summary, output_report_path)
        return summary

    # Set deterministic random seed
    rng = random.Random(random_seed)

    # Track seen hashes across the whole dataset to prevent data leakage
    seen_hashes: Dict[str, str] = {}  # hash -> first seen path

    for cdir in sorted(class_dirs, key=lambda x: x.name):
        class_name = cdir.name
        raw_files = [
            f for f in cdir.iterdir()
            if f.is_file() and not f.name.startswith(".") and f.suffix.lower() in (".jpg", ".jpeg", ".png")
        ]

        # Deduplicate within and across subsets to prevent data leakage
        unique_files = []
        for fpath in raw_files:
            fhash = compute_file_hash(fpath)
            if fhash in seen_hashes:
                summary["warnings"].append(
                    f"Data leakage guard: Omitting duplicate image '{fpath.name}' in '{class_name}'. "
                    f"Identical content already registered at '{seen_hashes[fhash]}'."
                )
                continue
            seen_hashes[fhash] = f"{class_name}/{fpath.name}"
            unique_files.append(fpath)

        n_samples = len(unique_files)
        if n_samples < 3:
            summary["warnings"].append(
                f"Class '{class_name}' has only {n_samples} unique image(s). Cannot partition into 3 distinct subsets."
            )

        # Shuffle deterministically
        rng.shuffle(unique_files)

        # Calculate partition indices
        n_train = int(round(n_samples * train_ratio))
        n_val = int(round(n_samples * val_ratio))

        # Ensure at least 1 sample in val and test if n_samples >= 3
        if n_samples >= 3:
            n_val = max(1, n_val)
            n_test = n_samples - n_train - n_val
            if n_test < 1:
                n_train -= 1
                n_test = 1
        else:
            n_test = n_samples - n_train - n_val

        train_files = unique_files[:n_train]
        val_files = unique_files[n_train:n_train + n_val]
        test_files = unique_files[n_train + n_val:]

        summary["classes_processed"] += 1
        summary["total_images"] += n_samples
        summary["subsets"]["train"] += len(train_files)
        summary["subsets"]["val"] += len(val_files)
        summary["subsets"]["test"] += len(test_files)

        summary["per_class_summary"][class_name] = {
            "total": n_samples,
            "train": len(train_files),
            "val": len(val_files),
            "test": len(test_files),
        }

        # Build manifest
        for tf in train_files:
            rel = f"{class_name}/{tf.name}"
            summary["manifest"]["train"].append(rel)
        for vf in val_files:
            rel = f"{class_name}/{vf.name}"
            summary["manifest"]["val"].append(rel)
        for tsf in test_files:
            rel = f"{class_name}/{tsf.name}"
            summary["manifest"]["test"].append(rel)

        # If not dry run and output_dir specified, copy files safely
        if not dry_run and output_dir:
            out_root = Path(output_dir)
            for subset_name, flist in [("train", train_files), ("val", val_files), ("test", test_files)]:
                target_class_dir = out_root / subset_name / class_name
                target_class_dir.mkdir(parents=True, exist_ok=True)
                for f in flist:
                    dest = target_class_dir / f.name
                    if not dest.exists():
                        shutil.copy2(f, dest)

    _save_split_summary(summary, output_report_path)
    return summary


def _save_split_summary(summary: Dict[str, Any], output_path: Optional[Path | str]) -> None:
    """Saves split summary to reports/."""
    if output_path is None:
        from config import Config
        reports_dir = Path(Config.REPORTS_DIR)
        reports_dir.mkdir(parents=True, exist_ok=True)
        output_path = reports_dir / "dataset_split_summary.json"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info("Dataset split summary saved to %s", output_path)
    except Exception as err:
        logger.error("Failed to write split summary: %s", err)
