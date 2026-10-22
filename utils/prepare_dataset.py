"""
Reproducible Dataset Splitting & Manifest Generator (utils/prepare_dataset.py)
Partitions raw plant disease leaf images into leak-free Train (70%), Validation (15%),
and Held-Out Test (15%) subsets.
Preserves raw images unchanged, detects exact duplicate file hashes, and generates
a single source-of-truth split manifest and class mapping dictionary.

Usage:
    python utils/prepare_dataset.py --dataset-root dataset/raw --seed 42
"""

import argparse
import hashlib
import json
import logging
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Configure UTF-8 on Windows consoles if supported
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import Config

logger = logging.getLogger(__name__)


def compute_file_hash(filepath: Path) -> str:
    """Computes SHA-256 hash of file content to prevent identical image data leakage."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def prepare_dataset_manifest(
    dataset_root: Optional[Union[str, Path]] = None,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
    manifest_output_path: Optional[Union[str, Path]] = None,
    class_indices_path: Optional[Union[str, Path]] = None,
    summary_output_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Creates a deterministic, stratified dataset split manifest from raw class directories.
    Leaves original images unchanged.

    Returns:
        Structured summary dictionary.
    """
    if dataset_root is None:
        dataset_root = Config.BASE_DIR / "dataset" / "raw"
    else:
        dataset_root = Path(dataset_root)

    # Validate split ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-4:
        raise ValueError(
            f"Split ratios must sum to 1.0 (got train={train_ratio}, val={val_ratio}, test={test_ratio}, sum={total_ratio})"
        )

    summary: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "dataset_root": str(dataset_root.resolve()),
        "ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        "random_seed": random_seed,
        "classes_discovered": 0,
        "total_unique_images": 0,
        "duplicates_omitted": 0,
        "subsets": {"train": 0, "val": 0, "test": 0},
        "class_indices": {},
        "per_class_counts": {},
        "warnings": [],
        "manifest": {"train": [], "val": [], "test": []},
    }

    if not dataset_root.exists() or not dataset_root.is_dir():
        summary["warnings"].append(
            f"Dataset root directory '{dataset_root}' does not exist. "
            "Please download the dataset and place class subfolders in dataset/raw/."
        )
        _save_json(summary, summary_output_path or (Config.BASE_DIR / "reports" / "training" / "dataset_split_summary.json"))
        return summary

    class_dirs = [d for d in dataset_root.iterdir() if d.is_dir() and not d.name.startswith(".")]

    if not class_dirs:
        summary["warnings"].append(
            f"Dataset directory '{dataset_root}' is empty or contains no class subfolders."
        )
        _save_json(summary, summary_output_path or (Config.BASE_DIR / "reports" / "training" / "dataset_split_summary.json"))
        return summary

    # Sort class names alphabetically to ensure deterministic index assignment
    sorted_class_names = sorted([d.name for d in class_dirs])
    class_to_idx = {name: idx for idx, name in enumerate(sorted_class_names)}
    summary["class_indices"] = class_to_idx
    summary["classes_discovered"] = len(sorted_class_names)

    # Set deterministic random seed
    rng = random.Random(random_seed)

    # Global hash registry to detect duplicates across classes and avoid leakage
    seen_hashes: Dict[str, str] = {}

    for class_name in sorted_class_names:
        cdir = dataset_root / class_name
        image_files = [
            f for f in cdir.iterdir()
            if f.is_file() and not f.name.startswith(".") and f.suffix.lower() in (".jpg", ".jpeg", ".png")
        ]

        class_unique_records = []
        for img_path in sorted(image_files, key=lambda x: x.name):
            fhash = compute_file_hash(img_path)
            rel_path = f"{class_name}/{img_path.name}"

            if fhash in seen_hashes:
                summary["duplicates_omitted"] += 1
                summary["warnings"].append(
                    f"Duplicate omitted to prevent leakage: '{rel_path}' is identical to '{seen_hashes[fhash]}'."
                )
                continue

            seen_hashes[fhash] = rel_path
            class_unique_records.append({
                "relative_path": rel_path,
                "class_name": class_name,
                "class_index": class_to_idx[class_name],
            })

        n_samples = len(class_unique_records)
        if n_samples < 3:
            summary["warnings"].append(
                f"Class '{class_name}' has only {n_samples} unique image(s). "
                "Insufficient samples to guarantee presence across all 3 splits."
            )

        # Shuffle deterministically
        rng.shuffle(class_unique_records)

        # Compute partition boundaries
        n_train = int(round(n_samples * train_ratio))
        n_val = int(round(n_samples * val_ratio))

        if n_samples >= 3:
            n_val = max(1, n_val)
            n_test = n_samples - n_train - n_val
            if n_test < 1:
                n_train -= 1
                n_test = 1
        else:
            n_test = n_samples - n_train - n_val

        train_records = class_unique_records[:n_train]
        val_records = class_unique_records[n_train:n_train + n_val]
        test_records = class_unique_records[n_train + n_val:]

        summary["total_unique_images"] += n_samples
        summary["subsets"]["train"] += len(train_records)
        summary["subsets"]["val"] += len(val_records)
        summary["subsets"]["test"] += len(test_records)

        summary["per_class_counts"][class_name] = {
            "total": n_samples,
            "train": len(train_records),
            "val": len(val_records),
            "test": len(test_records),
        }

        summary["manifest"]["train"].extend(train_records)
        summary["manifest"]["val"].extend(val_records)
        summary["manifest"]["test"].extend(test_records)

    # 1. Save class_indices.json
    indices_file = class_indices_path or (Config.BASE_DIR / "model" / "class_indices.json")
    _save_json(class_to_idx, indices_file)

    # 2. Save manifest JSON
    manifest_file = manifest_output_path or (Config.BASE_DIR / "reports" / "training" / "dataset_manifest.json")
    _save_json(
        {
            "metadata": {
                "timestamp": summary["timestamp"],
                "dataset_root": summary["dataset_root"],
                "random_seed": random_seed,
                "class_indices": class_to_idx,
                "subsets_count": summary["subsets"],
            },
            "manifest": summary["manifest"],
        },
        manifest_file,
    )

    # 3. Save readable summary JSON
    summary_file = summary_output_path or (Config.BASE_DIR / "reports" / "training" / "dataset_split_summary.json")
    _save_json(summary, summary_file)

    logger.info(
        "Dataset manifest prepared: %d train, %d val, %d test (classes=%d)",
        summary["subsets"]["train"],
        summary["subsets"]["val"],
        summary["subsets"]["test"],
        summary["classes_discovered"],
    )

    return summary


def _save_json(data: Any, output_path: Union[str, Path]) -> None:
    """Helper to save dictionary as formatted JSON."""
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare reproducible stratified dataset split manifest (70% train, 15% val, 15% test)."
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=str(Config.BASE_DIR / "dataset" / "raw"),
        help="Root directory of raw class-organized images (default: dataset/raw)",
    )
    parser.add_argument(
        "--train",
        type=float,
        default=0.70,
        help="Train split ratio (default: 0.70)",
    )
    parser.add_argument(
        "--val",
        type=float,
        default=0.15,
        help="Validation split ratio (default: 0.15)",
    )
    parser.add_argument(
        "--test",
        type=float,
        default=0.15,
        help="Test split ratio (default: 0.15)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic random seed (default: 42)",
    )
    parser.add_argument(
        "--manifest-output",
        type=str,
        default=str(Config.BASE_DIR / "reports" / "training" / "dataset_manifest.json"),
        help="Path for saving JSON dataset split manifest",
    )
    parser.add_argument(
        "--class-indices-output",
        type=str,
        default=str(Config.BASE_DIR / "model" / "class_indices.json"),
        help="Path for saving class index mapping dictionary",
    )
    parser.add_argument(
        "--summary-output",
        type=str,
        default=str(Config.BASE_DIR / "reports" / "training" / "dataset_split_summary.json"),
        help="Path for saving JSON split summary report",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Reproducible Dataset Splitter & Manifest Generator (Phase 3)")
    print("=" * 70)
    print(f"Dataset Root:    {args.dataset_root}")
    print(f"Split Ratios:    Train={args.train * 100:.0f}%, Val={args.val * 100:.0f}%, Test={args.test * 100:.0f}%")
    print(f"Random Seed:     {args.seed}")
    print(f"Manifest File:   {args.manifest_output}")
    print(f"Class Indices:   {args.class_indices_output}")
    print("-" * 70)

    try:
        summary = prepare_dataset_manifest(
            dataset_root=args.dataset_root,
            train_ratio=args.train,
            val_ratio=args.val,
            test_ratio=args.test,
            random_seed=args.seed,
            manifest_output_path=args.manifest_output,
            class_indices_path=args.class_indices_output,
            summary_output_path=args.summary_output,
        )

        if summary.get("classes_discovered", 0) == 0:
            print("\n[NOTICE] No classes were processed. Ensure dataset exists in dataset/raw/.")
            if summary.get("warnings"):
                for w in summary["warnings"]:
                    print(f"  [!] {w}")
        else:
            print(f"\n[OK] Partitioned {summary['classes_discovered']} classes ({summary['total_unique_images']} unique images):")
            print(f"  * Training:   {summary['subsets']['train']} images ({summary['subsets']['train'] / max(1, summary['total_unique_images']) * 100:.1f}%)")
            print(f"  * Validation: {summary['subsets']['val']} images ({summary['subsets']['val'] / max(1, summary['total_unique_images']) * 100:.1f}%)")
            print(f"  * Test:       {summary['subsets']['test']} images ({summary['subsets']['test'] / max(1, summary['total_unique_images']) * 100:.1f}%)")

            if summary.get("duplicates_omitted", 0) > 0:
                print(f"  * Duplicates: {summary['duplicates_omitted']} duplicate files omitted to prevent data leakage.")

            print("\nPer-Class Distribution:")
            for cname, counts in summary["per_class_counts"].items():
                print(f"  * {cname:<35}: Train={counts['train']} | Val={counts['val']} | Test={counts['test']} (Total={counts['total']})")

            if summary.get("warnings"):
                print("\nWarnings:")
                for w in summary["warnings"]:
                    print(f"  [!] {w}")

        print("\n" + "=" * 70)
        print(f"Manifest saved to:      {args.manifest_output}")
        print(f"Class indices saved to: {args.class_indices_output}")
        print(f"Split summary saved to: {args.summary_output}")
        print("=" * 70)

    except Exception as err:
        print(f"\n[ERROR] Dataset preparation failed: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
