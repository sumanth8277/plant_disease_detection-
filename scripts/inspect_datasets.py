"""
Multi-Dataset Inspection Tool (scripts/inspect_datasets.py)
===========================================================
Recursively scans, validates, and audits plant disease datasets:
1. PlantVillage Dataset
2. New Plant Diseases Dataset (Augmented)
3. PlantDoc Dataset
4. Plant Pathology 2020 (FGVC7)
5. NeemLeaf Dataset

Supports JPG, JPEG, PNG, and WEBP.
Detects corrupt images, unreadable files, dimensions, and label formats.
Parses CSV annotations (e.g., Plant Pathology 2020) and folder-based annotations.
Detects exact duplicates via SHA-256 and near-duplicates via perceptual hashing (dHash).
Saves comprehensive reports to:
  - reports/dataset_inspection.json
  - reports/class_distribution.csv
"""

import argparse
import csv
import hashlib
import json
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from PIL import Image, UnidentifiedImageError

# Configure stdout encoding on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("inspect_datasets")

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

KNOWN_DATASET_DEFINITIONS = {
    "PlantVillage": {
        "name": "PlantVillage Dataset",
        "official_url": "https://github.com/spMohanty/PlantVillage-Dataset",
        "description": "54,000+ laboratory leaf images across 38 crop-disease categories.",
        "type": "folder_based",
        "typical_paths": ["PlantVillage", "plantvillage", "raw/PlantVillage", "PlantVillage-Dataset"],
    },
    "NewPlantDiseases": {
        "name": "New Plant Diseases Dataset (Augmented)",
        "official_url": "https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset",
        "description": "87,000+ augmented and raw crop leaf images spanning 38 classes.",
        "type": "folder_based",
        "typical_paths": ["NewPlantDiseases", "new-plant-diseases-dataset", "New Plant Diseases Dataset(Augmented)", "raw/NewPlantDiseases"],
    },
    "PlantDoc": {
        "name": "PlantDoc Dataset",
        "official_url": "https://github.com/pratikkayal/PlantDoc-Dataset",
        "description": "2,500+ in-situ, real-world field images across 27 classes with bounding boxes.",
        "type": "annotation_or_folder",
        "typical_paths": ["PlantDoc", "PlantDoc-Dataset", "plantdoc", "raw/PlantDoc"],
    },
    "PlantPathology2020": {
        "name": "Plant Pathology 2020 (FGVC7)",
        "official_url": "https://www.kaggle.com/competitions/plant-pathology-2020-fgvc7/data",
        "description": "Folio images of apple leaves with CSV annotations: healthy, rust, scab, multiple_diseases.",
        "type": "csv_based",
        "typical_paths": ["PlantPathology2020", "plant-pathology-2020-fgvc7", "plant_pathology_2020", "raw/PlantPathology2020"],
    },
    "NeemLeaf": {
        "name": "NeemLeaf Dataset",
        "official_url": "https://data.mendeley.com/datasets/h3786mkn7t/1",
        "description": "Images of Azadirachta indica (Neem) leaves categorized by appearance/condition.",
        "type": "folder_based",
        "typical_paths": ["NeemLeaf", "neemleaf", "neem_leaf", "raw/NeemLeaf", "Neem"],
    },
}


def compute_sha256(filepath: Path) -> str:
    """Computes SHA-256 hash for exact duplicate detection."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_dhash(img: Image.Image, hash_size: int = 8) -> str:
    """Computes 64-bit difference hash (dHash) for near-duplicate perceptual detection."""
    try:
        resized = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
        pixels = list(resized.tobytes())
        diff = []
        for row in range(hash_size):
            for col in range(hash_size):
                pixel_left = pixels[row * (hash_size + 1) + col]
                pixel_right = pixels[row * (hash_size + 1) + col + 1]
                diff.append(pixel_left > pixel_right)

        decimal_value = 0
        hex_string = []
        for idx, val in enumerate(diff):
            if val:
                decimal_value += 2 ** (idx % 8)
            if (idx % 8) == 7:
                hex_string.append(hex(decimal_value)[2:].rjust(2, "0"))
                decimal_value = 0
        return "".join(hex_string)
    except Exception:
        return ""


def hamming_distance(hex1: str, hex2: str) -> int:
    """Calculates hamming distance between two hex hash strings."""
    if not hex1 or not hex2 or len(hex1) != len(hex2):
        return 999
    try:
        val1 = int(hex1, 16)
        val2 = int(hex2, 16)
        xor = val1 ^ val2
        return bin(xor).count("1")
    except ValueError:
        return 999


def parse_plant_pathology_csv(csv_path: Path) -> Dict[str, str]:
    """
    Parses Plant Pathology 2020 FGVC7 train.csv.
    Format: image_id, healthy, multiple_diseases, rust, scab
    Returns mapping of image_id -> label class string.
    """
    annotations: Dict[str, str] = {}
    if not csv_path.exists():
        return annotations

    with open(csv_path, mode="r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = row.get("image_id") or row.get("image")
            if not image_id:
                continue
            # Determine class from 1-hot columns
            if row.get("healthy") == "1":
                cls = "Apple___healthy"
            elif row.get("multiple_diseases") == "1":
                cls = "Apple___multiple_diseases"
            elif row.get("rust") == "1":
                cls = "Apple___Cedar_apple_rust"
            elif row.get("scab") == "1":
                cls = "Apple___Apple_scab"
            else:
                cls = "Apple___unspecified"
            annotations[image_id.strip()] = cls
    return annotations


def detect_dataset_type_and_name(directory: Path) -> Tuple[str, str]:
    """Identifies which of the 5 datasets a directory corresponds to."""
    dir_name_lower = directory.name.lower()
    for ds_key, defn in KNOWN_DATASET_DEFINITIONS.items():
        for candidate in defn["typical_paths"]:
            cand_name = Path(candidate).name.lower()
            if cand_name == dir_name_lower or cand_name in dir_name_lower:
                return ds_key, defn["name"]

    # Check for Plant Pathology train.csv
    if (directory / "train.csv").exists():
        return "PlantPathology2020", KNOWN_DATASET_DEFINITIONS["PlantPathology2020"]["name"]

    return "CustomOrMerged", directory.name


def inspect_single_dataset(
    dataset_dir: Path,
    dataset_key: str,
    compute_perceptual: bool = True,
    max_dhash_samples: int = 500,
) -> Dict[str, Any]:
    """Deeply inspects a single dataset directory."""
    logger.info("Scanning dataset at: %s (Detected type: %s)", dataset_dir, dataset_key)

    result = {
        "key": dataset_key,
        "path": str(dataset_dir.resolve()),
        "scanned_at": datetime.now().isoformat(),
        "total_files_scanned": 0,
        "valid_images": 0,
        "corrupt_files": [],
        "extensions": defaultdict(int),
        "dimensions": defaultdict(int),
        "classes": defaultdict(int),
        "class_examples": defaultdict(list),
        "annotations": {},
        "exact_duplicates": [],
        "near_duplicates": [],
        "has_csv_annotations": False,
    }

    # Check for CSV annotations
    csv_files = list(dataset_dir.glob("*.csv")) + list(dataset_dir.glob("train/*.csv"))
    csv_label_map: Dict[str, str] = {}
    if csv_files:
        for cf in csv_files:
            if "train" in cf.name.lower():
                result["has_csv_annotations"] = True
                csv_label_map = parse_plant_pathology_csv(cf)
                result["annotations"]["csv_file"] = str(cf.name)
                result["annotations"]["records_found"] = len(csv_label_map)
                break

    sha_map: Dict[str, List[Path]] = defaultdict(list)
    dhash_list: List[Tuple[str, Path, str]] = []  # (dhash, path, class)

    for root, _, files in os.walk(dataset_dir):
        root_path = Path(root)
        for fname in files:
            fpath = root_path / fname
            ext = fpath.suffix.lower()

            if ext not in SUPPORTED_EXTENSIONS:
                continue

            result["total_files_scanned"] += 1
            result["extensions"][ext] += 1

            # Validate Image Integrity and Dimensions
            try:
                with Image.open(fpath) as img:
                    img.verify()
                # Reopen to read dimensions and compute dhash
                with Image.open(fpath) as img:
                    dim_str = f"{img.width}x{img.height}"
                    result["dimensions"][dim_str] += 1
                    result["valid_images"] += 1

                    # Determine class label
                    stem = fpath.stem
                    if stem in csv_label_map:
                        cls_name = csv_label_map[stem]
                    else:
                        cls_name = root_path.name
                        # If inside train/val/test subfolder, check parent
                        if cls_name.lower() in {"train", "val", "valid", "test", "images"} and root_path.parent != dataset_dir:
                            cls_name = root_path.parent.name

                    result["classes"][cls_name] += 1
                    if len(result["class_examples"][cls_name]) < 3:
                        result["class_examples"][cls_name].append(str(fpath.name))

                    # SHA-256 for exact duplicate
                    sha = compute_sha256(fpath)
                    sha_map[sha].append(fpath)

                    # Perceptual hash for near duplicate (sample up to max_dhash_samples)
                    if compute_perceptual and len(dhash_list) < max_dhash_samples:
                        dh = compute_dhash(img)
                        if dh:
                            dhash_list.append((dh, fpath, cls_name))

            except (UnidentifiedImageError, OSError, Exception) as err:
                result["corrupt_files"].append({"file": str(fpath), "error": str(err)})

    # Detect exact duplicates
    for sha, paths in sha_map.items():
        if len(paths) > 1:
            result["exact_duplicates"].append({
                "sha256": sha,
                "count": len(paths),
                "files": [str(p) for p in paths[:5]],
            })

    # Detect near duplicates (Hamming distance <= 4)
    if compute_perceptual and len(dhash_list) > 1:
        n = len(dhash_list)
        for i in range(min(n, 200)):
            for j in range(i + 1, min(n, 200)):
                dh1, p1, c1 = dhash_list[i]
                dh2, p2, c2 = dhash_list[j]
                dist = hamming_distance(dh1, dh2)
                if dist <= 4:
                    result["near_duplicates"].append({
                        "file1": str(p1),
                        "file2": str(p2),
                        "class1": c1,
                        "class2": c2,
                        "distance": dist,
                    })

    # Convert defaultdicts to regular dicts for JSON serialization
    result["extensions"] = dict(result["extensions"])
    result["dimensions"] = dict(result["dimensions"])
    result["classes"] = dict(result["classes"])
    result["class_examples"] = dict(result["class_examples"])

    return result


def inspect_all_datasets(
    data_dir: Path,
    output_json: Path,
    output_csv: Path,
) -> Dict[str, Any]:
    """Inspects all datasets located within data_dir, checking for all 5 targets."""
    logger.info("Initializing comprehensive dataset inspection across: %s", data_dir)

    report: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "scan_root": str(data_dir.resolve()),
        "five_targets": KNOWN_DATASET_DEFINITIONS,
        "datasets_found": {},
        "missing_datasets": [],
        "overall_summary": {
            "total_datasets_scanned": 0,
            "total_images_scanned": 0,
            "total_valid_images": 0,
            "total_corrupt_files": 0,
            "total_exact_duplicate_groups": 0,
            "total_classes_detected": 0,
        },
        "consolidated_classes": defaultdict(int),
    }

    subdirs = [d for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]

    # Check for direct class subfolders vs dataset parent folders
    # If subdirs are class folders directly (e.g. Tomato___healthy), treat data_dir as one dataset
    has_direct_classes = any("___" in d.name for d in subdirs)

    candidate_dataset_dirs = []
    if has_direct_classes:
        candidate_dataset_dirs.append(data_dir)
    else:
        for d in subdirs:
            # Also check subdirectories (e.g. dataset/raw/PlantVillage)
            if d.name.lower() in {"raw", "processed"}:
                for child in d.iterdir():
                    if child.is_dir() and not child.name.startswith("."):
                        candidate_dataset_dirs.append(child)
            else:
                candidate_dataset_dirs.append(d)

    found_keys = set()
    for ds_path in candidate_dataset_dirs:
        ds_key, ds_name = detect_dataset_type_and_name(ds_path)
        found_keys.add(ds_key)
        inspection_data = inspect_single_dataset(ds_path, ds_key)
        report["datasets_found"][ds_key] = inspection_data

        # Accumulate metrics
        report["overall_summary"]["total_images_scanned"] += inspection_data["total_files_scanned"]
        report["overall_summary"]["total_valid_images"] += inspection_data["valid_images"]
        report["overall_summary"]["total_corrupt_files"] += len(inspection_data["corrupt_files"])
        report["overall_summary"]["total_exact_duplicate_groups"] += len(inspection_data["exact_duplicates"])

        for cls, count in inspection_data["classes"].items():
            report["consolidated_classes"][cls] += count

    report["overall_summary"]["total_datasets_scanned"] = len(report["datasets_found"])
    report["overall_summary"]["total_classes_detected"] = len(report["consolidated_classes"])
    report["consolidated_classes"] = dict(report["consolidated_classes"])

    # Document missing datasets among the 5 targets
    for target_key, defn in KNOWN_DATASET_DEFINITIONS.items():
        if target_key not in found_keys:
            report["missing_datasets"].append({
                "dataset_key": target_key,
                "name": defn["name"],
                "official_url": defn["official_url"],
                "description": defn["description"],
                "expected_directory": str(data_dir / "raw" / target_key),
                "acquisition_guidance": (
                    f"Download from {defn['official_url']} and extract image contents into "
                    f"'{data_dir / 'raw' / target_key}'."
                ),
            })

    # Save JSON report
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info("Saved JSON inspection report: %s", output_json)

    # Save CSV summary
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset_source", "class_name", "valid_image_count", "corrupt_count"])
        for ds_key, ds_info in report["datasets_found"].items():
            for cname, count in ds_info["classes"].items():
                writer.writerow([ds_key, cname, count, len(ds_info["corrupt_files"])])

        if not report["datasets_found"]:
            writer.writerow(["NO_DATASETS_FOUND", "N/A", 0, 0])

    logger.info("Saved CSV class distribution report: %s", output_csv)
    return report


def main():
    parser = argparse.ArgumentParser(description="Multi-Dataset Foliar Disease Inspection Tool")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(BASE_DIR / "dataset"),
        help="Root dataset directory to scan (default: dataset/)",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=str(BASE_DIR / "reports" / "dataset_inspection.json"),
        help="Output path for JSON report",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=str(BASE_DIR / "reports" / "class_distribution.csv"),
        help="Output path for CSV summary",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)

    print("=" * 75)
    print("Multi-Dataset Foliar Disease Inspection Pipeline")
    print("Targets: PlantVillage, New Plant Diseases, PlantDoc, Plant Pathology 2020, NeemLeaf")
    print("=" * 75)
    print(f"Scanning target: {data_dir.resolve()}")
    print(f"Output JSON:     {output_json}")
    print(f"Output CSV:      {output_csv}")
    print("-" * 75)

    report = inspect_all_datasets(data_dir, output_json, output_csv)

    print("\n[INSPECTION SUMMARY]")
    print(f"Datasets Scanned:       {report['overall_summary']['total_datasets_scanned']}")
    print(f"Total Valid Images:     {report['overall_summary']['total_valid_images']}")
    print(f"Corrupt Images:         {report['overall_summary']['total_corrupt_files']}")
    print(f"Exact Duplicate Groups: {report['overall_summary']['total_exact_duplicate_groups']}")
    print(f"Classes Detected:       {report['overall_summary']['total_classes_detected']}")

    if report["missing_datasets"]:
        print("\n[MISSING DATASETS]")
        for m in report["missing_datasets"]:
            print(f"  * {m['name']} ({m['dataset_key']})")
            print(f"    Expected Location: {m['expected_directory']}")
            print(f"    Official URL:      {m['official_url']}")

    print("=" * 75)


if __name__ == "__main__":
    main()
