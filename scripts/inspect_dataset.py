"""
Command-Line Interface for PlantVillage Dataset Inspection
Usage:
    python scripts/inspect_dataset.py
    python scripts/inspect_dataset.py --path dataset/PlantVillage
"""

import argparse
import sys
from pathlib import Path

# Configure utf-8 stdout on Windows if supported
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
from utils.dataset_inspector import PLANTVILLAGE_SOURCE_INFO, inspect_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Inspect PlantVillage foliar disease dataset for image integrity, class distribution, and duplicates."
    )
    parser.add_argument(
        "--path",
        type=str,
        default=str(Config.DATASET_DIR / "PlantVillage"),
        help="Path to the PlantVillage dataset directory (default: dataset/PlantVillage)",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=15,
        help="Minimum recommended samples per class (default: 15)",
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default=str(Config.REPORTS_DIR / "dataset_inspection_report.json"),
        help="Path where JSON report should be saved",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("PlantVillage Foliar Dataset Inspection Tool")
    print("=" * 70)
    print(f"Dataset Target: {args.path}")
    print(f"Output Report:  {args.output_report}")
    print("-" * 70)

    report = inspect_dataset(
        dataset_path=args.path,
        output_report_path=args.output_report,
        min_samples_threshold=args.min_samples,
    )

    status = report.get("status")
    print(f"Status: {status.upper()}")

    if status == "not_found":
        print("\n[NOTICE] The target dataset folder was not found on disk.")
        print("PlantVillage dataset is not downloaded by default.")
        print("\nLegitimate Acquisition Sources:")
        for url in PLANTVILLAGE_SOURCE_INFO["repository_urls"]:
            print(f"  * {url}")
        print("\nExpected directory organization:")
        print("  dataset/")
        print("    PlantVillage/")
        print("      Tomato___Early_blight/")
        print("      Tomato___Late_blight/")
        print("      Tomato___healthy/")
        print("      Potato___Early_blight/")
        print("      Potato___Late_blight/")
        print("      Potato___healthy/")
    elif status == "empty":
        print("\n[WARNING] Dataset directory exists but contains no class subfolders.")
    else:
        print(f"\n[OK] Found {report['total_classes']} classes with {report['total_images']} total images.")
        print("\nClass Distribution Summary:")
        for cname, details in report["classes"].items():
            print(f"  * {cname}: {details['valid_images']} valid ({details['corrupt_images']} corrupt)")

        if report.get("duplicate_groups"):
            print(f"\n[WARNING] Detected {len(report['duplicate_groups'])} exact duplicate file groups.")

        imbalance = report.get("class_imbalance", {})
        if imbalance:
            print(f"\nClass Imbalance Ratio: {imbalance.get('imbalance_ratio', 1.0)}x")
            print(f"  Min Class Size: {imbalance.get('min_class_size')}")
            print(f"  Max Class Size: {imbalance.get('max_class_size')}")
            print(f"  Mean Class Size: {imbalance.get('mean_class_size')}")

    if report.get("warnings"):
        print("\nInspection Warnings:")
        for w in report["warnings"]:
            print(f"  [!] {w}")

    print("\n" + "=" * 70)
    print(f"Detailed machine-readable JSON saved to: {args.output_report}")
    print("=" * 70)


if __name__ == "__main__":
    main()
