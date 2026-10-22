"""
Command-Line Interface for Reproducible Dataset Splitting
Usage:
    python scripts/prepare_dataset_split.py --dry-run
    python scripts/prepare_dataset_split.py --path dataset/PlantVillage --train 0.70 --val 0.15 --test 0.15
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
from utils.dataset_splitter import prepare_dataset_split


def main():
    parser = argparse.ArgumentParser(
        description="Prepare reproducible, stratified train/validation/test split for plant disease leaf images."
    )
    parser.add_argument(
        "--path",
        type=str,
        default=str(Config.DATASET_DIR / "PlantVillage"),
        help="Path to source dataset directory (default: dataset/PlantVillage)",
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
        help="Random seed for deterministic splitting (default: 42)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Config.DATASET_DIR / "split"),
        help="Target directory if copying files (used when not in dry-run mode)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Execute physical file copy into train/val/test folders. By default, only a dry-run summary is produced.",
    )
    parser.add_argument(
        "--report",
        type=str,
        default=str(Config.REPORTS_DIR / "dataset_split_summary.json"),
        help="Path where JSON split summary will be stored",
    )

    args = parser.parse_args()
    is_dry_run = not args.apply

    print("=" * 70)
    print("PlantVillage Dataset Split Preparation (Phase 2 & 3 Utility)")
    print("=" * 70)
    print(f"Source Directory: {args.path}")
    print(f"Ratios:           Train={args.train * 100:.0f}%, Val={args.val * 100:.0f}%, Test={args.test * 100:.0f}%")
    print(f"Random Seed:      {args.seed}")
    print(f"Execution Mode:   {'DRY RUN (Manifest Only)' if is_dry_run else 'APPLY (Copy Files)'}")
    print(f"Report Output:    {args.report}")
    print("-" * 70)

    try:
        summary = prepare_dataset_split(
            dataset_path=args.path,
            train_ratio=args.train,
            val_ratio=args.val,
            test_ratio=args.test,
            random_seed=args.seed,
            output_dir=args.output_dir if not is_dry_run else None,
            dry_run=is_dry_run,
            output_report_path=args.report,
        )

        if summary.get("classes_processed", 0) == 0:
            print("\n[NOTICE] No classes were processed. Ensure source dataset exists before running split.")
            if summary.get("warnings"):
                for w in summary["warnings"]:
                    print(f"  [!] {w}")
        else:
            print(f"\n[OK] Partitioned {summary['classes_processed']} classes ({summary['total_images']} total images):")
            print(f"  * Training:   {summary['subsets']['train']} images ({summary['subsets']['train'] / max(1, summary['total_images']) * 100:.1f}%)")
            print(f"  * Validation: {summary['subsets']['val']} images ({summary['subsets']['val'] / max(1, summary['total_images']) * 100:.1f}%)")
            print(f"  * Test:       {summary['subsets']['test']} images ({summary['subsets']['test'] / max(1, summary['total_images']) * 100:.1f}%)")

            print("\nPer-Class Distribution:")
            for cname, counts in summary["per_class_summary"].items():
                print(f"  * {cname:<35}: Train={counts['train']} | Val={counts['val']} | Test={counts['test']} (Total={counts['total']})")

            if summary.get("warnings"):
                print("\nSplit Warnings:")
                for w in summary["warnings"]:
                    print(f"  [!] {w}")

        print("\n" + "=" * 70)
        print(f"Split summary written to: {args.report}")
        print("=" * 70)

    except Exception as err:
        print(f"\n[ERROR] Failed to prepare dataset split: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
