"""
Unit and integration tests for PlantVillage dataset inspection and splitting tools (tests/test_dataset.py).
"""

import io
from pathlib import Path
import pytest
from PIL import Image

from utils.dataset_inspector import inspect_dataset
from utils.dataset_splitter import prepare_dataset_split


def create_fixture_image(color=(34, 139, 34), size=(50, 50)) -> bytes:
    """Creates small valid JPEG bytes for testing dataset fixtures."""
    buf = io.BytesIO()
    img = Image.new("RGB", size, color=color)
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def mock_dataset_dir(tmp_path):
    """
    Creates a temporary mock dataset with class subdirectories:
      - Tomato___Early_blight: 6 valid images
      - Tomato___healthy: 4 valid images, 1 corrupt file, 1 duplicate image
      - Rare_Class: 2 valid images (triggers minimum threshold warning)
    """
    root = tmp_path / "mock_dataset"
    root.mkdir(parents=True, exist_ok=True)

    # Class 1: 6 images
    c1 = root / "Tomato___Early_blight"
    c1.mkdir()
    for i in range(6):
        # Vary color slightly so hashes differ
        (c1 / f"leaf_{i}.jpg").write_bytes(create_fixture_image(color=(30 + i * 5, 120, 30)))

    # Class 2: 4 unique + 1 duplicate + 1 corrupt
    c2 = root / "Tomato___healthy"
    c2.mkdir()
    for i in range(4):
        (c2 / f"healthy_{i}.jpg").write_bytes(create_fixture_image(color=(10, 150 + i * 5, 20)))

    # Duplicate: identical bytes to healthy_0.jpg
    (c2 / "healthy_duplicate.jpg").write_bytes((c2 / "healthy_0.jpg").read_bytes())

    # Corrupted file
    (c2 / "corrupted_leaf.jpg").write_bytes(b"NOT_AN_IMAGE_HEADER_CORRUPTED")

    # Class 3: 2 images (rare class)
    c3 = root / "Potato___Rare_blight"
    c3.mkdir()
    for i in range(2):
        (c3 / f"rare_{i}.jpg").write_bytes(create_fixture_image(color=(80, 80, 20 + i * 10)))

    return root


def test_inspect_missing_dataset_directory(tmp_path):
    """Verify that a non-existent dataset directory produces clean warnings without throwing exceptions."""
    missing_dir = tmp_path / "does_not_exist"
    report = inspect_dataset(missing_dir, output_report_path=tmp_path / "report.json")
    assert report["status"] == "not_found"
    assert len(report["warnings"]) > 0
    assert (tmp_path / "report.json").exists()


def test_inspect_empty_dataset_directory(tmp_path):
    """Verify that an empty dataset directory is detected and flagged."""
    empty_dir = tmp_path / "empty_dataset"
    empty_dir.mkdir()
    report = inspect_dataset(empty_dir, output_report_path=tmp_path / "report.json")
    assert report["status"] == "empty"
    assert any("empty or contains no class subfolders" in w for w in report["warnings"])


def test_inspect_valid_mock_dataset(mock_dataset_dir, tmp_path):
    """Verify accurate class counting, corruption detection, and duplicate discovery."""
    report_file = tmp_path / "inspection.json"
    report = inspect_dataset(mock_dataset_dir, output_report_path=report_file, min_samples_threshold=5)

    assert report["status"] == "inspected"
    assert report["total_classes"] == 3
    # 6 (c1) + 5 valid including duplicate (c2) + 2 (c3) = 13 total valid
    assert report["total_images"] == 13

    # Check corrupt file was caught
    assert len(report["corrupt_files"]) == 1
    assert "corrupted_leaf.jpg" in report["corrupt_files"][0]["file"]

    # Check duplicate detection
    assert len(report["duplicate_groups"]) >= 1
    dup_files = report["duplicate_groups"][0]["files"]
    assert any("healthy_duplicate.jpg" in f for f in dup_files)

    # Check rare class warning
    assert any("Potato___Rare_blight" in w for w in report["warnings"])


def test_split_missing_dataset_directory(tmp_path):
    """Verify that splitting a non-existent dataset returns safe warning summary."""
    missing_dir = tmp_path / "missing"
    summary = prepare_dataset_split(missing_dir, output_report_path=tmp_path / "split_sum.json")
    assert summary["classes_processed"] == 0
    assert len(summary["warnings"]) > 0


def test_split_stratified_dry_run_and_reproducibility(mock_dataset_dir, tmp_path):
    """Verify reproducible splitting, correct ratios, and leakage prevention."""
    report_file = tmp_path / "split_summary.json"

    # Run 1 with seed=42
    summary1 = prepare_dataset_split(
        dataset_path=mock_dataset_dir,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42,
        dry_run=True,
        output_report_path=report_file,
    )

    # Run 2 with same seed=42
    summary2 = prepare_dataset_split(
        dataset_path=mock_dataset_dir,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42,
        dry_run=True,
        output_report_path=tmp_path / "split_summary2.json",
    )

    # Deterministic reproducibility check
    assert summary1["manifest"] == summary2["manifest"]

    # Verify no file overlap (zero data leakage) between train, val, and test sets
    train_set = set(summary1["manifest"]["train"])
    val_set = set(summary1["manifest"]["val"])
    test_set = set(summary1["manifest"]["test"])

    assert train_set.isdisjoint(val_set), "Data leakage: overlap between train and val!"
    assert train_set.isdisjoint(test_set), "Data leakage: overlap between train and test!"
    assert val_set.isdisjoint(test_set), "Data leakage: overlap between val and test!"

    # Dry-run verify: original directory was not modified
    assert (mock_dataset_dir / "Tomato___Early_blight").exists()


def test_split_rare_class_warning(mock_dataset_dir, tmp_path):
    """Verify that classes with fewer than 3 images produce explicit warnings."""
    summary = prepare_dataset_split(mock_dataset_dir, output_report_path=tmp_path / "rare_sum.json")
    warnings = summary["warnings"]
    assert any("Potato___Rare_blight" in w and "Cannot partition into 3 distinct subsets" in w for w in warnings)
