"""
Unit tests for dataset preparation and manifest generation (tests/test_dataset_preparation.py).
Tests stratified splitting, data leakage guards, class mapping, and manifest persistence.
"""

import json
from pathlib import Path
import pytest
from PIL import Image

from utils.prepare_dataset import prepare_dataset_manifest


def make_test_image(color=(34, 139, 34)) -> bytes:
    """Creates a tiny valid JPEG image as bytes."""
    import io
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), color=color).save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def synthetic_raw_dataset(tmp_path):
    """
    Creates a temporary raw dataset with multiple class folders:
      - Tomato___Early_blight: 10 images
      - Tomato___Late_blight: 10 images
      - Tomato___healthy: 10 images
      - Rare_Class: 2 images
    """
    raw_root = tmp_path / "raw"
    raw_root.mkdir(parents=True)

    classes = {
        "Tomato___Early_blight": 10,
        "Tomato___Late_blight": 10,
        "Tomato___healthy": 10,
        "Rare_Class": 2,
    }

    for cname, count in classes.items():
        cdir = raw_root / cname
        cdir.mkdir()
        for i in range(count):
            img_bytes = make_test_image(color=(i * 15 % 255, 100, 50))
            (cdir / f"img_{i}.jpg").write_bytes(img_bytes)

    return raw_root


def test_prepare_dataset_missing_directory(tmp_path):
    """Verify that a missing dataset directory is handled gracefully without exception."""
    missing = tmp_path / "missing_dir"
    summary = prepare_dataset_manifest(dataset_root=missing, summary_output_path=tmp_path / "sum.json")
    assert summary["classes_discovered"] == 0
    assert len(summary["warnings"]) > 0


def test_prepare_dataset_manifest_and_split_ratios(synthetic_raw_dataset, tmp_path):
    """Verify stratified splitting proportions, manifest creation, and class mapping."""
    manifest_file = tmp_path / "manifest.json"
    indices_file = tmp_path / "class_indices.json"
    summary_file = tmp_path / "summary.json"

    summary = prepare_dataset_manifest(
        dataset_root=synthetic_raw_dataset,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42,
        manifest_output_path=manifest_file,
        class_indices_path=indices_file,
        summary_output_path=summary_file,
    )

    assert summary["classes_discovered"] == 4
    assert indices_file.exists()
    assert manifest_file.exists()
    assert summary_file.exists()

    # Verify class indices
    with open(indices_file, "r") as f:
        indices = json.load(f)
    assert len(indices) == 4
    assert "Tomato___Early_blight" in indices

    # Check that subsets are populated
    assert summary["subsets"]["train"] > 0
    assert summary["subsets"]["val"] > 0
    assert summary["subsets"]["test"] > 0

    # Verify zero data leakage (no path in multiple splits)
    train_paths = {r["relative_path"] for r in summary["manifest"]["train"]}
    val_paths = {r["relative_path"] for r in summary["manifest"]["val"]}
    test_paths = {r["relative_path"] for r in summary["manifest"]["test"]}

    assert train_paths.isdisjoint(val_paths), "Data leakage between train and val!"
    assert train_paths.isdisjoint(test_paths), "Data leakage between train and test!"
    assert val_paths.isdisjoint(test_paths), "Data leakage between val and test!"

    # Verify original raw dataset was not modified
    assert (synthetic_raw_dataset / "Tomato___Early_blight" / "img_0.jpg").exists()


def test_prepare_dataset_deterministic_seed(synthetic_raw_dataset, tmp_path):
    """Verify that using the same random seed produces identical split manifests."""
    sum1 = prepare_dataset_manifest(
        dataset_root=synthetic_raw_dataset,
        random_seed=42,
        manifest_output_path=tmp_path / "man1.json",
        class_indices_path=tmp_path / "idx1.json",
        summary_output_path=tmp_path / "sum1.json",
    )
    sum2 = prepare_dataset_manifest(
        dataset_root=synthetic_raw_dataset,
        random_seed=42,
        manifest_output_path=tmp_path / "man2.json",
        class_indices_path=tmp_path / "idx2.json",
        summary_output_path=tmp_path / "sum2.json",
    )

    assert sum1["manifest"] == sum2["manifest"]


def test_prepare_dataset_duplicate_omission(synthetic_raw_dataset, tmp_path):
    """Verify that exact duplicate images are flagged and omitted from duplicate inclusion."""
    # Create duplicate file in Tomato___healthy
    healthy_dir = synthetic_raw_dataset / "Tomato___healthy"
    orig_bytes = (healthy_dir / "img_0.jpg").read_bytes()
    (healthy_dir / "img_0_duplicate.jpg").write_bytes(orig_bytes)

    summary = prepare_dataset_manifest(
        dataset_root=synthetic_raw_dataset,
        manifest_output_path=tmp_path / "man.json",
        class_indices_path=tmp_path / "idx.json",
        summary_output_path=tmp_path / "sum.json",
    )

    assert summary["duplicates_omitted"] >= 1
    assert any("Duplicate omitted" in w for w in summary["warnings"])
