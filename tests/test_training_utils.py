"""
Unit tests for training utilities and inspection CLI tools (tests/test_training_utils.py).
Tests inspect_dataset_directory, training prerequisites validation, and error guards.
"""

import json
from pathlib import Path
import pytest
from PIL import Image

from utils.evaluate_model import evaluate_trained_model
from utils.inspect_dataset import inspect_dataset_directory
from utils.train_model import train_disease_classifier


def create_sample_leaf(path: Path, color=(34, 139, 34)) -> None:
    """Helper to save a valid test JPEG image."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (48, 48), color=color).save(path, format="JPEG")


def test_inspect_dataset_directory_missing_path(tmp_path):
    """Verify that inspecting a non-existent dataset directory returns clean warnings."""
    missing = tmp_path / "does_not_exist"
    report = inspect_dataset_directory(dataset_root=missing, output_report_path=tmp_path / "rep.json")
    assert report["status"] == "not_found"
    assert len(report["warnings"]) > 0


def test_inspect_dataset_directory_with_corrupted_file(tmp_path):
    """Verify that corrupt images are caught and recorded in corrupt_files list."""
    root = tmp_path / "raw_classes"
    create_sample_leaf(root / "Tomato___Early_blight" / "leaf1.jpg")
    create_sample_leaf(root / "Tomato___Early_blight" / "leaf2.jpg")

    # Add corrupted file
    corrupt_path = root / "Tomato___Early_blight" / "corrupted.jpg"
    corrupt_path.write_bytes(b"\xff\xd8\xff\xe0" + b"CorruptBytes" * 20)

    report = inspect_dataset_directory(dataset_root=root, output_report_path=tmp_path / "rep.json")
    assert report["status"] == "inspected"
    assert report["total_classes"] == 1
    assert report["total_valid_images"] == 2
    assert len(report["corrupt_files"]) == 1
    assert "corrupted.jpg" in report["corrupt_files"][0]["file"]


def test_train_model_missing_manifest_raises_error(tmp_path):
    """Verify that train_disease_classifier raises FileNotFoundError when manifest is absent."""
    missing_manifest = tmp_path / "missing_manifest.json"
    with pytest.raises(FileNotFoundError, match="Dataset manifest not found"):
        train_disease_classifier(manifest_path=missing_manifest)


def test_train_model_empty_train_records_raises_error(tmp_path):
    """Verify that train_disease_classifier raises ValueError when manifest has no training records."""
    manifest_file = tmp_path / "manifest.json"
    indices_file = tmp_path / "class_indices.json"

    with open(indices_file, "w") as f:
        json.dump({"Tomato___healthy": 0}, f)

    with open(manifest_file, "w") as f:
        json.dump({"manifest": {"train": [], "val": []}}, f)

    with pytest.raises(ValueError, match="does not contain sufficient train/val records"):
        train_disease_classifier(manifest_path=manifest_file, class_indices_path=indices_file)


def test_evaluate_model_missing_model_raises_error(tmp_path):
    """Verify that evaluate_trained_model raises FileNotFoundError when model is missing."""
    missing_model = tmp_path / "plant_disease_model.keras"
    with pytest.raises(FileNotFoundError, match="Trained model not found"):
        evaluate_trained_model(model_path=missing_model)
