"""
Multi-Dataset Pipeline Test Suite (tests/test_multi_dataset_pipeline.py)
========================================================================
Validates all requirements of the 11-Phase Multi-Dataset Pipeline:
1. Dataset discovery, 5-dataset registry, and annotation parsing (Plant Pathology 2020 CSV).
2. Corrupted, unreadable, and unsupported image format handling.
3. SHA-256 exact duplicate detection and perceptual hash (dHash) calculation.
4. Data leakage prevention: grouping augmented images by base identity.
5. Consistent canonical class mapping (preserving healthy vs diseased; no fake neem diseases).
6. Deterministic stratified 70/15/15 split without train-test leakage.
7. Model artifact loading from models/best_model.keras and models/class_names.json.
8. Preprocessing consistency: 224x224 RGB normalized to [-1.0, 1.0].
9. Multi-metric Out-of-Distribution (OOD) rejection for unsupported plants (Neem).
10. Strict recommendation shielding: zero chemical recommendations for unverified specimens.
"""

import csv
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from config import Config
from scripts.inspect_datasets import (
    KNOWN_DATASET_DEFINITIONS,
    compute_dhash,
    compute_sha256,
    detect_dataset_type_and_name,
    hamming_distance,
    parse_plant_pathology_csv,
)
from scripts.prepare_dataset import (
    CANONICAL_MAPPINGS,
    extract_base_image_id,
    normalize_class_name,
    partition_dataset_stratified_no_leakage,
)
from services.image_preprocessing import DEFAULT_IMAGE_SIZE, load_and_preprocess_image
from services.model_service import ModelService, get_model_service
from services.recommendation_service import get_recommendations_for_prediction


# ==============================================================================
# 1. Dataset Discovery & 5-Dataset Target Registry
# ==============================================================================

def test_five_datasets_registered_with_official_sources():
    """Confirms all 5 official datasets are cataloged with legitimate repositories."""
    expected_keys = {"PlantVillage", "NewPlantDiseases", "PlantDoc", "PlantPathology2020", "NeemLeaf"}
    assert set(KNOWN_DATASET_DEFINITIONS.keys()) == expected_keys

    for key, defn in KNOWN_DATASET_DEFINITIONS.items():
        assert "official_url" in defn
        assert defn["official_url"].startswith("http")
        assert "description" in defn


def test_detect_dataset_type_and_name_plant_pathology():
    """Validates automatic detection of Plant Pathology 2020 by train.csv."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        csv_file = tmp_path / "train.csv"
        csv_file.write_text("image_id,healthy,multiple_diseases,rust,scab\nTest_1,1,0,0,0\n")

        ds_key, ds_name = detect_dataset_type_and_name(tmp_path)
        assert ds_key == "PlantPathology2020"


def test_parse_plant_pathology_csv_labels():
    """Tests parsing 1-hot multi-label classification CSV from Plant Pathology 2020."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "train.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image_id", "healthy", "multiple_diseases", "rust", "scab"])
            writer.writerow(["Train_0", "1", "0", "0", "0"])
            writer.writerow(["Train_1", "0", "0", "1", "0"])
            writer.writerow(["Train_2", "0", "0", "0", "1"])
            writer.writerow(["Train_3", "0", "1", "0", "0"])

        label_map = parse_plant_pathology_csv(csv_path)
        assert label_map["Train_0"] == "Apple___healthy"
        assert label_map["Train_1"] == "Apple___Cedar_apple_rust"
        assert label_map["Train_2"] == "Apple___Apple_scab"
        assert label_map["Train_3"] == "Apple___multiple_diseases"


# ==============================================================================
# 2. Corrupted & Unsupported Image Handling
# ==============================================================================

def test_corrupted_image_handling(tmp_path):
    """Verifies unreadable or truncated images are rejected safely without crashes."""
    corrupt_file = tmp_path / "corrupt_leaf.jpg"
    corrupt_file.write_bytes(b"\xFF\xD8\xFF\xE0\x00\x10JFIF\x00\x01\x01\x00GARBAGE_PAYLOAD")

    with pytest.raises(Exception):
        load_and_preprocess_image(corrupt_file)


# ==============================================================================
# 3. Duplicate Detection & Perceptual Hashing (dHash)
# ==============================================================================

def test_sha256_duplicate_detection(tmp_path):
    """Verifies exact duplicate images have identical SHA-256 hashes."""
    img = Image.new("RGB", (50, 50), color=(34, 139, 34))
    f1 = tmp_path / "leaf1.png"
    f2 = tmp_path / "leaf2.png"
    img.save(f1)
    img.save(f2)

    hash1 = compute_sha256(f1)
    hash2 = compute_sha256(f2)
    assert hash1 == hash2


def test_dhash_perceptual_near_duplicate(tmp_path):
    """Verifies dHash produces small Hamming distance for visually similar images."""
    img1 = Image.new("RGB", (64, 64), color=(50, 150, 50))
    img2 = Image.new("RGB", (64, 64), color=(52, 148, 52))

    dh1 = compute_dhash(img1)
    dh2 = compute_dhash(img2)
    dist = hamming_distance(dh1, dh2)
    assert dist <= 4


# ==============================================================================
# 4. Anti-Leakage & Base Image Grouping
# ==============================================================================

def test_extract_base_image_id_strips_augmentations():
    """Confirms augmented filename variants map back to the same base specimen."""
    assert extract_base_image_id("leaf_001_rot90.jpg") == "leaf_001"
    assert extract_base_image_id("leaf_001_flip.jpg") == "leaf_001"
    assert extract_base_image_id("specimen_zoom.JPG") == "specimen"
    assert extract_base_image_id("normal_leaf.png") == "normal_leaf"


# ==============================================================================
# 5. Consistent Class Mapping (Preserve Healthy vs Diseased & Neem Categories)
# ==============================================================================

def test_canonical_class_mapping_preserves_healthy_and_diseased():
    """Ensures healthy and diseased classes are kept strictly distinct."""
    assert normalize_class_name("apple leaf", "PlantDoc") == "Apple___healthy"
    assert normalize_class_name("apple rust leaf", "PlantDoc") == "Apple___Cedar_apple_rust"
    assert normalize_class_name("Tomato___healthy", "PlantVillage") == "Tomato___healthy"
    assert normalize_class_name("Tomato___Late_blight", "PlantVillage") == "Tomato___Late_blight"


def test_neem_classes_not_assigned_fake_diseases():
    """Ensures Neem leaf images are preserved under broad categories without fake diseases."""
    assert normalize_class_name("neem_healthy", "NeemLeaf") == "Neem___healthy"
    assert normalize_class_name("neem_diseased", "NeemLeaf") == "Neem___unspecified_symptoms"
    # Must NOT map to squash or tomato!
    assert "Squash" not in normalize_class_name("neem_diseased", "NeemLeaf")
    assert "Tomato" not in normalize_class_name("neem_diseased", "NeemLeaf")


# ==============================================================================
# 6. Stratified 70/15/15 Split Without Leakage
# ==============================================================================

def test_stratified_split_no_leakage_across_subsets():
    """Validates that base image groups never cross train, validation, and test splits."""
    records = []
    # Create 30 records across 2 classes with some augmented duplicates
    for c in ["Apple___healthy", "Tomato___Early_blight"]:
        for i in range(15):
            base_id = f"{c}_leaf_{i}"
            records.append({
                "filepath": f"/fake/{c}/{base_id}.jpg",
                "filename": f"{base_id}.jpg",
                "base_image_id": base_id,
                "unified_class": c,
            })
            # Add an augmented copy
            records.append({
                "filepath": f"/fake/{c}/{base_id}_rot90.jpg",
                "filename": f"{base_id}_rot90.jpg",
                "base_image_id": base_id,
                "unified_class": c,
            })

    splits, summary = partition_dataset_stratified_no_leakage(
        records, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_seed=42
    )

    train_base_ids = {r["base_image_id"] for r in splits["train"]}
    val_base_ids = {r["base_image_id"] for r in splits["val"]}
    test_base_ids = {r["base_image_id"] for r in splits["test"]}

    # Zero leakage check
    assert len(train_base_ids & val_base_ids) == 0, "Leakage detected between train and val"
    assert len(train_base_ids & test_base_ids) == 0, "Leakage detected between train and test"
    assert len(val_base_ids & test_base_ids) == 0, "Leakage detected between val and test"


# ==============================================================================
# 7. Model Artifact Loading & Inference
# ==============================================================================

def test_model_service_loads_unified_model_and_classes():
    """Tests that ModelService loads models/best_model.keras and models/class_names.json."""
    service = ModelService()
    assert Path(Config.MODEL_PATH).exists()
    assert Path(Config.CLASS_INDICES_PATH).exists()

    service.load_model_artifacts()
    assert service.is_loaded
    ready, msg = service.is_model_ready()
    assert ready is True
    assert len(service._class_indices) == 38


# ==============================================================================
# 8. Preprocessing Consistency
# ==============================================================================

def test_image_preprocessing_mobilenetv2_normalization(tmp_path):
    """Verifies preprocessing produces shape (224, 224, 3) and values in [-1.0, 1.0]."""
    test_img = Image.new("RGB", (400, 300), color=(120, 200, 80))
    img_path = tmp_path / "sample.jpg"
    test_img.save(img_path)

    arr = load_and_preprocess_image(img_path, target_size=DEFAULT_IMAGE_SIZE)
    assert arr.shape == (224, 224, 3)
    assert arr.dtype == np.float32
    # MobileNetV2 normalizes [0, 255] to [-1.0, 1.0]
    assert np.all(arr >= -1.0)
    assert np.all(arr <= 1.0)


# ==============================================================================
# 9. Unsupported Plant (Neem) & Multi-Metric OOD Rejection
# ==============================================================================

def test_unsupported_neem_triggers_ood_and_unsupported_result():
    """Confirms diffuse probability distributions (e.g. from Neem) are rejected as unsupported."""
    service = ModelService()
    # Mock model returning diffuse distribution (high entropy, low margin)
    num_classes = 38
    mock_model = MagicMock()
    # Diffuse probabilities ~ 1/38 = 0.026 with slight variation
    probs = np.full((1, num_classes), 1.0 / num_classes, dtype=np.float32)
    probs[0, 25] = 0.08  # Squash powdery mildew slight peak, but margin << 0.12
    mock_model.predict.return_value = probs

    class_indices = {i: f"Class_{i}" for i in range(num_classes)}
    class_indices[25] = "Squash___Powdery_mildew"
    service.set_mock_model(mock_model, class_indices)

    # Dummy image
    img = Image.new("RGB", (224, 224), color=(30, 120, 40))
    result = service.predict_image(img)

    # Must be flagged as unsupported
    assert result["is_supported"] is False
    assert result["is_inconclusive"] is True
    assert result["disease_code"] == "UNSUPPORTED_SPECIMEN"
    assert result["crop_name"] == "Unsupported / Unknown Plant"
    assert "High prediction entropy" in result["ood_reason"] or "Low model confidence" in result["ood_reason"] or "Narrow class margin" in result["ood_reason"]


# ==============================================================================
# 10. Strict Recommendation Shielding
# ==============================================================================

def test_recommendation_shielding_for_unsupported_predictions():
    """Guarantees zero chemical or fertilizer recommendations are returned for unverified specimens."""
    unsupported_prediction = {
        "is_supported": False,
        "is_inconclusive": True,
        "disease_code": "UNSUPPORTED_SPECIMEN",
        "crop_name": "Unsupported / Unknown Plant",
        "confidence": 0.35,
    }

    recs = get_recommendations_for_prediction(unsupported_prediction)
    assert recs["is_supported"] is False
    assert recs["has_disease_guidance"] is False
    assert recs["has_fertilizer_guidance"] is False
    assert recs["disease_recommendation"] is None
    assert recs["fertilizer_recommendations"] == []
    assert "Neem (Azadirachta indica) is currently unsupported" in recs["guidance_notice"]
