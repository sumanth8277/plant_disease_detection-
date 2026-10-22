"""
Test Suite for Hierarchical Plant Identification & Neem Handling
(tests/test_plant_identification_and_neem.py)
==================================================================
Tests:
1. Plant species identification separates botanical species from disease diagnosis.
2. Positive identification of Neem (Azadirachta indica) while declaring disease unsupported.
3. Separate fields: identified plant, predicted disease, confidence meaning, supported flags.
4. Strict withholding of pesticides and fertilizers for Neem and unsupported plants.
5. Calibrated rejection of non-plant, severely blurred, and poor-quality images.
6. Verification that supported crops (e.g. Tomato) receive verified diagnoses and recommendations.
"""

from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
import pytest
from PIL import Image

from services.model_service import ModelService
from services.plant_identifier_service import (
    analyze_leaf_morphology_and_color,
    run_hierarchical_identification,
)
from services.recommendation_service import get_recommendations_for_prediction


def test_neem_morphology_detection():
    """Validates morphology analyzer detects characteristic olive/forest green foliage."""
    # Synthetic foliage patch matching typical Neem leaf color distribution
    img = Image.new("RGB", (200, 200), color=(75, 115, 50))
    res = analyze_leaf_morphology_and_color(img)

    assert res["is_valid_plant_image"] is True
    assert res["foliage_percentage"] >= 50.0
    assert res["is_neem_candidate"] is True


def test_non_plant_rejection_low_foliage():
    """Validates that images lacking foliage (e.g. blank, gray, random object) are gated out."""
    non_plant_img = Image.new("RGB", (100, 100), color=(180, 180, 180))
    res = analyze_leaf_morphology_and_color(non_plant_img)

    assert res["foliage_percentage"] < 15.0
    assert res["is_valid_plant_image"] is False


def test_hierarchical_identification_neem_output():
    """Tests hierarchical identifier returns proper Neem plant status and unsupported disease status."""
    class_indices = {i: f"Class_{i}" for i in range(38)}
    # Flat/diffuse probability distribution across 38 classes ~ 0.026
    probs = np.full(38, 1.0 / 38, dtype=np.float32)
    top_indices = [0, 1, 2]

    # Create dummy leaf image
    img = Image.new("RGB", (100, 100), color=(70, 110, 45))
    hier = run_hierarchical_identification(
        image_source=img,
        probs=probs,
        class_indices=class_indices,
        top_indices=top_indices,
        margin=0.01,
        normalized_entropy=0.85,
    )

    assert hier["identified_plant"] == "Neem (Azadirachta indica)"
    assert hier["is_plant_supported"] is True
    assert hier["is_disease_supported"] is False
    assert hier["disease_status_message"] == "Neem identified, but neem disease detection is not yet supported"
    assert hier["is_neem"] is True


def test_model_service_predict_image_neem_integration():
    """Tests end-to-end predict_image output for a Neem specimen."""
    service = ModelService()
    num_classes = 38
    mock_model = MagicMock()
    # Diffuse distribution simulating ~14.5% top score for Squash powdery mildew
    probs = np.full((1, num_classes), 0.02, dtype=np.float32)
    probs[0, 25] = 0.1446  # 14.46%
    mock_model.predict.return_value = probs

    class_indices = {i: f"Class_{i}" for i in range(num_classes)}
    class_indices[25] = "Squash___Powdery_mildew"
    service.set_mock_model(mock_model, class_indices)

    # Leaf image matching Neem color
    neem_img = Image.new("RGB", (224, 224), color=(72, 108, 48))
    result = service.predict_image(neem_img)

    # Validate separate fields
    assert result["identified_plant"] == "Neem (Azadirachta indica)"
    assert result["is_plant_supported"] is True
    assert result["is_disease_supported"] is False
    assert result["disease_code"] == "NEEM_DISEASE_UNSUPPORTED"
    assert result["disease_status_message"] == "Neem identified, but neem disease detection is not yet supported"
    assert "Neem (Azadirachta indica)" in result["crop_name"]

    # Recommendation check: ZERO pesticides, ZERO fertilizers
    recs = get_recommendations_for_prediction(result)
    assert recs["has_disease_guidance"] is False
    assert recs["has_fertilizer_guidance"] is False
    assert recs["disease_recommendation"] is None
    assert recs["fertilizer_recommendations"] == []
    assert "strictly withheld" in recs["guidance_notice"]


def test_supported_crop_receives_verified_diagnosis_and_recommendations():
    """Confirms that a true supported crop (e.g. Tomato Early Blight) receives verified recommendations."""
    service = ModelService()
    num_classes = 38
    mock_model = MagicMock()
    # High confidence on Tomato Early Blight (Class 29)
    probs = np.full((1, num_classes), 0.005, dtype=np.float32)
    probs[0, 29] = 0.85
    mock_model.predict.return_value = probs

    class_indices = {i: f"Class_{i}" for i in range(num_classes)}
    class_indices[29] = "Tomato___Early_blight"
    service.set_mock_model(mock_model, class_indices)

    # Tomato leaf image
    tomato_img = Image.new("RGB", (224, 224), color=(40, 160, 50))
    result = service.predict_image(tomato_img)

    assert result["is_plant_supported"] is True
    assert result["is_disease_supported"] is True
    assert result["disease_code"] == "TOMATO_EARLY_BLIGHT"
    assert result["crop_name"] == "Tomato"

    # Must provide disease guidance
    recs = get_recommendations_for_prediction(result)
    assert recs["has_disease_guidance"] is True
    assert recs["disease_recommendation"] is not None
    assert "Alternaria solani" in recs["disease_recommendation"]["disease_name"]
