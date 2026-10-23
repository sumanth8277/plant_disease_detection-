"""
Tests for Unsupported Plant Handling, Out-of-Distribution (OOD) Detection,
and Recommendation Protection (tests/test_unsupported_plants.py).
Ensures uncataloged plants like Neem cannot trigger incorrect crop diagnoses
or harmful pesticide/fertilizer recommendations.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from config import Config
from services.model_service import ModelService, get_model_service
from services.recommendation_service import get_recommendations_for_prediction


def test_neem_is_not_in_training_classes():
    """Verify that Neem is honestly verified as NOT present in the model's 38 classes."""
    class_indices_path = Path("models/class_indices.json") if Path("models/class_indices.json").exists() else Path("model/class_indices.json")
    assert class_indices_path.exists(), "class_indices.json must exist"

    with open(class_indices_path, "r", encoding="utf-8") as f:
        class_indices = json.load(f)

    assert len(class_indices) == 38
    neem_matches = [cls for cls in class_indices if "neem" in cls.lower()]
    assert len(neem_matches) == 0, "Neem must not be present in the 38 PlantVillage classes"


def test_class_mapping_consistency_and_dimensions():
    """Verify that class_indices.json matches disease_class_mapping.json exactly."""
    indices_path = Path("models/class_indices.json") if Path("models/class_indices.json").exists() else Path("model/class_indices.json")
    mapping_path = Path("models/disease_class_mapping.json") if Path("models/disease_class_mapping.json").exists() else Path("model/disease_class_mapping.json")

    with open(indices_path, "r", encoding="utf-8") as f:
        class_indices = json.load(f)

    with open(mapping_path, "r", encoding="utf-8") as f:
        disease_mapping = json.load(f)["mappings"]

    # Every class in class_indices must have a valid mapping entry
    for class_name in class_indices.keys():
        assert class_name in disease_mapping, f"Class {class_name} missing from disease_class_mapping.json"
        entry = disease_mapping[class_name]
        assert "crop_name" in entry
        assert "disease_name" in entry
        assert "disease_code" in entry


def test_model_output_dimension_matches_class_indices():
    """Verify that the loaded Keras model output matches len(class_indices)."""
    ms = get_model_service()
    ready, _ = ms.is_model_ready()
    assert ready is True, "Model must be ready"
    assert ms._model.output_shape[-1] == len(ms._class_indices) == 38


def test_ood_diffuse_probabilities_flagged_as_unsupported():
    """
    Verify that an input producing diffuse probabilities (characteristic of OOD leaves like Neem)
    is flagged as unsupported and does NOT assign a specific disease name.
    """
    ms = ModelService()
    ms._is_loaded = True
    ms._model = MagicMock()
    # Mock 38 classes where the top probability is low (12.5%, similar to Neem output)
    mock_probs = np.full(38, 0.02)
    mock_probs[25] = 0.125  # Index 25 is Squash___Powdery_mildew
    mock_probs = mock_probs / np.sum(mock_probs)
    ms._model.predict.return_value = np.array([mock_probs], dtype=np.float32)

    indices_path = Path("models/class_indices.json") if Path("models/class_indices.json").exists() else Path("model/class_indices.json")
    with open(indices_path, "r", encoding="utf-8") as f:
        indices = json.load(f)
    ms._class_indices = {v: k for k, v in indices.items()}

    with patch("services.model_service.load_and_preprocess_image", return_value=np.zeros((224, 224, 3))):
        result = ms.predict_image("dummy_path.png")

    assert result["is_supported"] is False
    assert result["is_inconclusive"] is True
    assert result["crop_name"] == "Unsupported / Unknown Plant"
    assert result["disease_name"] == "Unsupported or Inconclusive Specimen"
    assert result["disease_code"] == "UNSUPPORTED_SPECIMEN"
    assert "Squash" not in result["crop_name"]
    assert "Powdery Mildew" not in result["disease_name"]
    assert result["confidence_percent"] < 40.0


def test_recommendation_protection_withholds_pesticides_for_unsupported_plant():
    """
    Verify that get_recommendations_for_prediction NEVER returns squash pesticides
    or crop fertilizers for an unsupported specimen.
    """
    unsupported_pred = {
        "crop_name": "Unsupported / Unknown Plant",
        "disease_name": "Unsupported or Inconclusive Specimen",
        "disease_code": "UNSUPPORTED_SPECIMEN",
        "is_healthy": False,
        "is_supported": False,
        "is_inconclusive": True,
        "candidate_raw_class": "Squash___Powdery_mildew",
    }

    rec_bundle = get_recommendations_for_prediction(unsupported_pred)

    assert rec_bundle["is_supported"] is False
    assert rec_bundle["has_disease_guidance"] is False
    assert rec_bundle["disease_recommendation"] is None
    assert rec_bundle["has_fertilizer_guidance"] is False
    assert rec_bundle["fertilizer_recommendations"] == []
    assert "Squash" not in rec_bundle["crop_name"]
    assert "Neem" in rec_bundle["guidance_notice"]
    assert "withheld" in rec_bundle["guidance_notice"].lower()


def test_supported_plant_receives_valid_recommendations():
    """
    Verify that an actual supported crop prediction (e.g. Tomato Early Blight)
    with high confidence continues to receive full treatments and fertilizer schedules.
    """
    supported_pred = {
        "crop_name": "Tomato",
        "disease_name": "Early Blight (Alternaria solani)",
        "disease_code": "TOMATO_EARLY_BLIGHT",
        "is_healthy": False,
        "is_supported": True,
        "is_inconclusive": False,
        "confidence": 0.92,
        "confidence_percent": 92.0,
    }

    rec_bundle = get_recommendations_for_prediction(supported_pred)

    assert rec_bundle["is_supported"] is True
    assert rec_bundle["has_disease_guidance"] is True
    assert rec_bundle["disease_recommendation"] is not None
    assert "symptoms" in rec_bundle["disease_recommendation"]
    assert "treatment_guidance" in rec_bundle["disease_recommendation"]
    assert rec_bundle["has_fertilizer_guidance"] is True
    assert len(rec_bundle["fertilizer_recommendations"]) > 0


def test_actual_uploaded_neem_image_behavior():
    """
    Verify the actual neem leaf uploaded by the user is safely diagnosed as
    unsupported rather than asserting Squash Powdery Mildew.
    """
    neem_path = Path("static/uploads/3b4b2e90f91c4a99b79ea43aa1e16e82.png")
    if not neem_path.exists():
        pytest.skip("User neem sample not present in uploads")

    ms = get_model_service()
    res = ms.predict_image(neem_path)

    # Must be flagged unsupported for disease detection
    assert res["is_supported"] is False
    assert res["is_inconclusive"] is True
    assert res["crop_name"] in ["Neem (Azadirachta indica)", "Unsupported / Unknown Plant"]
    assert "Squash" not in res["crop_name"]
    assert res["confidence_percent"] < 40.0

    # Must not yield squash treatments
    recs = get_recommendations_for_prediction(res)
    assert recs["has_disease_guidance"] is False
    assert recs["has_fertilizer_guidance"] is False
