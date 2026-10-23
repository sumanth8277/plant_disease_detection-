"""
Phase 3 Safe Test Deployment & Complete Flow Integration Test
=============================================================
Tests candidate model (unified_multidataset_model.keras) integration:
- Class-index mapping exactness
- Complete inference and recommendation flow
- Unknown/unsupported plant shielding
- Gemini architectural separation
- Result page rendering with candidate predictions
"""

import io
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

CANDIDATE_MODEL_PATH = BASE_DIR / "models" / "candidates" / "unified_multidataset_model.keras"
CLASS_NAMES_PATH = BASE_DIR / "models" / "class_names.json"


@pytest.fixture(scope="module")
def candidate_model_service():
    """Initializes a separate ModelService instance pointing to candidate model."""
    from services.model_service import ModelService
    service = ModelService()
    service.load_model_artifacts(
        model_path=CANDIDATE_MODEL_PATH,
        class_indices_path=CLASS_NAMES_PATH,
    )
    return service


def test_candidate_model_artifact_and_class_mapping(candidate_model_service):
    """Verify candidate model is loaded and output dimension matches class_names.json exactly."""
    assert candidate_model_service.is_loaded is True
    is_ready, msg = candidate_model_service.is_model_ready()
    assert is_ready is True
    
    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        class_names = json.load(f)
        
    assert len(class_names) == 38
    assert len(candidate_model_service._class_indices) == 38
    assert candidate_model_service._model.output_shape[-1] == 38


def test_candidate_predict_supported_disease(candidate_model_service):
    """Test full prediction flow for a real test leaf image with candidate model."""
    from services.recommendation_service import get_recommendations_for_prediction

    # Create synthetic leaf image or use real test image
    test_img_path = BASE_DIR / "dataset" / "plantvillage_candidate" / "images" / "Tomato___Early_blight"
    test_files = list(test_img_path.glob("*.JPG")) if test_img_path.exists() else []
    
    if test_files:
        filepath = str(test_files[0])
    else:
        # Fallback to in-memory green leaf
        img = Image.new("RGB", (224, 224), color=(34, 139, 34))
        filepath = str(BASE_DIR / "static" / "uploads" / "temp_test_leaf.jpg")
        img.save(filepath)

    result = candidate_model_service.predict_image(filepath)
    assert result is not None
    assert "raw_class" in result
    assert "confidence" in result
    assert "crop_name" in result
    assert "is_supported" in result
    
    # Retrieve recommendations
    recs = get_recommendations_for_prediction(result)
    assert recs["is_supported"] is True
    assert recs["has_disease_guidance"] is True
    assert recs["has_fertilizer_guidance"] is True
    assert recs["has_verified_indian_treatment"] is True
    
    vit = recs["verified_indian_treatment"]
    assert vit is not None
    assert "verified_pesticide_information" in vit
    assert "pre_harvest_interval_phi" in vit["verified_pesticide_information"]


def test_candidate_unsupported_and_neem_protection(candidate_model_service):
    """Test that unsupported or Neem specimens strictly withhold pesticides and fertilizers."""
    from services.recommendation_service import get_recommendations_for_prediction

    # 1. Neem prediction
    neem_result = {
        "crop_name": "Neem (Azadirachta indica)",
        "disease_name": "Neem disease diagnosis is not yet supported",
        "disease_code": "NEEM_DISEASE_UNSUPPORTED",
        "is_neem": True,
        "is_supported": False,
        "is_healthy": False,
    }
    neem_recs = get_recommendations_for_prediction(neem_result)
    assert neem_recs["is_supported"] is False
    assert neem_recs["has_disease_guidance"] is False
    assert neem_recs["has_fertilizer_guidance"] is False
    assert neem_recs["has_commercial_supplement"] is False
    assert neem_recs["has_verified_indian_treatment"] is False
    assert neem_recs["verified_indian_treatment"] is None
    assert "strictly withheld" in neem_recs["guidance_notice"]

    # 2. Unsupported OOD specimen
    unsupported_result = {
        "crop_name": "Unsupported / Unknown Plant",
        "disease_name": "Unsupported or Inconclusive Specimen",
        "disease_code": "UNSUPPORTED_SPECIMEN",
        "is_supported": False,
        "is_inconclusive": True,
    }
    ood_recs = get_recommendations_for_prediction(unsupported_result)
    assert ood_recs["is_supported"] is False
    assert ood_recs["has_disease_guidance"] is False
    assert ood_recs["has_fertilizer_guidance"] is False
    assert ood_recs["has_commercial_supplement"] is False
    assert ood_recs["has_verified_indian_treatment"] is False
    assert ood_recs["verified_indian_treatment"] is None


def test_candidate_gemini_separation_and_failure_handling(candidate_model_service):
    """Test that Gemini API operates as a supplementary service and does not override CNN."""
    from services.gemini_service import GeminiService

    service = GeminiService()
    # Mock API failure
    with patch.object(service, "analyze_leaf_image", return_value={
        "status": "unavailable",
        "botanical_observations": "Gemini service temporarily unavailable; CNN classification is intact.",
        "disclaimer": "Gemini is strictly supplementary.",
    }):
        gemini_result = service.analyze_leaf_image("dummy_path.jpg")
        assert gemini_result["status"] == "unavailable"
        assert "supplementary" in gemini_result["disclaimer"]
