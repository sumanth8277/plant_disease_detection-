"""
Unit tests for RecommendationService (tests/test_recommendations.py)
Tests disease and fertilizer recommendation lookup, parameter binding,
missing record handling, and healthy foliage routing without requiring live MySQL.
"""

from unittest.mock import MagicMock, patch
import pytest

from services.recommendation_service import (
    get_disease_recommendation,
    get_fertilizer_recommendations,
    get_recommendations_for_prediction,
)


def test_get_disease_recommendation_by_code_found():
    """Verify exact disease_code lookup returns matching database record."""
    mock_row = {
        "id": 1,
        "disease_code": "TOMATO_EARLY_BLIGHT",
        "crop_name": "Tomato",
        "disease_name": "Early Blight (Alternaria solani)",
        "symptoms": "Dark brown target-like spots",
        "treatment_guidance": "Prune lower leaves",
        "prevention_guidance": "Crop rotation",
        "source_reference": "ICAR-IIHR",
        "review_status": "educational_only",
    }

    with patch("services.recommendation_service.check_db_connection", return_value=(True, "OK")), \
         patch("services.recommendation_service.execute_query", return_value=mock_row) as mock_exec:
        result = get_disease_recommendation("Tomato", disease_code="TOMATO_EARLY_BLIGHT")
        assert result is not None
        assert result["disease_code"] == "TOMATO_EARLY_BLIGHT"
        assert result["crop_name"] == "Tomato"
        # Ensure parameterized query was used
        assert mock_exec.called
        args, kwargs = mock_exec.call_args
        assert "%s" in args[0]
        assert ("TOMATO_EARLY_BLIGHT",) == args[1]


def test_get_disease_recommendation_missing_returns_none():
    """Verify that an uncataloged disease returns None safely."""
    with patch("services.recommendation_service.check_db_connection", return_value=(True, "OK")), \
         patch("services.recommendation_service.execute_query", return_value=None):
        result = get_disease_recommendation("UnknownCrop", disease_code="UNKNOWN_DISEASE")
        assert result is None


def test_get_fertilizer_recommendations_crop_match():
    """Verify crop-specific fertilizer records are retrieved."""
    mock_rows = [
        {
            "id": 1,
            "crop_name": "Tomato",
            "growth_stage": "Early Vegetative",
            "soil_condition": "Standard sandy-loam",
            "recommendation_text": "Balanced starter fertilizer (NPK 10-10-10)",
            "source_reference": "ICAR Guide",
            "review_status": "educational_only",
        }
    ]

    with patch("services.recommendation_service.check_db_connection", return_value=(True, "OK")), \
         patch("services.recommendation_service.execute_query", return_value=mock_rows):
        results = get_fertilizer_recommendations("Tomato")
        assert len(results) == 1
        assert results[0]["crop_name"] == "Tomato"
        assert "NPK 10-10-10" in results[0]["recommendation_text"]


def test_get_recommendations_for_healthy_foliage():
    """Verify healthy foliage returns preventive cultural care and no chemical fungicides."""
    pred_data = {
        "crop_name": "Tomato",
        "disease_code": "TOMATO_HEALTHY",
        "disease_name": "Healthy Foliage",
        "is_healthy": True,
    }

    with patch("services.recommendation_service.check_db_connection", return_value=(True, "OK")), \
         patch("services.recommendation_service.execute_query", return_value=[]):
        bundle = get_recommendations_for_prediction(pred_data)
        assert bundle["is_healthy"] is True
        assert bundle["has_disease_guidance"] is True
        rec = bundle["disease_recommendation"]
        assert "No therapeutic chemical treatments" in rec["treatment_guidance"]
        assert "Balanced irrigation" in rec["prevention_guidance"] or "Continue balanced" in rec["prevention_guidance"]
