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


def test_get_commercial_supplement_recommendation():
    """Verify commercial treatment and fertilizer catalog retrieval."""
    diseased_pred = {
        "crop_name": "Tomato",
        "disease_code": "TOMATO_EARLY_BLIGHT",
        "disease_name": "Early Blight (Alternaria solani)",
        "raw_class": "Tomato___Early_blight",
        "is_healthy": False,
    }

    with patch("services.recommendation_service.check_db_connection", return_value=(False, "Offline")):
        bundle = get_recommendations_for_prediction(diseased_pred)
        assert bundle["has_commercial_supplement"] is True
        supp = bundle["commercial_supplement"]
        assert supp is not None
        assert "NATIVO" in supp["recommended_supplement_name"] or "Fungicide" in supp["supplement_category"]
        assert "buy_or_source_link" in supp

    # Test Neem withholding
    neem_pred = {
        "crop_name": "Neem",
        "disease_code": "NEEM_DISEASE_UNSUPPORTED",
        "is_neem": True,
        "is_supported": False,
    }
    neem_bundle = get_recommendations_for_prediction(neem_pred)
    assert neem_bundle["has_commercial_supplement"] is False
    assert neem_bundle["commercial_supplement"] is None
    assert neem_bundle["has_verified_indian_treatment"] is False
    assert neem_bundle["verified_indian_treatment"] is None


def test_get_verified_indian_agronomic_treatment_found():
    """Verify verified Indian CIBRC/ICAR treatment record retrieval."""
    from services.recommendation_service import get_verified_indian_agronomic_treatment

    pred_data = {
        "crop_name": "Tomato",
        "disease_code": "TOMATO_EARLY_BLIGHT",
        "disease_name": "Early Blight (Alternaria solani)",
        "raw_class": "Tomato___Early_blight",
        "is_healthy": False,
        "is_supported": True,
    }

    result = get_verified_indian_agronomic_treatment(pred_data)
    assert result is not None
    assert result["crop"] == "Tomato"
    assert "non_chemical_management" in result
    pesticide = result["verified_pesticide_information"]
    assert "Mancozeb" in pesticide["approved_active_ingredient"] or "Mancozeb" in pesticide["cibrc_registered_formulation"]
    assert "pre_harvest_interval_phi" in pesticide
    assert "source_url" in result
    assert "safety_notes" in result


def test_get_verified_indian_agronomic_treatment_viral_fallback():
    """Verify systemic viral infection returns explicit 'Verified treatment information unavailable' for chemical doses."""
    from services.recommendation_service import get_verified_indian_agronomic_treatment

    pred_data = {
        "crop_name": "Tomato",
        "disease_code": "TOMATO_MOSAIC_VIRUS",
        "disease_name": "Tomato Mosaic Virus",
        "raw_class": "Tomato___Tomato_mosaic_virus",
        "is_healthy": False,
        "is_supported": True,
    }

    result = get_verified_indian_agronomic_treatment(pred_data)
    assert result is not None
    pesticide = result["verified_pesticide_information"]
    assert "Verified treatment information unavailable" in pesticide["recommended_dosage"]


def test_get_verified_indian_agronomic_treatment_withholding():
    """Verify treatment is withheld for unverified, unsupported, or neem specimens."""
    from services.recommendation_service import get_verified_indian_agronomic_treatment

    unsupported_pred = {
        "crop_name": "Unknown",
        "disease_code": "UNSUPPORTED_SPECIMEN",
        "is_supported": False,
    }
    assert get_verified_indian_agronomic_treatment(unsupported_pred) is None

    neem_pred = {
        "crop_name": "Neem",
        "disease_code": "NEEM_DISEASE_UNSUPPORTED",
        "is_neem": True,
        "is_supported": False,
    }
    assert get_verified_indian_agronomic_treatment(neem_pred) is None

