"""
Tests for Gemini Multimodal API Integration (tests/test_gemini_integration.py)
=============================================================================
Validates:
1. GeminiService initialization, configuration, and API key masking.
2. Decoupled architecture: CNN primary classifier + Gemini supplementary layer.
3. Image payload generation, resizing, downsampling, and corruption rejection.
4. API response parsing, qualitative certainty levels, and chemical dosage sanitization.
5. Error resilience: timeouts, network failures, 429 rate limits, and 503 high demand.
6. Flask endpoints: /health/gemini and /api/gemini/analyze/<id>.
7. Agronomic safety: Ensuring chemical recommendations are never fabricated.
"""

import json
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests
from PIL import Image

from app import create_app
from config import TestingConfig
from services.gemini_service import GeminiService, get_gemini_service
from services.prediction_service import run_gemini_analysis_for_record, run_prediction_pipeline


@pytest.fixture
def test_app():
    """Flask application fixture with TestingConfig."""
    app = create_app(TestingConfig)
    return app


@pytest.fixture
def test_client(test_app):
    """Flask test client fixture."""
    return test_app.test_client()


@pytest.fixture
def sample_leaf_image_path(tmp_path):
    """Generates a temporary valid leaf image file for testing."""
    img_path = tmp_path / "test_leaf.jpg"
    img = Image.new("RGB", (600, 600), color=(34, 139, 34))
    img.save(img_path, format="JPEG")
    return img_path


@pytest.fixture
def corrupt_image_path(tmp_path):
    """Generates a temporary corrupt image file."""
    bad_path = tmp_path / "corrupt_file.jpg"
    bad_path.write_bytes(b"NOT_A_VALID_IMAGE_DATA_12345")
    return bad_path


# ==============================================================================
# 1. Configuration & Security Tests
# ==============================================================================

def test_gemini_service_configured():
    """Tests that GeminiService detects when it is configured."""
    svc = GeminiService(api_key="TEST_API_KEY_123456789", model_name="gemini-3.5-flash-lite")
    assert svc.is_configured is True
    assert svc.model_name == "gemini-3.5-flash-lite"


def test_gemini_service_unconfigured():
    """Tests that GeminiService handles empty API keys gracefully."""
    svc = GeminiService(api_key="", model_name="gemini-3.5-flash-lite")
    assert svc.is_configured is False

    res = svc.analyze_leaf_image(Image.new("RGB", (100, 100)))
    assert res["status"] == "not_configured"
    assert "not configured" in res["visual_symptoms"].lower()


def test_api_key_never_exposed_in_health_endpoint(test_client):
    """Verifies that /health/gemini never exposes the API key."""
    resp = test_client.get("/health/gemini")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "status" in data
    assert "model" in data
    assert "role" in data
    assert "api_key" not in data
    assert "key" not in data
    # Ensure key string is not anywhere in raw response body
    raw_text = resp.data.decode("utf-8")
    assert "AQ." not in raw_text


# ==============================================================================
# 2. Image Processing & Downsampling Tests
# ==============================================================================

def test_prepare_image_payload_valid(sample_leaf_image_path):
    """Tests valid image downsampling and base64 encoding."""
    svc = GeminiService(api_key="TEST_KEY_123456789")
    b64_str, mime = svc._prepare_image_payload(sample_leaf_image_path, max_dimension=400)
    assert mime == "image/jpeg"
    assert isinstance(b64_str, str)
    assert len(b64_str) > 100


def test_prepare_image_payload_rejects_corrupt(corrupt_image_path):
    """Verifies that corrupt images are cleanly rejected by PIL verification."""
    svc = GeminiService(api_key="TEST_KEY_123456789")
    res = svc.analyze_leaf_image(corrupt_image_path)
    assert res["status"] == "invalid_image"
    assert res["is_leaf_clear"] is False
    assert "could not be parsed" in res["visual_symptoms"]


def test_prepare_image_missing_file(tmp_path):
    """Verifies that nonexistent file paths return file_not_found status."""
    svc = GeminiService(api_key="TEST_KEY_123456789")
    missing_path = tmp_path / "non_existent.jpg"
    res = svc.analyze_leaf_image(missing_path)
    assert res["status"] == "file_not_found"


# ==============================================================================
# 3. API Response Parsing & Sanitization Tests
# ==============================================================================

def test_analyze_leaf_image_successful_mock(sample_leaf_image_path):
    """Tests parsing of a well-formed Gemini JSON response."""
    mock_gemini_json = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": json.dumps({
                                "identified_plant_species": "Apple (Malus domestica)",
                                "visual_symptoms": "Necrotic circular lesions with dark borders (frogeye leaf spot pattern).",
                                "suspected_condition": "Black Rot (Botryosphaeria obtusa)",
                                "certainty_level": "High",
                                "is_leaf_clear": True,
                                "botanical_observations": "Characteristic fungal spots observed on mature leaf lamina.",
                                "recommended_next_step": "Prune mummified fruit and improve orchard airflow.",
                            })
                        }
                    ]
                }
            }
        ]
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_gemini_json

    svc = GeminiService(api_key="TEST_KEY_123456789")
    with patch.object(svc, "_execute_api_call", return_value=mock_resp):
        res = svc.analyze_leaf_image(sample_leaf_image_path)

    assert res["status"] == "success"
    assert res["identified_plant_species"] == "Apple (Malus domestica)"
    assert "Necrotic" in res["visual_symptoms"]
    assert res["certainty_level"] == "High"
    assert res["is_leaf_clear"] is True
    assert "CNN statistical classification" in res["disclaimer"]


def test_chemical_dosage_sanitization(sample_leaf_image_path):
    """Verifies that any attempted chemical dosage advice from LLM is sanitized."""
    mock_gemini_json = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": json.dumps({
                                "identified_plant_species": "Tomato (Solanum lycopersicum)",
                                "visual_symptoms": "Blight spots visible.",
                                "suspected_condition": "Early Blight",
                                "certainty_level": "Moderate",
                                "is_leaf_clear": True,
                                "botanical_observations": "Apply 2.5 grams per liter of copper fungicide immediately.",
                                "recommended_next_step": "Consult agricultural extension officer.",
                            })
                        }
                    ]
                }
            }
        ]
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_gemini_json

    svc = GeminiService(api_key="TEST_KEY_123456789")
    with patch.object(svc, "_execute_api_call", return_value=mock_resp):
        res = svc.analyze_leaf_image(sample_leaf_image_path)

    assert res["status"] == "success"
    # Dosage instruction should be stripped/sanitized
    assert "2.5 grams per liter" not in res["botanical_observations"]


# ==============================================================================
# 4. Error Resilience & Fallback Tests
# ==============================================================================

def test_api_timeout_handling(sample_leaf_image_path):
    """Verifies graceful fallback when Gemini API times out."""
    svc = GeminiService(api_key="TEST_KEY_123456789", timeout=1)
    with patch.object(svc, "_execute_api_call", side_effect=requests.exceptions.Timeout("Read timeout")):
        res = svc.analyze_leaf_image(sample_leaf_image_path)

    assert res["status"] == "unavailable"
    assert "CNN primary classifier remains fully operational" in res["disclaimer"]


def test_api_rate_limit_429(sample_leaf_image_path):
    """Verifies handling of HTTP 429 rate limit."""
    mock_resp = MagicMock()
    mock_resp.status_code = 429
    mock_resp.text = "Quota exceeded"

    svc = GeminiService(api_key="TEST_KEY_123456789")
    with patch.object(svc, "_execute_api_call", return_value=mock_resp):
        res = svc.analyze_leaf_image(sample_leaf_image_path)

    assert res["status"] == "rate_limited"
    assert "quota has been reached" in res["visual_symptoms"]


def test_api_high_demand_503(sample_leaf_image_path):
    """Verifies that 503 high demand tries fallback models gracefully."""
    mock_resp_503 = MagicMock()
    mock_resp_503.status_code = 503
    mock_resp_503.text = "High demand"

    svc = GeminiService(api_key="TEST_KEY_123456789")
    with patch.object(svc, "_execute_api_call", return_value=mock_resp_503):
        res = svc.analyze_leaf_image(sample_leaf_image_path)

    assert res["status"] == "unavailable"
    assert "Service Unavailable" in res["identified_plant_species"]


# ==============================================================================
# 5. Pipeline & Decoupled Probability Tests
# ==============================================================================

def test_no_probability_fabrication_in_pipeline(sample_leaf_image_path):
    """
    Confirms that CNN confidence and Gemini certainty remain completely separate.
    CNN confidence is a numerical percentage (e.g. 98.7%), while Gemini certainty
    is a qualitative string (e.g. 'High' or 'Moderate').
    """
    mock_gemini_json = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": json.dumps({
                                "identified_plant_species": "Apple (Malus domestica)",
                                "visual_symptoms": "Dark lesions present.",
                                "suspected_condition": "Apple Scab",
                                "certainty_level": "Moderate",
                                "is_leaf_clear": True,
                                "botanical_observations": "Lesions on upper cuticle.",
                                "recommended_next_step": "Sanitation.",
                            })
                        }
                    ]
                }
            }
        ]
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_gemini_json

    with patch("services.gemini_service.GeminiService._execute_api_call", return_value=mock_resp):
        pipeline_res = run_prediction_pipeline(
            image_source=sample_leaf_image_path,
            stored_relative_path="uploads/test_leaf.jpg",
            upload_id="test_up_123",
        )

    # 1. CNN confidence must be numerical
    assert isinstance(pipeline_res["confidence"], float)
    assert isinstance(pipeline_res["confidence_percent"], float)
    # 2. Gemini certainty must be qualitative string
    gemini_data = pipeline_res["gemini_analysis"]
    assert gemini_data["certainty_level"] == "Moderate"
    assert not isinstance(gemini_data["certainty_level"], (int, float))
    # 3. Source attributions must be explicit and distinct
    assert "MobileNetV2" in pipeline_res["model_version"]
    assert "Gemini" in gemini_data["source"]


# ==============================================================================
# 6. Endpoint Tests
# ==============================================================================

def test_api_gemini_analyze_invalid_id(test_client):
    """Tests validation on malformed prediction IDs."""
    resp = test_client.post("/api/gemini/analyze/bad_id_with_symbols!@#$")
    assert resp.status_code == 400
    assert resp.get_json()["success"] is False


def test_api_gemini_analyze_not_found(test_client):
    """Tests 404 response when prediction record does not exist."""
    resp = test_client.post("/api/gemini/analyze/nonexistent_id_99999")
    assert resp.status_code == 404
    assert resp.get_json()["success"] is False
