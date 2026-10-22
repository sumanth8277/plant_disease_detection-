"""
Integration tests for Phase 4 Flask routes and results workflow (tests/test_phase4_routes.py).
Tests model health checks, upload inference pipeline, and results rendering with mocks.
"""

import io
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from PIL import Image

from app import create_app
from config import TestingConfig
from services.model_service import get_model_service


@pytest.fixture
def client(tmp_path):
    """Provides a Flask test client configured with a temporary upload folder."""
    TestingConfig.UPLOAD_FOLDER = tmp_path / "uploads"
    TestingConfig.UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
    app = create_app(TestingConfig)
    with app.test_client() as test_client:
        yield test_client


def create_sample_png_bytes(width=100, height=100, color=(0, 200, 0)):
    """Helper creating raw bytes of a valid PNG image."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


def test_health_model_endpoint_unavailable(client):
    """Verify GET /health/model returns 503 when model weights are not loaded."""
    with patch.object(get_model_service(), "is_model_ready", return_value=(False, "Weights not found")):
        response = client.get("/health/model")
        assert response.status_code == 503
        data = response.get_json()
        assert data["status"] == "unavailable"
        assert "Weights not found" in data["message"]


def test_health_model_endpoint_ready(client):
    """Verify GET /health/model returns 200 when model is loaded and ready."""
    with patch.object(get_model_service(), "is_model_ready", return_value=(True, "Model ready")), \
         patch.object(get_model_service(), "get_model_metadata", return_value={"num_classes": 38, "metadata": {"model_version": "v1.0"}}):
        response = client.get("/health/model")
        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "ready"
        assert data["num_classes"] == 38


def test_upload_post_when_model_ready_redirects_to_results(client):
    """Verify that uploading a valid leaf image when model is ready runs inference and redirects to /results."""
    img_bytes = create_sample_png_bytes()

    mock_pipeline_res = {
        "success": True,
        "crop_name": "Tomato",
        "disease_name": "Early Blight",
        "disease_code": "TOMATO_EARLY_BLIGHT",
        "is_healthy": False,
        "confidence": 0.9421,
        "confidence_percent": 94.21,
        "prediction_id": 101,
        "upload_id": "upload_abc123",
        "stored_relative_path": "uploads/test.png",
        "persisted": True,
        "persist_message": "Persisted successfully",
    }

    with patch.object(get_model_service(), "is_model_ready", return_value=(True, "Ready")), \
         patch("routes.main_routes.run_prediction_pipeline", return_value=mock_pipeline_res):

        response = client.post(
            "/upload",
            data={"leaf_image": (img_bytes, "healthy_leaf.png")},
            content_type="multipart/form-data",
            follow_redirects=False,
        )

        assert response.status_code == 302
        assert "/results/101" in response.headers["Location"]


def test_upload_post_when_model_not_ready_redirects_to_upload_success(client):
    """Verify that uploading an image when model is unavailable still saves upload and gives informative notice."""
    img_bytes = create_sample_png_bytes()

    with patch.object(get_model_service(), "is_model_ready", return_value=(False, "Weights not trained yet")):
        response = client.post(
            "/upload",
            data={"leaf_image": (img_bytes, "leaf.png")},
            content_type="multipart/form-data",
            follow_redirects=False,
        )

        assert response.status_code == 302
        assert "/upload/success/" in response.headers["Location"]


def test_view_result_page_renders_successfully(client):
    """Verify GET /results/<id> renders diagnosis, confidence, and treatment sections."""
    mock_result_data = {
        "crop_name": "Tomato",
        "disease_name": "Early Blight (Alternaria solani)",
        "disease_code": "TOMATO_EARLY_BLIGHT",
        "is_healthy": False,
        "confidence": 0.952,
        "confidence_percent": 95.2,
        "image_path": "uploads/sample.png",
        "prediction_id": 101,
        "upload_id": "upload_101",
        "model_version": "MobileNetV2-v1.0",
        "persisted": True,
        "persist_message": "Saved to database",
        "has_disease_guidance": True,
        "disease_recommendation": {
            "symptoms": "Dark brown target-like concentric rings",
            "treatment_guidance": "Prune lower infected foliage",
            "prevention_guidance": "Crop rotation with non-solanaceous crops",
            "source_reference": "ICAR-IIHR Advisory",
            "review_status": "educational_only",
        },
        "has_fertilizer_guidance": True,
        "fertilizer_recommendations": [
            {
                "growth_stage": "Early Vegetative",
                "soil_condition": "Standard sandy-loam",
                "recommendation_text": "Balanced starter fertilizer NPK 10-10-10",
                "source_reference": "ICAR Guide",
            }
        ],
        "top_predictions": [
            {
                "crop_name": "Tomato",
                "disease_name": "Early Blight",
                "confidence_percent": 95.2,
            }
        ],
    }

    with patch("routes.main_routes.get_prediction_record", return_value=(mock_result_data, False)):
        response = client.get("/results/101")
        assert response.status_code == 200
        html = response.data.decode("utf-8")
        assert "Early Blight (Alternaria solani)" in html
        assert "Tomato" in html
        assert "95.2%" in html
        assert "Prune lower infected foliage" in html
        assert "Balanced starter fertilizer" in html
        assert "Responsible Agronomic Use Notice" in html


def test_view_result_page_not_found(client):
    """Verify GET /results/<unknown_id> returns 404."""
    with patch("routes.main_routes.get_prediction_record", return_value=(None, False)):
        response = client.get("/results/999999")
        assert response.status_code == 404
        html = response.data.decode("utf-8")
        assert "Diagnosis Result Not Found" in html
