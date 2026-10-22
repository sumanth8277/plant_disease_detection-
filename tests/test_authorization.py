"""
Authorization & Access Control Security Tests (tests/test_authorization.py)
Tests private prediction isolation, open-redirect prevention, and ownership verification.
"""

from unittest.mock import patch
import pytest

from app import create_app
from config import TestingConfig
from utils.auth import is_safe_url


@pytest.fixture
def client():
    """Provides test client in testing mode."""
    app = create_app(TestingConfig)
    with app.test_client() as test_client:
        yield test_client


def test_is_safe_url_prevention():
    """Verify open-redirect protection accepts internal paths and rejects external domains."""
    app = create_app(TestingConfig)
    with app.test_request_context("http://localhost:5000/login"):
        # Safe internal URLs
        assert is_safe_url("/history") is True
        assert is_safe_url("/upload") is True
        assert is_safe_url("http://localhost:5000/history") is True

        # Unsafe external URLs (open redirect attacks)
        assert is_safe_url("https://malicious-site.com") is False
        assert is_safe_url("//evil.com/hack") is False
        assert is_safe_url("http://attacker.com/steal") is False
        assert is_safe_url(None) is False
        assert is_safe_url("") is False


def test_user_cannot_access_another_users_prediction(client):
    """Verify that User A is blocked with 403 from viewing User B's prediction."""
    # User 1 is logged in
    with client.session_transaction() as sess:
        sess["user_id"] = 1
        sess["user_name"] = "user_one"

    # Prediction 50 belongs to User 2
    mock_prediction = {
        "id": 50,
        "user_id": 2,  # Owned by user 2
        "crop_name": "Corn",
        "disease_name": "Common Rust",
        "confidence": 0.88,
        "confidence_percent": 88.0,
    }

    # get_prediction_record returns (None, True) when user_id does not match
    with patch("routes.main_routes.get_prediction_record", return_value=(None, True)):
        response = client.get("/results/50")
        assert response.status_code == 403
        html = response.data.decode("utf-8")
        assert "Access Denied" in html


def test_user_can_access_own_prediction(client):
    """Verify that User A can successfully view their own prediction."""
    with client.session_transaction() as sess:
        sess["user_id"] = 1
        sess["user_name"] = "user_one"

    mock_prediction = {
        "id": 51,
        "user_id": 1,  # Owned by user 1
        "crop_name": "Tomato",
        "disease_name": "Early Blight",
        "disease_code": "TOMATO_EARLY_BLIGHT",
        "confidence": 0.92,
        "confidence_percent": 92.0,
        "is_healthy": False,
        "image_path": "uploads/leaf.png",
        "has_disease_guidance": False,
        "has_fertilizer_guidance": False,
    }

    with patch("routes.main_routes.get_prediction_record", return_value=(mock_prediction, False)):
        response = client.get("/results/51")
        assert response.status_code == 200
        html = response.data.decode("utf-8")
        assert "Tomato" in html
        assert "Early Blight" in html


def test_guest_can_access_unowned_prediction(client):
    """Verify that guest unowned predictions (user_id=None) are accessible by prediction ID."""
    mock_prediction = {
        "id": 52,
        "user_id": None,  # Guest prediction
        "crop_name": "Potato",
        "disease_name": "Late Blight",
        "disease_code": "POTATO_LATE_BLIGHT",
        "confidence": 0.91,
        "confidence_percent": 91.0,
        "is_healthy": False,
        "image_path": "uploads/leaf2.png",
        "has_disease_guidance": False,
        "has_fertilizer_guidance": False,
    }

    with patch("routes.main_routes.get_prediction_record", return_value=(mock_prediction, False)):
        response = client.get("/results/52")
        assert response.status_code == 200
        html = response.data.decode("utf-8")
        assert "Potato" in html
        assert "Late Blight" in html
