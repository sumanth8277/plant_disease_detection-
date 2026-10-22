"""
Route tests for Authentication & Dashboard (tests/test_auth_routes.py)
Tests registration, login, logout, and history views with mocks.
"""

from unittest.mock import MagicMock, patch
import pytest

from app import create_app
from config import TestingConfig


@pytest.fixture
def client():
    """Provides test client in testing mode (CSRF disabled for concise route testing)."""
    app = create_app(TestingConfig)
    with app.test_client() as test_client:
        yield test_client


def test_register_get_view(client):
    """Verify GET /register renders registration form."""
    response = client.get("/register")
    assert response.status_code == 200
    html = response.data.decode("utf-8")
    assert "Create an Agrotech Account" in html
    assert "Username" in html
    assert "Password" in html


def test_register_post_success_redirects_to_login(client):
    """Verify successful POST /register redirects to login."""
    with patch("routes.auth_routes.register_user", return_value=(True, 10, "Registration successful!")):
        response = client.post(
            "/register",
            data={
                "username": "testuser",
                "email": "test@example.com",
                "password": "Password123",
                "confirm_password": "Password123",
            },
            follow_redirects=False,
        )
        assert response.status_code == 302
        assert "/login" in response.headers["Location"]


def test_register_post_failure_returns_400(client):
    """Verify failed registration returns 400 with error feedback."""
    with patch("routes.auth_routes.register_user", return_value=(False, None, "Passwords do not match.")):
        response = client.post(
            "/register",
            data={
                "username": "testuser",
                "email": "test@example.com",
                "password": "Password123",
                "confirm_password": "WrongPassword456",
            },
        )
        assert response.status_code == 400
        html = response.data.decode("utf-8")
        assert "Passwords do not match." in html


def test_login_get_view(client):
    """Verify GET /login renders login form."""
    response = client.get("/login")
    assert response.status_code == 200
    html = response.data.decode("utf-8")
    assert "Log In to Agrotech" in html


def test_login_post_success_establishes_session(client):
    """Verify successful login stores user_id and redirects."""
    mock_uinfo = {"id": 25, "name": "karthik", "email": "karthik@example.com"}

    with patch("routes.auth_routes.authenticate_user", return_value=(True, mock_uinfo, "Login successful.")):
        response = client.post(
            "/login",
            data={"username_or_email": "karthik", "password": "MyPassword123"},
            follow_redirects=False,
        )
        assert response.status_code == 302
        with client.session_transaction() as sess:
            assert sess.get("user_id") == 25
            assert sess.get("user_name") == "karthik"


def test_login_post_failure_returns_401(client):
    """Verify invalid credentials return 401."""
    with patch("routes.auth_routes.authenticate_user", return_value=(False, None, "Invalid username/email or password.")):
        response = client.post(
            "/login",
            data={"username_or_email": "baduser", "password": "wrong"},
        )
        assert response.status_code == 401
        html = response.data.decode("utf-8")
        assert "Invalid username/email or password." in html


def test_logout_post_clears_session(client):
    """Verify POST /logout clears user session."""
    with client.session_transaction() as sess:
        sess["user_id"] = 99
        sess["user_name"] = "logout_user"

    response = client.post("/logout", follow_redirects=False)
    assert response.status_code == 302

    with client.session_transaction() as sess:
        assert "user_id" not in sess


def test_logout_get_rejected(client):
    """Verify GET /logout is not permitted (Method Not Allowed)."""
    response = client.get("/logout")
    assert response.status_code == 405


def test_history_requires_login(client):
    """Verify GET /history redirects unauthenticated users to login."""
    response = client.get("/history", follow_redirects=False)
    assert response.status_code == 302
    assert "/login" in response.headers["Location"]


def test_history_view_when_authenticated(client):
    """Verify GET /history displays diagnostic dashboard when logged in."""
    with client.session_transaction() as sess:
        sess["user_id"] = 10
        sess["user_name"] = "farmer_dan"

    mock_history_payload = {
        "predictions": [
            {
                "id": 1,
                "crop_name": "Tomato",
                "disease_name": "Early Blight",
                "confidence_percent": 94.5,
                "confidence": 0.945,
                "is_healthy": False,
                "model_version": "v1.0",
                "created_at": None,
            }
        ],
        "total": 1,
        "page": 1,
        "per_page": 10,
        "total_pages": 1,
    }

    with patch("routes.auth_routes.get_user_prediction_history", return_value=mock_history_payload):
        response = client.get("/history")
        assert response.status_code == 200
        html = response.data.decode("utf-8")
        assert "Diagnostic History Dashboard" in html
        assert "Tomato" in html
        assert "Early Blight" in html
