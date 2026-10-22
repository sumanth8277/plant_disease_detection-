"""
Unit and integration tests for Flask application routes and error handling (tests/test_app.py).
"""

from unittest.mock import MagicMock, patch
import pytest
from app import create_app
from config import TestingConfig
from database.db import sanitize_db_error
from mysql.connector import Error, errorcode


@pytest.fixture
def client():
    """Provides a test client with TestingConfig."""
    app = create_app(TestingConfig)
    with app.test_client() as test_client:
        yield test_client


def test_app_creation():
    """Verify application factory initializes correctly in testing mode."""
    app = create_app(TestingConfig)
    assert app.config["TESTING"] is True
    assert app.config["DB_NAME"] == "plant_disease_test_db"


def test_homepage_response(client):
    """Verify GET / returns 200 OK and renders Agrotech branding, title, diagnostics, and upload area."""
    response = client.get("/")
    assert response.status_code == 200
    html_content = response.data.decode("utf-8")
    assert "Agrotech" in html_content
    assert "Plant Disease Detection" in html_content
    assert "How the Diagnostic System Works" in html_content
    assert "Upload" in html_content


def test_health_endpoint(client):
    """Verify GET /health returns 200 OK and safe JSON payload without exposing credentials."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "healthy"
    assert "Plant Disease Detection" in data["service"]
    # Ensure sensitive configuration keys are not leaked
    assert "DB_PASSWORD" not in data
    assert "SECRET_KEY" not in data


def test_health_db_endpoint_live(client):
    """
    Verify GET /health/db returns structured diagnostic JSON.
    Returns 200 if MySQL is active or 503 if MySQL is unavailable.
    """
    response = client.get("/health/db")
    assert response.status_code in (200, 503)
    data = response.get_json()
    assert "status" in data
    assert "database" in data
    assert "message" in data
    # Ensure no credentials are leaked in response
    assert "password" not in str(data).lower() or "password" in data.get("message", "").lower()


def test_health_db_endpoint_mock_healthy(client):
    """Verify GET /health/db returns 200 when database check succeeds."""
    with patch("routes.main_routes.check_db_connection", return_value=(True, "MySQL connection healthy.")):
        response = client.get("/health/db")
        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "healthy"


def test_health_db_endpoint_mock_failure(client):
    """Verify GET /health/db returns 503 when database check fails."""
    with patch("routes.main_routes.check_db_connection", return_value=(False, "Connection refused at 127.0.0.1:3306")):
        response = client.get("/health/db")
        assert response.status_code == 503
        data = response.get_json()
        assert data["status"] == "unavailable"


def test_upload_page_get(client):
    """Verify GET /upload renders dedicated upload page."""
    response = client.get("/upload")
    assert response.status_code == 200
    html = response.data.decode("utf-8")
    assert "Upload Plant Leaf Image" in html


def test_controlled_serve_upload_invalid_filename_404(client):
    """Verify controlled route /uploads/<filename> rejects invalid or malicious filenames."""
    # Invalid extension
    res = client.get("/uploads/malicious.php")
    assert res.status_code == 404

    # Path traversal attempt
    res = client.get("/uploads/..%2f..%2fconfig.py")
    assert res.status_code == 404

    # Non-existent safe filename
    res = client.get("/uploads/00000000000000000000000000000000.jpg")
    assert res.status_code == 404


def test_db_error_sanitization():
    """Verify that sanitize_db_error masks sensitive details."""
    # Test Access Denied
    err_access = Error(errno=errorcode.ER_ACCESS_DENIED_ERROR, msg="Access denied for user 'admin'@'localhost'")
    msg = sanitize_db_error(err_access)
    assert "invalid username or password" in msg
    assert "admin" not in msg

    # Test Generic Exception
    err_generic = RuntimeError("Internal memory fault with secret=xyz")
    msg_gen = sanitize_db_error(err_generic)
    assert "secret" not in msg_gen
    assert "An unexpected database connection error occurred." == msg_gen
