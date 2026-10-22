"""
Unit tests for Authentication Service (tests/test_auth_service.py)
Tests registration validation, Werkzeug scrypt hashing, duplicate detection,
and credential authentication with mocks.
"""

from unittest.mock import MagicMock, patch
import pytest
from werkzeug.security import check_password_hash

from services.auth_service import (
    authenticate_user,
    get_user_prediction_history,
    register_user,
)
from utils.auth import (
    is_safe_url,
    validate_email_format,
    validate_password_strength,
    validate_username,
)


def test_validate_username_rules():
    """Verify username length and character constraint validation."""
    assert validate_username("karthik")[0] is True
    assert validate_username("user_123")[0] is True
    assert validate_username("ab")[0] is False  # Too short (< 3)
    assert validate_username("bad space")[0] is False  # Space not allowed
    assert validate_username("user@name")[0] is False  # Special chars not allowed


def test_validate_email_format_rules():
    """Verify email format syntax validation."""
    assert validate_email_format("farmer@example.com")[0] is True
    assert validate_email_format("invalid-email")[0] is False
    assert validate_email_format("@missinguser.com")[0] is False


def test_validate_password_strength_rules():
    """Verify password minimum complexity rules."""
    assert validate_password_strength("FarmerPass123")[0] is True
    assert validate_password_strength("short1")[0] is False  # < 8 chars
    assert validate_password_strength("allletterspassword")[0] is False  # No digit
    assert validate_password_strength("1234567890")[0] is False  # No letter


def test_register_user_password_mismatch():
    """Verify registration fails when confirm_password does not match."""
    success, uid, msg = register_user(
        username="farmer_john",
        email="john@example.com",
        password="ValidPassword123",
        confirm_password="DifferentPassword456",
    )
    assert success is False
    assert uid is None
    assert "Passwords do not match" in msg


def test_register_user_duplicate_email():
    """Verify duplicate email registration is rejected."""
    existing_user = {"id": 1, "name": "other_user", "email": "john@example.com"}

    with patch("services.auth_service.check_db_connection", return_value=(True, "OK")), \
         patch("services.auth_service.execute_query", return_value=existing_user):

        success, uid, msg = register_user(
            username="new_farmer",
            email="john@example.com",
            password="ValidPassword123",
            confirm_password="ValidPassword123",
        )
        assert success is False
        assert uid is None
        assert "email address already exists" in msg


def test_register_user_successful():
    """Verify successful user registration hashes password and calls cursor."""
    mock_cursor = MagicMock()
    mock_cursor.lastrowid = 15

    with patch("services.auth_service.check_db_connection", return_value=(True, "OK")), \
         patch("services.auth_service.execute_query", return_value=None), \
         patch("services.auth_service.get_db_cursor") as mock_ctx:
        mock_ctx.return_value.__enter__.return_value = mock_cursor

        success, uid, msg = register_user(
            username="fresh_user",
            email="fresh@example.com",
            password="SecurePassword123",
            confirm_password="SecurePassword123",
        )

        assert success is True
        assert uid == 15
        assert "Registration successful" in msg
        assert mock_cursor.execute.called
        args, _ = mock_cursor.execute.call_args
        # Verify password stored is a hashed string, not raw text
        stored_hash = args[1][2]
        assert stored_hash != "SecurePassword123"
        assert check_password_hash(stored_hash, "SecurePassword123")


def test_authenticate_user_invalid_credentials():
    """Verify invalid credentials return a generic error message."""
    with patch("services.auth_service.check_db_connection", return_value=(True, "OK")), \
         patch("services.auth_service.execute_query", return_value=None):
        success, uinfo, msg = authenticate_user("unknown_user", "WrongPass123")
        assert success is False
        assert uinfo is None
        assert "Invalid username/email or password" in msg


def test_authenticate_user_valid():
    """Verify valid credentials authenticate successfully and return profile dict."""
    from werkzeug.security import generate_password_hash
    real_hash = generate_password_hash("CorrectPassword123")
    mock_row = {
        "id": 7,
        "name": "valid_user",
        "email": "user@agrotech.local",
        "password_hash": real_hash,
    }

    with patch("services.auth_service.check_db_connection", return_value=(True, "OK")), \
         patch("services.auth_service.execute_query", return_value=mock_row):
        success, uinfo, msg = authenticate_user("valid_user", "CorrectPassword123")
        assert success is True
        assert uinfo["id"] == 7
        assert uinfo["name"] == "valid_user"
        assert "password_hash" not in uinfo
