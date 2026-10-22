"""
User Authentication & Account Service (services/auth_service.py)
Handles user registration, Werkzeug password hashing, secure credential verification,
account lookups, and user-specific prediction history pagination.
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

from werkzeug.security import check_password_hash, generate_password_hash

from database.db import check_db_connection, execute_query, get_db_cursor
from utils.auth import (
    validate_email_format,
    validate_password_strength,
    validate_username,
)

logger = logging.getLogger(__name__)


def register_user(
    username: str, email: str, password: str, confirm_password: str
) -> Tuple[bool, Optional[int], str]:
    """
    Validates registration fields, hashes password with Werkzeug,
    checks for unique username and email, and inserts new record into MySQL 'users' table.

    Returns:
        (success: bool, user_id: Optional[int], message: str)
    """
    # 1. Validation checks
    u_ok, u_err = validate_username(username)
    if not u_ok:
        return False, None, u_err or "Invalid username."

    e_ok, e_err = validate_email_format(email)
    if not e_ok:
        return False, None, e_err or "Invalid email address."

    if password != confirm_password:
        return False, None, "Passwords do not match. Please re-enter your password."

    p_ok, p_err = validate_password_strength(password)
    if not p_ok:
        return False, None, p_err or "Password does not meet security requirements."

    # 2. Database connectivity
    is_connected, db_msg = check_db_connection()
    if not is_connected:
        logger.error("Database unavailable during registration: %s", db_msg)
        return False, None, "Database service is currently unavailable. Please try again later."

    clean_username = username.strip()
    clean_email = email.strip().lower()

    # 3. Duplicate check
    check_sql = """
        SELECT id, name, email FROM users
        WHERE email = %s OR name = %s
        LIMIT 1;
    """
    try:
        existing = execute_query(check_sql, (clean_email, clean_username), fetchone=True)
        if existing:
            if existing["email"].lower() == clean_email:
                return False, None, "An account with this email address already exists."
            return False, None, "This username is already taken. Please choose another username."
    except Exception as err:
        logger.error("Error checking duplicate user: %s", err)
        return False, None, "Unable to verify user availability. Please try again."

    # 4. Hash password securely
    hashed_password = generate_password_hash(password, method="scrypt")

    # 5. Insert new user record
    insert_sql = """
        INSERT INTO users (name, email, password_hash)
        VALUES (%s, %s, %s);
    """
    try:
        with get_db_cursor(commit=True) as cursor:
            cursor.execute(insert_sql, (clean_username, clean_email, hashed_password))
            new_id = cursor.lastrowid
            logger.info("Successfully registered user id=%s (%s)", new_id, clean_username)
            return True, new_id, "Registration successful! You may now log in."
    except Exception as err:
        logger.error("Failed to insert user into database: %s", err)
        return False, None, "Registration failed due to a database error. Please try again."


def authenticate_user(
    username_or_email: str, password: str
) -> Tuple[bool, Optional[Dict[str, Any]], str]:
    """
    Authenticates a user by username or email and validates password hash.
    Returns generic failure messages to prevent user enumeration.

    Returns:
        (success: bool, user_dict: Optional[dict], message: str)
    """
    if not username_or_email or not password:
        return False, None, "Please provide both username/email and password."

    clean_identifier = username_or_email.strip()

    is_connected, _ = check_db_connection()
    if not is_connected:
        return False, None, "Database service is currently offline. Please try again later."

    lookup_sql = """
        SELECT id, name, email, password_hash
        FROM users
        WHERE name = %s OR email = %s
        LIMIT 1;
    """
    try:
        user_row = execute_query(lookup_sql, (clean_identifier, clean_identifier.lower()), fetchone=True)
        if not user_row:
            # Generic error message to prevent account enumeration
            return False, None, "Invalid username/email or password."

        stored_hash = user_row["password_hash"]
        if not check_password_hash(stored_hash, password):
            return False, None, "Invalid username/email or password."

        user_info = {
            "id": user_row["id"],
            "name": user_row["name"],
            "email": user_row["email"],
        }
        return True, user_info, "Login successful."

    except Exception as err:
        logger.error("Error during authentication: %s", err)
        return False, None, "An error occurred during authentication. Please try again."


def get_user_by_id(user_id: int) -> Optional[Dict[str, Any]]:
    """Retrieves user profile dictionary by ID, omitting sensitive password hash."""
    is_connected, _ = check_db_connection()
    if not is_connected:
        return None

    query = "SELECT id, name, email, created_at FROM users WHERE id = %s LIMIT 1;"
    try:
        row = execute_query(query, (user_id,), fetchone=True)
        return dict(row) if row else None
    except Exception as err:
        logger.error("Error retrieving user id=%s: %s", user_id, err)
        return None


def get_user_prediction_history(
    user_id: int, page: int = 1, per_page: int = 10
) -> Dict[str, Any]:
    """
    Retrieves paginated prediction records specifically belonging to the authenticated user.
    Uses parameterized SQL and sorts by created_at DESC.
    """
    is_connected, _ = check_db_connection()
    if not is_connected:
        return {
            "predictions": [],
            "total": 0,
            "page": page,
            "per_page": per_page,
            "total_pages": 0,
            "db_error": "Database is currently offline. Prediction history cannot be loaded.",
        }

    page = max(1, page)
    per_page = max(1, min(per_page, 50))
    offset = (page - 1) * per_page

    count_sql = "SELECT COUNT(*) as total FROM predictions WHERE user_id = %s;"
    records_sql = """
        SELECT id, image_path, crop_name, disease_code, disease_name,
               confidence, model_version, created_at
        FROM predictions
        WHERE user_id = %s
        ORDER BY created_at DESC
        LIMIT %s OFFSET %s;
    """

    try:
        count_row = execute_query(count_sql, (user_id,), fetchone=True)
        total_count = count_row["total"] if count_row else 0
        total_pages = math.ceil(total_count / per_page) if total_count > 0 else 1

        rows = execute_query(records_sql, (user_id, per_page, offset), fetchall=True)
        predictions = []
        if rows:
            for r in rows:
                item = dict(r)
                item["confidence"] = float(item["confidence"])
                item["confidence_percent"] = round(item["confidence"] * 100, 2)
                item["is_healthy"] = "healthy" in item.get("disease_code", "").lower() or "healthy" in item.get("disease_name", "").lower()
                predictions.append(item)

        return {
            "predictions": predictions,
            "total": total_count,
            "page": page,
            "per_page": per_page,
            "total_pages": total_pages,
            "db_error": None,
        }

    except Exception as err:
        logger.error("Error retrieving user history for user_id=%s: %s", user_id, err)
        return {
            "predictions": [],
            "total": 0,
            "page": page,
            "per_page": per_page,
            "total_pages": 0,
            "db_error": "Failed to query prediction records from database.",
        }
