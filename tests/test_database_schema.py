"""
Unit and integration tests for MySQL schema definitions (tests/test_database_schema.py).
"""

from pathlib import Path
import pytest
from database.db import check_db_connection


def test_schema_file_exists():
    """Verify database/schema.sql exists in project."""
    schema_path = Path(__file__).resolve().parent.parent / "database" / "schema.sql"
    assert schema_path.exists(), "database/schema.sql file is missing"
    assert schema_path.stat().st_size > 0, "database/schema.sql is empty"


def test_schema_sql_definitions():
    """
    Parses database/schema.sql to verify all required tables, keys,
    and constraints are properly defined.
    """
    schema_path = Path(__file__).resolve().parent.parent / "database" / "schema.sql"
    content = schema_path.read_text(encoding="utf-8")

    # Verify Database definition
    assert "CREATE DATABASE IF NOT EXISTS plant_disease_db" in content
    assert "utf8mb4" in content
    assert "InnoDB" in content

    # Verify Table A: users
    assert "CREATE TABLE IF NOT EXISTS users" in content
    assert "id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY" in content or "id BIGINT" in content
    assert "email VARCHAR(191) NOT NULL UNIQUE" in content or ("email" in content and "UNIQUE" in content)
    assert "password_hash" in content
    assert "created_at" in content

    # Verify Table B: predictions
    assert "CREATE TABLE IF NOT EXISTS predictions" in content
    assert "user_id" in content
    assert "image_path" in content
    assert "crop_name" in content
    assert "disease_code" in content
    assert "disease_name" in content
    assert "confidence" in content
    assert "model_version" in content
    assert "FOREIGN KEY (user_id) REFERENCES users(id)" in content
    assert "ON DELETE SET NULL" in content
    assert "chk_predictions_confidence" in content

    # Verify Table C: disease_recommendations
    assert "CREATE TABLE IF NOT EXISTS disease_recommendations" in content
    assert "disease_code VARCHAR(100) NOT NULL UNIQUE" in content
    assert "symptoms" in content
    assert "treatment_guidance" in content
    assert "prevention_guidance" in content
    assert "review_status" in content

    # Verify Table D: fertilizer_recommendations
    assert "CREATE TABLE IF NOT EXISTS fertilizer_recommendations" in content
    assert "growth_stage" in content
    assert "soil_condition" in content
    assert "recommendation_text" in content

    # Verify Table E: uploads (Phase 2)
    assert "CREATE TABLE IF NOT EXISTS uploads" in content
    assert "upload_id" in content
    assert "stored_filename" in content
    assert "relative_path" in content
    assert "mime_type" in content
    assert "file_size" in content
    assert "image_width" in content
    assert "image_height" in content
    assert "status" in content


def test_live_mysql_connectivity():
    """
    Checks if a live MySQL server is reachable.
    If MySQL server is not running or unconfigured, skips test with explicit explanation.
    """
    is_connected, msg = check_db_connection()
    if not is_connected:
        pytest.skip(
            f"Live MySQL Server is not currently reachable ({msg}). "
            "Skipping live database query test. Schema SQL definitions were verified offline."
        )
    assert is_connected is True
