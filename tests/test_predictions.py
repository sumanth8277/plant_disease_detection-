"""
Unit tests for PredictionService (tests/test_predictions.py)
Tests prediction record persistence, confidence validation, database transaction rollback,
and history retrieval.
"""

from unittest.mock import MagicMock, patch
import pytest

from services.prediction_service import (
    get_prediction_record,
    save_prediction_record,
)


def test_save_prediction_record_valid():
    """Verify that a valid prediction record is inserted with parameterized SQL."""
    mock_cursor = MagicMock()
    mock_cursor.lastrowid = 42

    with patch("services.prediction_service.check_db_connection", return_value=(True, "OK")), \
         patch("services.prediction_service.get_db_cursor") as mock_ctx:
        mock_ctx.return_value.__enter__.return_value = mock_cursor

        success, pred_id, msg = save_prediction_record(
            image_path="uploads/test_leaf.jpg",
            crop_name="Tomato",
            disease_code="TOMATO_EARLY_BLIGHT",
            disease_name="Early Blight",
            confidence=0.9642,
            model_version="v1.0",
            user_id=None,
            upload_id="abc12345",
        )

        assert success is True
        assert pred_id == 42
        assert mock_cursor.execute.called
        # Verify parameterized query was executed
        args, _ = mock_cursor.execute.call_args
        assert "%s" in args[0]


def test_save_prediction_record_invalid_confidence_rejected():
    """Verify that confidence out of [0, 1] range is rejected before database call."""
    with patch("services.prediction_service.check_db_connection") as mock_conn:
        success, pred_id, msg = save_prediction_record(
            image_path="uploads/test.jpg",
            crop_name="Tomato",
            disease_code="TOMATO_EARLY_BLIGHT",
            disease_name="Early Blight",
            confidence=1.5,  # Invalid confidence > 1.0
        )
        assert success is False
        assert pred_id is None
        assert "Invalid confidence" in msg
        assert not mock_conn.called


def test_save_prediction_record_database_offline():
    """Verify safe handling when MySQL server is offline."""
    with patch("services.prediction_service.check_db_connection", return_value=(False, "Connection refused")):
        success, pred_id, msg = save_prediction_record(
            image_path="uploads/test.jpg",
            crop_name="Tomato",
            disease_code="TOMATO_EARLY_BLIGHT",
            disease_name="Early Blight",
            confidence=0.85,
        )
        assert success is False
        assert pred_id is None
        assert "Database offline" in msg


def test_get_prediction_record_fallback_cache():
    """Verify that in-memory cache returns recent predictions if DB is offline."""
    from services.prediction_service import _RECENT_PREDICTIONS_CACHE
    _RECENT_PREDICTIONS_CACHE["cache_test_key"] = {
        "crop_name": "Tomato",
        "disease_name": "Late Blight",
        "confidence": 0.95,
        "confidence_percent": 95.0,
    }

    with patch("services.prediction_service.check_db_connection", return_value=(False, "Offline")):
        record, is_forbidden = get_prediction_record("cache_test_key")
        assert is_forbidden is False
        assert record is not None
        assert record["crop_name"] == "Tomato"
        assert record["confidence"] == 0.95
