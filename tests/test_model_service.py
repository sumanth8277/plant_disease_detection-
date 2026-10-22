"""
Unit tests for ModelService (tests/test_model_service.py)
Tests model artifact validation, mock inference, error handling,
and probability checks without requiring GPU or downloading real weights.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from PIL import Image

from services.model_service import (
    ModelArtifactValidationError,
    ModelInferenceError,
    ModelNotFoundError,
    ModelService,
)


@pytest.fixture
def mock_model_service():
    """Provides an isolated ModelService instance for unit testing."""
    return ModelService()


def test_model_service_initial_state_not_ready(mock_model_service, tmp_path):
    """Verify that a freshly instantiated ModelService starts unready when artifacts are missing."""
    with patch("services.model_service.Config.MODEL_PATH", tmp_path / "non_existent.keras"):
        ready, msg = mock_model_service.is_model_ready()
        assert ready is False
        assert "unavailable" in msg.lower() or "not found" in msg.lower() or "trained" in msg.lower()


def test_load_model_artifacts_missing_file_raises_error(mock_model_service, tmp_path):
    """Verify that attempting to load a non-existent model file raises ModelNotFoundError."""
    fake_model = tmp_path / "non_existent.keras"
    fake_indices = tmp_path / "indices.json"
    with pytest.raises(ModelNotFoundError):
        mock_model_service.load_model_artifacts(
            model_path=fake_model, class_indices_path=fake_indices
        )


def test_load_model_artifacts_invalid_class_indices(mock_model_service, tmp_path):
    """Verify that invalid or corrupt class indices files are rejected."""
    model_file = tmp_path / "dummy.keras"
    model_file.write_text("dummy binary content")
    bad_indices_file = tmp_path / "bad_indices.json"
    bad_indices_file.write_text('{"SingleClass": 0}')  # Less than 2 classes

    with pytest.raises(ModelArtifactValidationError):
        mock_model_service.load_model_artifacts(
            model_path=model_file, class_indices_path=bad_indices_file
        )


def test_mock_model_inference_valid(mock_model_service, tmp_path):
    """Verify forward-pass prediction, top-k ranking, and probability checks with a mock model."""
    class_indices = {
        0: "Apple___Apple_scab",
        1: "Tomato___Early_blight",
        2: "Tomato___healthy",
    }

    # Mock Keras model returning softmax probabilities: [0.05, 0.90, 0.05]
    mock_engine = MagicMock()
    mock_engine.predict.return_value = np.array([[0.05, 0.90, 0.05]], dtype=np.float32)

    mock_model_service.set_mock_model(mock_engine, class_indices)
    assert mock_model_service.is_model_ready()[0] is True

    # Create dummy 224x224 leaf image
    dummy_img = Image.new("RGB", (224, 224), color=(34, 139, 34))
    img_path = tmp_path / "test_leaf.jpg"
    dummy_img.save(img_path)

    result = mock_model_service.predict_image(img_path, top_k=3)

    assert result["success"] is True
    assert result["crop_name"] == "Tomato"
    assert result["disease_code"] == "TOMATO_EARLY_BLIGHT"
    assert result["is_healthy"] is False
    assert result["confidence"] == 0.90
    assert result["confidence_percent"] == 90.0
    assert len(result["top_predictions"]) == 3
    assert result["top_predictions"][0]["raw_class"] == "Tomato___Early_blight"
    assert result["top_predictions"][0]["confidence"] == 0.90


def test_mock_model_inference_nan_rejected(mock_model_service, tmp_path):
    """Verify that model outputs containing NaN are rejected with ModelInferenceError."""
    class_indices = {0: "Tomato___Early_blight", 1: "Tomato___healthy"}
    mock_engine = MagicMock()
    mock_engine.predict.return_value = np.array([[np.nan, 0.5]], dtype=np.float32)

    mock_model_service.set_mock_model(mock_engine, class_indices)

    dummy_img = Image.new("RGB", (224, 224), color=(50, 50, 50))
    with pytest.raises(ModelInferenceError):
        mock_model_service.predict_image(dummy_img)


def test_mock_model_inference_shape_mismatch(mock_model_service, tmp_path):
    """Verify that model output dimension mismatch with class count raises ModelInferenceError."""
    class_indices = {0: "Class_A", 1: "Class_B"}
    mock_engine = MagicMock()
    # Output length 3 does not match 2 classes
    mock_engine.predict.return_value = np.array([[0.2, 0.5, 0.3]], dtype=np.float32)

    mock_model_service.set_mock_model(mock_engine, class_indices)

    dummy_img = Image.new("RGB", (224, 224), color=(100, 100, 100))
    with pytest.raises(ModelInferenceError):
        mock_model_service.predict_image(dummy_img)
