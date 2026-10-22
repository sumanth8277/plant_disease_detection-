"""
Unit tests for MobileNetV2 disease classifier model construction and artifact management (tests/test_model_artifacts.py).
Tests dynamic class output dimensions, layer freeze/unfreeze state, and missing artifact error handling.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from models.disease_classifier import build_disease_classifier, unfreeze_for_fine_tuning
from utils.predict_image import predict_single_leaf_image
from utils.train_model import compute_file_sha256


def test_build_disease_classifier_output_dimensions():
    """Verify that build_disease_classifier creates a model whose output dimension matches num_classes."""
    # Test with 5 classes (offline random weights to avoid downloading ImageNet weights during test)
    model_5 = build_disease_classifier(num_classes=5, weights=None)
    assert model_5.output_shape == (None, 5)

    # Test with 38 classes (PlantVillage standard)
    model_38 = build_disease_classifier(num_classes=38, weights=None)
    assert model_38.output_shape == (None, 38)


def test_build_disease_classifier_frozen_backbone():
    """Verify that the MobileNetV2 backbone is frozen for initial stage-1 head training."""
    model = build_disease_classifier(num_classes=3, weights=None)

    # Locate base model
    import tensorflow as tf
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) or "mobilenetv2" in layer.name.lower():
            base_model = layer
            break

    assert base_model is not None
    assert base_model.trainable is False


def test_build_disease_classifier_invalid_class_count():
    """Verify ValueError is raised if num_classes is less than 2."""
    with pytest.raises(ValueError, match="num_classes must be at least 2"):
        build_disease_classifier(num_classes=1, weights=None)


def test_unfreeze_for_fine_tuning_keeps_batchnorm_frozen():
    """Verify that unfreezing for fine-tuning keeps all BatchNormalization layers frozen."""
    import tensorflow as tf
    model = build_disease_classifier(num_classes=3, weights=None)
    fine_tuned = unfreeze_for_fine_tuning(model, num_layers_to_unfreeze=20, fine_tune_learning_rate=1e-5)

    # Base model should now be trainable
    base_model = None
    for layer in fine_tuned.layers:
        if isinstance(layer, tf.keras.Model) or "mobilenetv2" in layer.name.lower():
            base_model = layer
            break

    assert base_model is not None
    assert base_model.trainable is True

    # Verify that all BatchNormalization layers remain frozen
    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            assert layer.trainable is False


def test_predict_single_image_missing_model_raises_error(tmp_path):
    """Verify that predict_single_leaf_image raises FileNotFoundError when model is missing."""
    dummy_img = tmp_path / "leaf.jpg"
    Image.new("RGB", (100, 100), color=(0, 200, 0)).save(dummy_img)

    with pytest.raises(FileNotFoundError, match="Trained model not found"):
        predict_single_leaf_image(
            image_path=dummy_img,
            model_path=tmp_path / "non_existent_model.keras",
            class_indices_path=tmp_path / "class_indices.json",
        )


def test_predict_single_image_missing_image_raises_error(tmp_path):
    """Verify that predict_single_leaf_image raises FileNotFoundError when input image is missing."""
    missing_img = tmp_path / "does_not_exist.jpg"
    with pytest.raises(FileNotFoundError, match="Input image not found"):
        predict_single_leaf_image(image_path=missing_img)


def test_mock_model_inference_smoke_test(tmp_path):
    """Verify end-to-end single-image inference against a mock saved model artifact."""
    import tensorflow as tf

    # 1. Create a tiny dummy model with (224, 224, 3) input and 3 classes output
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    outputs = tf.keras.layers.Dense(3, activation="softmax")(x)
    dummy_model = tf.keras.Model(inputs=inputs, outputs=outputs)

    mock_model_path = tmp_path / "mock_model.keras"
    dummy_model.save(str(mock_model_path))

    # Verify SHA-256 calculation
    checksum = compute_file_sha256(mock_model_path)
    assert len(checksum) == 64

    # 2. Create mock class indices
    class_indices = {
        "Tomato___Early_blight": 0,
        "Tomato___Late_blight": 1,
        "Tomato___healthy": 2,
    }
    mock_indices_path = tmp_path / "class_indices.json"
    with open(mock_indices_path, "w") as f:
        json.dump(class_indices, f)

    # 3. Create test image
    test_leaf = tmp_path / "test_leaf.jpg"
    Image.new("RGB", (224, 224), color=(34, 139, 34)).save(test_leaf)

    # 4. Run predict_single_leaf_image
    result = predict_single_leaf_image(
        image_path=test_leaf,
        model_path=mock_model_path,
        class_indices_path=mock_indices_path,
        top_k=3,
    )

    assert result["top_class"] in class_indices
    assert 0.0 <= result["confidence"] <= 1.0
    assert len(result["top_predictions"]) == 3
    assert "disclaimer" in result
