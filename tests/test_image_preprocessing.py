"""
Unit tests for image preprocessing service (tests/test_image_preprocessing.py).
Tests RGB conversion, 224x224 resizing, MobileNetV2 scaling, corrupt file handling, and augmentation layers.
"""

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from services.image_preprocessing import (
    ImagePreprocessingError,
    convert_image_to_rgb,
    get_augmentation_pipeline,
    load_and_preprocess_image,
)


def test_convert_rgb_already_rgb():
    """Verify that an RGB image passes through convert_image_to_rgb without modification."""
    img = Image.new("RGB", (100, 100), color=(50, 100, 150))
    converted = convert_image_to_rgb(img)
    assert converted.mode == "RGB"
    assert converted.size == (100, 100)


def test_convert_rgb_from_rgba():
    """Verify that RGBA image with transparency is composited onto white and converted to RGB."""
    # RGBA image with 50% opacity
    img = Image.new("RGBA", (80, 80), color=(255, 0, 0, 128))
    converted = convert_image_to_rgb(img)
    assert converted.mode == "RGB"
    # Red blended with white background should result in pinkish RGB values
    pixel = converted.getpixel((10, 10))
    assert len(pixel) == 3
    assert pixel[0] > 200  # High red channel


def test_convert_rgb_from_grayscale():
    """Verify that a single-channel grayscale ('L') image is expanded to 3-channel RGB."""
    img = Image.new("L", (50, 50), color=120)
    converted = convert_image_to_rgb(img)
    assert converted.mode == "RGB"
    pixel = converted.getpixel((0, 0))
    assert pixel == (120, 120, 120)


def test_load_and_preprocess_shape_and_scaling(tmp_path):
    """Verify that image is correctly resized to (224, 224, 3) and scaled to [-1, 1]."""
    test_file = tmp_path / "sample_leaf.jpg"
    img = Image.new("RGB", (400, 300), color=(100, 150, 200))
    img.save(test_file, format="JPEG")

    arr = load_and_preprocess_image(test_file)

    assert isinstance(arr, np.ndarray)
    assert arr.shape == (224, 224, 3)
    assert arr.dtype == np.float32
    # MobileNetV2 preprocessing maps [0, 255] to [-1.0, 1.0]
    assert arr.min() >= -1.01
    assert arr.max() <= 1.01


def test_load_and_preprocess_from_bytes():
    """Verify load_and_preprocess_image accepts raw image bytes directly."""
    buf = io.BytesIO()
    img = Image.new("RGB", (64, 64), color=(34, 139, 34))
    img.save(buf, format="PNG")

    arr = load_and_preprocess_image(buf.getvalue())
    assert arr.shape == (224, 224, 3)


def test_load_and_preprocess_corrupted_file(tmp_path):
    """Verify that a damaged/corrupt image raises ImagePreprocessingError."""
    corrupt_file = tmp_path / "damaged.jpg"
    corrupt_file.write_bytes(b"\xff\xd8\xff\xe0" + b"CorruptGarbageData" * 10)

    with pytest.raises(ImagePreprocessingError, match="Corrupted or invalid image"):
        load_and_preprocess_image(corrupt_file)


def test_load_and_preprocess_missing_file():
    """Verify that a non-existent path raises ImagePreprocessingError."""
    missing_path = Path("dataset/raw/non_existent_folder/missing.jpg")
    with pytest.raises(ImagePreprocessingError, match="does not exist"):
        load_and_preprocess_image(missing_path)


def test_augmentation_pipeline():
    """Verify that get_augmentation_pipeline creates a callable Keras layer."""
    aug_layer = get_augmentation_pipeline(random_flip="horizontal", rotation_factor=0.08, zoom_factor=0.08, seed=42)
    assert aug_layer is not None

    # Test layer on synthetic batch of shape (1, 224, 224, 3)
    synthetic_batch = np.zeros((1, 224, 224, 3), dtype=np.float32)
    augmented = aug_layer(synthetic_batch, training=True)
    assert augmented.shape == (1, 224, 224, 3)
