"""
Image Preprocessing Service (services/image_preprocessing.py)
Provides consistent, production-grade image preparation for MobileNetV2:
- RGB conversion (handling RGBA, grayscale, CMYK, Palette)
- 224x224 bilinear resizing
- MobileNetV2-specific normalization ([-1, 1] scaling via tf.keras.applications.mobilenet_v2.preprocess_input)
- Configurable training data augmentation layers
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)

# Standard input dimensions for MobileNetV2
DEFAULT_IMAGE_SIZE = (224, 224)


class ImagePreprocessingError(Exception):
    """Raised when image preprocessing fails or input is invalid."""
    pass


def convert_image_to_rgb(img: Image.Image) -> Image.Image:
    """
    Converts any Pillow image mode (RGBA, L, CMYK, P) to clean 3-channel RGB.
    Transparencies in RGBA/LA are composited onto a solid white background.
    """
    if img.mode == "RGB":
        return img

    if img.mode in ("RGBA", "LA") or ("transparency" in img.info):
        alpha_img = img.convert("RGBA")
        background = Image.new("RGBA", alpha_img.size, (255, 255, 255, 255))
        composed = Image.alpha_composite(background, alpha_img)
        return composed.convert("RGB")

    if img.mode in ("L", "P", "CMYK", "1"):
        return img.convert("RGB")

    return img.convert("RGB")


def load_and_preprocess_image(
    image_source: Union[str, Path, Image.Image, bytes],
    target_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> np.ndarray:
    """
    Loads, validates, converts to RGB, resizes, and scales an image for MobileNetV2.

    Returns:
        np.ndarray of shape (224, 224, 3), float32, scaled to [-1, 1].

    Raises:
        ImagePreprocessingError: if image cannot be read, is corrupt, or is unsupported.
    """
    img = None
    try:
        if isinstance(image_source, (str, Path)):
            fpath = Path(image_source)
            if not fpath.exists() or not fpath.is_file():
                raise ImagePreprocessingError(f"Image file does not exist: {fpath}")
            img = Image.open(fpath)
        elif isinstance(image_source, bytes):
            import io
            img = Image.open(io.BytesIO(image_source))
        elif isinstance(image_source, Image.Image):
            img = image_source
        else:
            raise ImagePreprocessingError(f"Unsupported image source type: {type(image_source)}")

        # Verify integrity
        img.load()
        # Normalize to RGB
        rgb_img = convert_image_to_rgb(img)

        # Bilinear resize to target dimension (224, 224)
        if rgb_img.size != target_size:
            resampling = getattr(Image, "Resampling", Image).BILINEAR
            rgb_img = rgb_img.resize(target_size, resample=resampling)

        # Convert to numpy float32 array
        arr = np.asarray(rgb_img, dtype=np.float32)

        # Apply MobileNetV2 preprocessing: (x / 127.5) - 1.0 (scales [0, 255] to [-1, 1])
        # We can implement directly or use tf.keras.applications.mobilenet_v2.preprocess_input
        try:
            from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
            processed = preprocess_input(arr)
        except ImportError:
            # Fallback pure-numpy implementation identical to tf.keras.applications.mobilenet_v2.preprocess_input
            processed = (arr / 127.5) - 1.0

        if processed.shape != (target_size[0], target_size[1], 3):
            raise ImagePreprocessingError(f"Unexpected preprocessed shape: {processed.shape}")

        return processed

    except (UnidentifiedImageError, OSError, SyntaxError) as err:
        logger.warning("Corrupt or invalid image encountered: %s", err)
        raise ImagePreprocessingError(f"Corrupted or invalid image: {err}") from err
    except ImagePreprocessingError:
        raise
    except Exception as err:
        logger.error("Unexpected error during image preprocessing: %s", err)
        raise ImagePreprocessingError(f"Preprocessing failed: {err}") from err
    finally:
        if img is not None and isinstance(image_source, (str, Path)):
            try:
                img.close()
            except Exception:
                pass


def get_augmentation_pipeline(
    random_flip: str = "horizontal",
    rotation_factor: float = 0.08,
    zoom_factor: float = 0.08,
    seed: Optional[int] = 42,
    **kwargs,
):
    """
    Creates a Keras Sequential layer for data augmentation during training.
    Does NOT augment validation or test sets. Modest transformations preserve
    foliar disease lesions and visual diagnostic characteristics.
    """
    import tensorflow as tf

    layers = []
    if random_flip in ("horizontal", "horizontal_and_vertical"):
        layers.append(tf.keras.layers.RandomFlip(mode=random_flip, seed=seed))
    if rotation_factor > 0:
        layers.append(tf.keras.layers.RandomRotation(factor=rotation_factor, fill_mode="reflect", seed=seed))
    if zoom_factor > 0:
        layers.append(tf.keras.layers.RandomZoom(height_factor=(-zoom_factor, zoom_factor), seed=seed))

    return tf.keras.Sequential(layers, name="foliar_augmentation_pipeline")
