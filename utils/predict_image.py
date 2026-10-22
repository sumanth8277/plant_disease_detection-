"""
Single-Image Inference Smoke Test CLI (utils/predict_image.py)
Development and debugging smoke test to verify that a trained MobileNetV2 model
and class mapping perform inference as expected on a single foliar photograph.

Note:
    This is an offline smoke test utility for Phase 3 engineering verification.
    Live web upload inference and pesticide/fertilizer advisory lookup will be
    integrated in Phase 4.

Usage:
    python utils/predict_image.py --image path/to/leaf.jpg
    python utils/predict_image.py --image path/to/leaf.jpg --top-k 3
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Configure UTF-8 on Windows consoles if supported
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import Config
from services.image_preprocessing import ImagePreprocessingError, load_and_preprocess_image

logger = logging.getLogger(__name__)


def predict_single_leaf_image(
    image_path: Path,
    model_path: Optional[Path] = None,
    class_indices_path: Optional[Path] = None,
    top_k: int = 3,
) -> Dict[str, Any]:
    """
    Runs model inference on a single foliar leaf image.

    Returns:
        Dictionary containing top class, confidence, and top-k predictions.
    """
    import tensorflow as tf

    model_path = model_path or (Config.BASE_DIR / "model" / "plant_disease_model.keras")
    class_indices_path = class_indices_path or (Config.BASE_DIR / "model" / "class_indices.json")

    # 1. Validation checks
    if not image_path.exists() or not image_path.is_file():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at '{model_path}'. "
            "Please run 'python utils/train_model.py' to produce a model artifact before running inference."
        )

    if not class_indices_path.exists():
        raise FileNotFoundError(
            f"Class mapping dictionary not found at '{class_indices_path}'. "
            "Please run 'python utils/prepare_dataset.py' to generate class mappings."
        )

    # 2. Load model and class indices
    model = tf.keras.models.load_model(str(model_path))

    with open(class_indices_path, "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)

    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(idx_to_class)

    # Validate output dimension
    expected_output_dim = model.output_shape[-1]
    if expected_output_dim != num_classes:
        raise ValueError(
            f"Model output dimension ({expected_output_dim}) does not match class mapping count ({num_classes})."
        )

    # 3. Preprocess image
    arr = load_and_preprocess_image(image_path)
    # Add batch dimension: (1, 224, 224, 3)
    batch_input = np.expand_dims(arr, axis=0)

    # 4. Forward pass
    probabilities = model.predict(batch_input, verbose=0)[0]

    # Rank predictions
    ranked_indices = np.argsort(probabilities)[::-1]
    effective_top_k = min(top_k, num_classes)

    top_class_idx = int(ranked_indices[0])
    top_class_name = idx_to_class[top_class_idx]
    top_confidence = float(probabilities[top_class_idx])

    top_predictions = [
        {
            "class_name": idx_to_class[int(idx)],
            "class_index": int(idx),
            "confidence": round(float(probabilities[idx]), 4),
            "percentage": f"{probabilities[idx] * 100:.2f}%",
        }
        for idx in ranked_indices[:effective_top_k]
    ]

    return {
        "image_path": str(image_path.resolve()),
        "top_class": top_class_name,
        "confidence": round(top_confidence, 4),
        "confidence_percentage": f"{top_confidence * 100:.2f}%",
        "top_predictions": top_predictions,
        "disclaimer": (
            "Notice: This prediction is an AI model estimation based on trained visual patterns. "
            "It does not constitute a guaranteed agronomic diagnosis."
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run development smoke test inference on a single plant leaf image."
    )
    parser.add_argument("--image", type=str, required=True, help="Path to leaf image file (.jpg, .jpeg, .png)")
    parser.add_argument("--model-path", type=str, default=str(Config.BASE_DIR / "model" / "plant_disease_model.keras"))
    parser.add_argument("--class-indices-path", type=str, default=str(Config.BASE_DIR / "model" / "class_indices.json"))
    parser.add_argument("--top-k", type=int, default=3, help="Number of top predictions to display (default: 3)")

    args = parser.parse_args()

    print("=" * 70)
    print("Plant Disease Inference Smoke Test (Phase 3 Development Utility)")
    print("=" * 70)
    print(f"Target Image:  {args.image}")
    print(f"Model Path:    {args.model_path}")
    print("-" * 70)

    try:
        result = predict_single_leaf_image(
            image_path=Path(args.image),
            model_path=Path(args.model_path),
            class_indices_path=Path(args.class_indices_path),
            top_k=args.top_k,
        )

        print(f"\n[RESULT] Top Predicted Class: {result['top_class']}")
        print(f"Confidence Score:    {result['confidence_percentage']}")
        print("\nTop Ranked Candidates:")
        for rank, item in enumerate(result["top_predictions"], 1):
            print(f"  {rank}. {item['class_name']:<35} ({item['percentage']})")

        print("\n" + "-" * 70)
        print(f"[NOTE] {result['disclaimer']}")
        print("=" * 70)

    except Exception as err:
        print(f"\n[ERROR] Inference failed: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
