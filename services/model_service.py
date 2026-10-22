"""
Model Service Module (services/model_service.py)
Manages MobileNetV2 deep learning artifact validation, lazy loading,
thread-safe caching, input preprocessing, and robust classification inference.
"""

import json
import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from config import Config
from services.image_preprocessing import (
    DEFAULT_IMAGE_SIZE,
    ImagePreprocessingError,
    load_and_preprocess_image,
)
from services.plant_identifier_service import run_hierarchical_identification
from utils.disease_mapping import normalize_disease_class

logger = logging.getLogger(__name__)


class ModelError(Exception):
    """Base exception for model service errors."""
    pass


class ModelNotFoundError(ModelError):
    """Raised when model weights or artifacts are missing."""
    pass


class ModelInferenceError(ModelError):
    """Raised when forward-pass inference fails or returns invalid probabilities."""
    pass


class ModelArtifactValidationError(ModelError):
    """Raised when model weights or class mappings are corrupted or mismatched."""
    pass


class ModelService:
    """
    Singleton-style service managing model loading, validation, and inference.
    Thread-safe and lazy-loaded to avoid blocking web server startup.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._model = None
        self._class_indices: Optional[Dict[int, str]] = None
        self._class_to_idx: Optional[Dict[str, int]] = None
        self._metadata: Dict[str, Any] = {}
        self._is_loaded = False
        self._load_error: Optional[str] = None

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def get_load_error(self) -> Optional[str]:
        return self._load_error

    def is_model_ready(self) -> Tuple[bool, str]:
        """
        Non-raising health check for model readiness.
        Returns:
            (True, "Model ready") if loaded and verified,
            (False, "<reason>") if unavailable.
        """
        if self._is_loaded and self._model is not None and self._class_indices:
            return True, "Model ready and loaded."

        # If not yet attempted, attempt safe load
        if not self._is_loaded and self._load_error is None:
            try:
                self.load_model_artifacts()
                if self._is_loaded:
                    return True, "Model ready and loaded."
            except Exception as err:
                return False, f"Model artifacts unavailable: {err}"

        return False, self._load_error or "Model artifacts have not been trained or generated yet."

    def load_model_artifacts(
        self,
        model_path: Optional[Union[str, Path]] = None,
        class_indices_path: Optional[Union[str, Path]] = None,
        metadata_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """
        Loads and validates model weights, class indices, and training metadata.
        Thread-safe; does nothing if already loaded unless called with explicit paths.
        """
        with self._lock:
            target_model = Path(model_path or Config.MODEL_PATH)
            target_indices = Path(class_indices_path or Config.CLASS_INDICES_PATH)
            target_metadata = Path(metadata_path or Config.MODEL_METADATA_PATH)

            # 1. Check artifact existence
            if not target_model.exists() or not target_model.is_file():
                self._load_error = (
                    f"Trained model artifact not found at '{target_model}'. "
                    "Please execute Phase 3 training (utils/train_model.py) or provide a valid .keras model."
                )
                self._is_loaded = False
                logger.warning(self._load_error)
                raise ModelNotFoundError(self._load_error)

            if not target_indices.exists() or not target_indices.is_file():
                self._load_error = (
                    f"Class indices mapping not found at '{target_indices}'. "
                    "Please execute dataset preparation (utils/prepare_dataset.py) to generate class mappings."
                )
                self._is_loaded = False
                logger.warning(self._load_error)
                raise ModelNotFoundError(self._load_error)

            # 2. Parse class indices
            try:
                with open(target_indices, "r", encoding="utf-8") as f:
                    raw_mapping = json.load(f)

                # Ensure mapping maps int index -> class name
                # raw_mapping can be:
                # - list: ["Apple___Apple_scab", ...] (from models/class_names.json)
                # - dict: {"Apple___Apple_scab": 0, ...} or {"0": "Apple___Apple_scab", ...}
                idx_to_class: Dict[int, str] = {}
                class_to_idx: Dict[str, int] = {}

                if isinstance(raw_mapping, list):
                    for idx, cname in enumerate(raw_mapping):
                        idx_to_class[idx] = str(cname)
                        class_to_idx[str(cname)] = idx
                elif isinstance(raw_mapping, dict):
                    for k, v in raw_mapping.items():
                        if isinstance(v, int):
                            idx_to_class[v] = str(k)
                            class_to_idx[str(k)] = v
                        elif str(k).isdigit():
                            idx_to_class[int(k)] = str(v)
                            class_to_idx[str(v)] = int(k)
                        else:
                            raise ValueError(f"Unrecognized class index entry: {k} -> {v}")
                else:
                    raise ValueError(f"Invalid class mapping data type: {type(raw_mapping)}")

                if len(idx_to_class) < 2:
                    raise ModelArtifactValidationError("Class indices file must contain at least 2 distinct classes.")

                self._class_indices = idx_to_class
                self._class_to_idx = class_to_idx

            except Exception as err:
                self._load_error = f"Failed to parse class indices from '{target_indices}': {err}"
                self._is_loaded = False
                logger.error(self._load_error)
                raise ModelArtifactValidationError(self._load_error) from err

            # 3. Load Model Metadata (optional but recommended)
            self._metadata = {
                "model_version": "MobileNetV2-v1.0",
                "input_shape": [224, 224, 3],
                "num_classes": len(self._class_indices),
            }
            if target_metadata.exists() and target_metadata.is_file():
                try:
                    with open(target_metadata, "r", encoding="utf-8") as f:
                        meta_file_data = json.load(f)
                        self._metadata.update(meta_file_data)
                except Exception as err:
                    logger.warning("Could not read model metadata file: %s", err)

            # 4. Load Keras Model
            try:
                import tensorflow as tf

                # Load with compile=False for faster inference without needing training optimizer state
                loaded_model = tf.keras.models.load_model(str(target_model), compile=False)

                # 5. Validate Output Dimensions
                expected_num_classes = len(self._class_indices)
                # Output shape can be (None, num_classes)
                output_shape = loaded_model.output_shape
                actual_num_classes = output_shape[-1] if isinstance(output_shape, tuple) else None

                if actual_num_classes != expected_num_classes:
                    raise ModelArtifactValidationError(
                        f"Model output dimension ({actual_num_classes}) does not match "
                        f"the number of mapped classes ({expected_num_classes}) in {target_indices}."
                    )

                self._model = loaded_model
                self._is_loaded = True
                self._load_error = None
                logger.info(
                    "Model successfully loaded from %s (classes=%d, input=%s)",
                    target_model,
                    actual_num_classes,
                    self._model.input_shape,
                )

            except ModelArtifactValidationError:
                raise
            except Exception as err:
                self._load_error = f"Failed to load Keras model from '{target_model}': {err}"
                self._is_loaded = False
                logger.error(self._load_error)
                raise ModelArtifactValidationError(self._load_error) from err

    def set_mock_model(self, mock_model: Any, class_indices: Dict[int, str], metadata: Optional[Dict[str, Any]] = None) -> None:
        """Helper for unit tests to inject a mock inference engine without disk artifacts."""
        with self._lock:
            self._model = mock_model
            self._class_indices = class_indices
            self._class_to_idx = {v: k for k, v in class_indices.items()}
            self._metadata = metadata or {
                "model_version": "MockModel-v1.0",
                "input_shape": [224, 224, 3],
                "num_classes": len(class_indices),
            }
            self._is_loaded = True
            self._load_error = None

    def get_model_metadata(self) -> Dict[str, Any]:
        """Returns metadata about the active model."""
        ready, msg = self.is_model_ready()
        return {
            "ready": ready,
            "status_message": msg,
            "is_loaded": self._is_loaded,
            "num_classes": len(self._class_indices) if self._class_indices else 0,
            "classes": sorted(list(self._class_indices.values())) if self._class_indices else [],
            "metadata": self._metadata,
        }

    def predict_image(
        self,
        image_source: Union[str, Path, bytes, Any],
        top_k: int = 3,
    ) -> Dict[str, Any]:
        """
        Preprocesses a leaf image, executes inference, and validates output probabilities.

        Args:
            image_source: Path, bytes, or Pillow Image object.
            top_k: Number of highest ranking predictions to include.

        Returns:
            dict containing:
              - success: True
              - raw_class: Top predicted class string
              - crop_name: Normalized crop name
              - disease_name: Normalized disease name
              - disease_code: Database query code
              - is_healthy: Boolean flag
              - confidence: Numeric probability in [0, 1]
              - confidence_percent: Confidence percentage (e.g. 96.42)
              - top_predictions: List of top classes with probabilities
              - model_version: Model version identifier

        Raises:
            ModelNotFoundError: if model is not loaded.
            ImagePreprocessingError: if image cannot be loaded or is corrupt.
            ModelInferenceError: if model inference produces NaN or invalid output.
        """
        ready, msg = self.is_model_ready()
        if not ready or self._model is None or not self._class_indices:
            raise ModelNotFoundError(
                f"Model is not ready for inference: {msg}. Please ensure Phase 3 training is complete."
            )

        # 1. Preprocess Image
        try:
            preprocessed_arr = load_and_preprocess_image(
                image_source, target_size=DEFAULT_IMAGE_SIZE
            )
        except Exception as err:
            logger.warning("Image preprocessing failed: %s", err)
            raise ImagePreprocessingError(f"Image preprocessing failed: {err}") from err

        # 2. Add batch dimension: shape (1, 224, 224, 3)
        batch = np.expand_dims(preprocessed_arr, axis=0)

        # 3. Model Forward Pass
        try:
            if hasattr(self._model, "predict"):
                raw_output = self._model.predict(batch, verbose=0)
            elif callable(self._model):
                raw_output = self._model(batch)
            else:
                raise ModelInferenceError("Loaded model object is neither callable nor has a predict method.")
        except Exception as err:
            logger.error("Model forward pass failed: %s", err)
            raise ModelInferenceError(f"Model forward pass failed: {err}") from err

        # Convert to numpy array if tensor
        if hasattr(raw_output, "numpy"):
            raw_output = raw_output.numpy()
        probs = np.asarray(raw_output, dtype=np.float32)

        # 4. Validate Probability Vector
        if probs.ndim == 2:
            probs = probs[0]
        elif probs.ndim != 1:
            raise ModelInferenceError(f"Expected 1D or 2D model output, got shape {probs.shape}")

        if len(probs) != len(self._class_indices):
            raise ModelInferenceError(
                f"Model output size ({len(probs)}) does not match class count ({len(self._class_indices)})."
            )

        if not np.all(np.isfinite(probs)):
            raise ModelInferenceError("Model returned non-finite probabilities (NaN or Inf).")

        if np.any(probs < -1e-5):
            raise ModelInferenceError("Model output contains negative probabilities.")

        # Ensure probabilities sum to 1.0 (softmax normalization if needed)
        prob_sum = float(np.sum(probs))
        if prob_sum <= 0.0:
            raise ModelInferenceError("Sum of model probabilities is zero.")
        if abs(prob_sum - 1.0) > 0.05:
            # Softmax was not applied in head or slightly drifting, normalize safely
            probs = probs / prob_sum

        # Clip values to [0.0, 1.0]
        probs = np.clip(probs, 0.0, 1.0)

        # 5. Top Predictions and Out-Of-Distribution (OOD) / Unsupported Plant Detection
        k = min(max(1, top_k), len(probs))
        top_indices = np.argsort(probs)[::-1][:k]

        primary_idx = int(top_indices[0])
        primary_conf = float(probs[primary_idx])
        second_conf = float(probs[top_indices[1]]) if len(top_indices) > 1 else 0.0
        margin = float(primary_conf - second_conf)

        # Multi-metric OOD & Uncertainty Analysis
        # Closed-set softmax models force probabilities to sum to 1.0, which can assign arbitrary classes
        # to out-of-distribution inputs like Neem leaves. We compute entropy and margin to detect diffuse distributions.
        entropy = -float(np.sum(probs * np.log(probs + 1e-12)))
        max_possible_entropy = float(np.log(len(probs)))
        normalized_entropy = float(entropy / max_possible_entropy) if max_possible_entropy > 0 else 0.0

        # Heuristic thresholds for supported crop leaf identification
        # Note: A simple confidence threshold alone cannot reliably catch all OOD samples (overconfidence is a known
        # limitation of deep neural networks on anomalous inputs). We combine confidence, entropy, and margin.
        MIN_CONFIDENCE_THRESHOLD = 0.40
        MAX_NORMALIZED_ENTROPY = 0.65
        MIN_PREDICTION_MARGIN = 0.12

        is_supported = True
        ood_reason = None

        if primary_conf < MIN_CONFIDENCE_THRESHOLD:
            is_supported = False
            ood_reason = (
                f"Low model confidence ({primary_conf * 100:.1f}% < {MIN_CONFIDENCE_THRESHOLD * 100:.0f}%). "
                "The specimen cannot be reliably matched to any supported crop category."
            )
        elif normalized_entropy > MAX_NORMALIZED_ENTROPY:
            is_supported = False
            ood_reason = (
                f"High prediction entropy ({normalized_entropy:.2f} > {MAX_NORMALIZED_ENTROPY:.2f}). "
                "Model activations are diffuse across unrelated plant families, typical of an unsupported plant species."
            )
        elif margin < MIN_PREDICTION_MARGIN:
            is_supported = False
            ood_reason = (
                f"Narrow class margin ({margin * 100:.1f}% < {MIN_PREDICTION_MARGIN * 100:.0f}%). "
                "The model cannot distinguish between top candidate classes, indicating an out-of-distribution specimen."
            )

        primary_class = self._class_indices.get(primary_idx, f"Class_{primary_idx}")
        primary_norm = normalize_disease_class(primary_class)

        # 6. Hierarchical Plant Species Identification & Diagnosis Gate
        hier = run_hierarchical_identification(
            image_source=image_source,
            probs=probs,
            class_indices=self._class_indices,
            top_indices=top_indices,
            margin=margin,
            normalized_entropy=normalized_entropy,
        )

        is_neem = hier.get("is_neem", False)
        is_plant_supported = hier.get("is_plant_supported", is_supported)
        is_disease_supported = hier.get("is_disease_supported", is_supported)

        display_conf = primary_conf
        if is_neem:
            crop_name = "Neem (Azadirachta indica)"
            disease_name = "Neem disease diagnosis is not yet supported"
            disease_code = "NEEM_DISEASE_UNSUPPORTED"
            is_healthy = False
            is_supported = False  # strictly withhold crop-specific chemical recommendations
            is_inconclusive = True
            ood_reason = "Neem identified, but neem disease detection is not yet supported"
            display_conf = min(float(primary_conf), 0.25)
        elif not is_supported or not is_plant_supported:
            crop_name = "Unsupported / Unknown Plant"
            disease_name = "Unsupported or Inconclusive Specimen"
            disease_code = "UNSUPPORTED_SPECIMEN"
            is_healthy = False
            is_supported = False
            is_inconclusive = True
            display_conf = min(float(primary_conf), 0.25)
        else:
            crop_name = primary_norm["crop_name"]
            disease_name = primary_norm["disease_name"]
            disease_code = primary_norm["disease_code"]
            is_healthy = primary_norm["is_healthy"]
            is_inconclusive = False

        top_list: List[Dict[str, Any]] = []
        for idx in top_indices:
            cls_name = self._class_indices.get(int(idx), f"Class_{idx}")
            cls_prob = float(probs[idx])
            cls_norm = normalize_disease_class(cls_name)
            top_list.append(
                {
                    "rank": len(top_list) + 1,
                    "raw_class": cls_name,
                    "crop_name": cls_norm["crop_name"],
                    "disease_name": cls_norm["disease_name"],
                    "disease_code": cls_norm["disease_code"],
                    "is_healthy": cls_norm["is_healthy"],
                    "confidence": round(cls_prob, 4),
                    "confidence_percent": round(cls_prob * 100, 2),
                    "status": "candidate" if not is_supported else "verified_crop",
                }
            )

        return {
            "success": True,
            "identified_plant": hier.get("identified_plant", crop_name if is_supported else "Unsupported / Unknown Plant"),
            "is_plant_supported": is_plant_supported,
            "plant_confidence": hier.get("plant_confidence", round(primary_conf, 4)),
            "plant_confidence_percent": hier.get("plant_confidence_percent", round(primary_conf * 100, 2)),
            "is_disease_supported": is_disease_supported,
            "predicted_disease": hier.get("predicted_disease", disease_name),
            "disease_status_message": hier.get("disease_status_message", "Plant species and disease condition verified." if is_supported else ("Neem identified, but neem disease detection is not yet supported" if is_neem else "Unsupported or Inconclusive Specimen")),
            "confidence_meaning": hier.get("confidence_meaning", "High diagnostic confidence." if primary_conf >= 0.70 else "Moderate diagnostic confidence."),
            "is_neem": is_neem,
            "crop_probabilities": hier.get("crop_probabilities", {}),
            "is_supported": is_supported,
            "is_inconclusive": is_inconclusive,
            "ood_reason": ood_reason,
            "raw_class": primary_class if (is_supported and not is_neem) else ("Neem_Specimen" if is_neem else "Unsupported_Plant"),
            "candidate_raw_class": primary_class,
            "crop_name": crop_name,
            "disease_name": disease_name,
            "disease_code": disease_code,
            "is_healthy": is_healthy,
            "confidence": round(display_conf if (is_neem or not is_supported or not is_plant_supported) else primary_conf, 4),
            "confidence_percent": round((display_conf if (is_neem or not is_supported or not is_plant_supported) else primary_conf) * 100, 2),
            "entropy": round(entropy, 4),
            "normalized_entropy": round(normalized_entropy, 4),
            "prediction_margin": round(margin, 4),
            "top_predictions": top_list,
            "model_version": self._metadata.get("model_version", "MobileNetV2-v1.0"),
            "supported_crops": [
                "Apple", "Blueberry", "Cherry", "Corn (Maize)", "Grape", "Orange",
                "Peach", "Bell Pepper", "Potato", "Raspberry", "Soybean", "Squash",
                "Strawberry", "Tomato",
            ],
        }


# Global singleton instance for application use
_model_service_instance: Optional[ModelService] = None


def get_model_service() -> ModelService:
    """Returns the shared ModelService singleton instance."""
    global _model_service_instance
    if _model_service_instance is None:
        _model_service_instance = ModelService()
    return _model_service_instance
