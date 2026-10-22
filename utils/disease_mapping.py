"""
Disease Label Normalization and Mapping Utility (utils/disease_mapping.py)
Translates raw model prediction class names (e.g., 'Tomato___Early_blight')
into clean display names, standardized disease codes, and healthy vs. diseased flags.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional

from config import Config

logger = logging.getLogger(__name__)

# Cached mapping dictionary loaded from JSON
_MAPPING_CACHE: Optional[Dict[str, Dict[str, Any]]] = None


def load_disease_mappings(mapping_file: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    """
    Loads explicit class-name-to-disease mappings from model/disease_class_mapping.json.
    Caches the parsed mapping in memory.
    """
    global _MAPPING_CACHE
    if _MAPPING_CACHE is not None:
        return _MAPPING_CACHE

    target_path = Path(mapping_file or Config.DISEASE_MAPPING_PATH)
    if target_path.exists() and target_path.is_file():
        try:
            with open(target_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                _MAPPING_CACHE = data.get("mappings", {})
                logger.info("Loaded %d explicit class mappings from %s", len(_MAPPING_CACHE), target_path)
                return _MAPPING_CACHE
        except Exception as err:
            logger.warning("Could not read disease mapping file %s: %s", target_path, err)

    _MAPPING_CACHE = {}
    return _MAPPING_CACHE


def clean_crop_name(raw_crop: str) -> str:
    """Standardizes crop names from dataset folder conventions into clean presentation text."""
    crop_clean = raw_crop.strip().replace("_", " ")
    # Handle known parenthetical names
    if "corn" in crop_clean.lower() and "maize" in crop_clean.lower():
        return "Corn (Maize)"
    if "cherry" in crop_clean.lower():
        return "Cherry"
    if "pepper" in crop_clean.lower() and "bell" in crop_clean.lower():
        return "Bell Pepper"
    return crop_clean.title()


def clean_disease_name(raw_disease: str) -> str:
    """Standardizes condition/disease names from folder conventions into clean presentation text."""
    disease_clean = raw_disease.strip().replace("_", " ")
    # Remove trailing underscores or artifact characters
    disease_clean = re.sub(r"\s+", " ", disease_clean).strip()
    if disease_clean.lower() == "healthy":
        return "Healthy Foliage"
    return disease_clean.title()


def generate_fallback_disease_code(crop_name: str, disease_name: str, is_healthy: bool) -> str:
    """
    Generates a deterministic uppercase disease code when no explicit mapping exists.
    e.g., 'Tomato', 'Early Blight' -> 'TOMATO_EARLY_BLIGHT'.
    """
    if is_healthy:
        clean_crop = re.sub(r"[^A-Za-z0-9]+", "_", crop_name.upper()).strip("_")
        return f"{clean_crop}_HEALTHY"

    clean_crop = re.sub(r"[^A-Za-z0-9]+", "_", crop_name.upper()).strip("_")
    # Take the primary condition words, removing parenthesized latin names
    base_disease = re.sub(r"\(.*?\)", "", disease_name)
    clean_disease = re.sub(r"[^A-Za-z0-9]+", "_", base_disease.upper()).strip("_")
    return f"{clean_crop}_{clean_disease}"


def normalize_disease_class(raw_class_name: str) -> Dict[str, Any]:
    """
    Normalizes a model output class label into a structured, validated metadata dictionary.

    Args:
        raw_class_name: e.g. 'Tomato___Early_blight' or 'Corn_(maize)___Common_rust_'

    Returns:
        dict:
          - raw_class: Original unedited class label
          - crop_name: Human-friendly display crop name
          - disease_name: Human-friendly display condition name
          - disease_code: Stable uppercase code for MySQL recommendation joins
          - is_healthy: Boolean indicating if foliage is healthy
    """
    if not raw_class_name or not isinstance(raw_class_name, str):
        return {
            "raw_class": str(raw_class_name),
            "crop_name": "Unknown Crop",
            "disease_name": "Unclassified Condition",
            "disease_code": "UNKNOWN",
            "is_healthy": False,
        }

    raw_clean = raw_class_name.strip()

    # 1. Check explicit JSON mapping first
    mappings = load_disease_mappings()
    if raw_clean in mappings:
        item = mappings[raw_clean]
        return {
            "raw_class": raw_clean,
            "crop_name": item.get("crop_name", "Unknown Crop"),
            "disease_name": item.get("disease_name", "Unknown Condition"),
            "disease_code": item.get("disease_code", "UNKNOWN"),
            "is_healthy": bool(item.get("is_healthy", False)),
        }

    # 2. Algorithmic fallback using PlantVillage 'Crop___Condition' delimiter
    if "___" in raw_clean:
        parts = raw_clean.split("___", 1)
        raw_crop, raw_condition = parts[0], parts[1]
    else:
        # Fallback single underscore or space separation
        parts = re.split(r"[-_ ]{2,}", raw_clean, maxsplit=1)
        raw_crop = parts[0] if parts else "Unknown Crop"
        raw_condition = parts[1] if len(parts) > 1 else "Unknown Condition"

    is_healthy = "healthy" in raw_condition.lower()
    crop_display = clean_crop_name(raw_crop)
    disease_display = clean_disease_name(raw_condition)
    disease_code = generate_fallback_disease_code(crop_display, disease_display, is_healthy)

    return {
        "raw_class": raw_clean,
        "crop_name": crop_display,
        "disease_name": disease_display,
        "disease_code": disease_code,
        "is_healthy": is_healthy,
    }
