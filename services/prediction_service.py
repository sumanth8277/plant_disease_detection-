"""
Prediction Service Module (services/prediction_service.py)
Coordinates model inference, disease label normalization, agronomic recommendation lookup,
and transactional prediction persistence to the MySQL 'predictions' table.
"""

import logging
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from config import Config
from database.db import check_db_connection, execute_query, get_db_cursor
from services.gemini_service import get_gemini_service
from services.model_service import ModelError, get_model_service
from services.recommendation_service import get_recommendations_for_prediction

logger = logging.getLogger(__name__)

# Bounded in-memory fallback cache for recent predictions (useful when MySQL is offline)
_RECENT_PREDICTIONS_CACHE: Dict[str, Dict[str, Any]] = {}
MAX_CACHE_ENTRIES = 100


class PredictionServiceError(Exception):
    """Raised when prediction orchestration or persistence encounters an unrecoverable error."""
    pass


def save_prediction_record(
    image_path: str,
    crop_name: str,
    disease_code: str,
    disease_name: str,
    confidence: float,
    model_version: str = "MobileNetV2-v1.0",
    user_id: Optional[int] = None,
    upload_id: Optional[str] = None,
) -> Tuple[bool, Optional[int], str]:
    """
    Saves a verified AI prediction into the MySQL 'predictions' table using parameterized SQL.
    Also updates the status of the matching upload in 'uploads' to 'analyzed'.

    Returns:
        (success: bool, prediction_id: Optional[int], message: str)
    """
    # Validate confidence constraint: 0.0000 <= confidence <= 1.0000
    if not (0.0 <= confidence <= 1.0):
        return False, None, f"Invalid confidence value {confidence}; must be in range [0.0, 1.0]."

    is_connected, db_msg = check_db_connection()
    if not is_connected:
        return False, None, f"Database offline: {db_msg}. Prediction was not persisted."

    insert_sql = """
        INSERT INTO predictions (
            user_id, image_path, crop_name, disease_code, disease_name,
            confidence, model_version
        ) VALUES (%s, %s, %s, %s, %s, %s, %s);
    """

    update_upload_sql = """
        UPDATE uploads
        SET status = 'analyzed'
        WHERE upload_id = %s;
    """

    dec_confidence = Decimal(str(round(confidence, 4)))

    try:
        with get_db_cursor(commit=True) as cursor:
            cursor.execute(
                insert_sql,
                (
                    user_id,
                    image_path,
                    crop_name,
                    disease_code,
                    disease_name,
                    dec_confidence,
                    model_version,
                ),
            )
            prediction_id = cursor.lastrowid

            if upload_id:
                try:
                    cursor.execute(update_upload_sql, (upload_id,))
                except Exception as err:
                    logger.warning("Could not update upload status for %s: %s", upload_id, err)

            logger.info("Saved prediction record id=%s for crop=%s, disease=%s", prediction_id, crop_name, disease_code)
            return True, prediction_id, "Prediction persisted successfully."

    except Exception as err:
        logger.error("Failed to insert prediction into MySQL: %s", err)
        return False, None, f"Database persistence failed: {err}"


def get_prediction_record(
    prediction_id: Union[int, str],
    current_user_id: Optional[int] = None,
) -> Tuple[Optional[Dict[str, Any]], bool]:
    """
    Retrieves a stored prediction by integer ID or alphanumeric identifier.
    Enforces authorization: if prediction belongs to a registered user,
    only that user may view it.

    Returns:
        (record_dict, is_forbidden)
        - If found and authorized: (dict, False)
        - If found but belongs to another user: (None, True)
        - If not found: (None, False)
    """
    str_id = str(prediction_id).strip()

    # 1. Check MySQL if online and identifier is integer ID
    is_connected, _ = check_db_connection()
    if is_connected and str_id.isdigit():
        query = """
            SELECT id, user_id, image_path, crop_name, disease_code, disease_name,
                   confidence, model_version, created_at
            FROM predictions
            WHERE id = %s
            LIMIT 1;
        """
        try:
            row = execute_query(query, (int(str_id),), fetchone=True)
            if row:
                # Ownership check: if record belongs to a user, check permission
                rec_user_id = row.get("user_id")
                if rec_user_id is not None:
                    if current_user_id is None or int(current_user_id) != int(rec_user_id):
                        logger.warning(
                            "Unauthorized access attempt to prediction id=%s by user_id=%s",
                            str_id,
                            current_user_id,
                        )
                        return None, True

                res = dict(row)
                res["confidence"] = float(res["confidence"])
                res["confidence_percent"] = round(res["confidence"] * 100, 2)
                res["is_healthy"] = "healthy" in res.get("disease_code", "").lower() or "healthy" in res.get("disease_name", "").lower()
                is_neem = res.get("disease_code") == "NEEM_DISEASE_UNSUPPORTED" or "neem" in res.get("crop_name", "").lower()
                res["is_neem"] = is_neem
                res["identified_plant"] = res.get("crop_name", "Unknown Plant")
                res["predicted_disease"] = res.get("disease_name", "Unknown Condition")
                res["is_plant_supported"] = not (res.get("disease_code") == "UNSUPPORTED_SPECIMEN")
                res["is_disease_supported"] = not (res.get("disease_code") in {"UNSUPPORTED_SPECIMEN", "NEEM_DISEASE_UNSUPPORTED"})
                if is_neem:
                    res["disease_status_message"] = "Neem identified, but neem disease detection is not yet supported"
                    res["confidence_meaning"] = "Specimen exhibits verified Azadirachta indica foliage morphology."
                elif res.get("disease_code") == "UNSUPPORTED_SPECIMEN":
                    res["disease_status_message"] = "Specimen cannot be reliably identified as any supported crop."
                    res["confidence_meaning"] = "Low model confidence with high entropy across known crop classes."
                else:
                    res["disease_status_message"] = f"Condition diagnosed in {res.get('crop_name')} leaf sample."
                    res["confidence_meaning"] = "Reliable statistical match to verified visual crop symptoms."

                rec_bundle = get_recommendations_for_prediction(res)
                res.update(rec_bundle)

                # Attach cached Gemini analysis if available
                if str_id in _RECENT_PREDICTIONS_CACHE and "gemini_analysis" in _RECENT_PREDICTIONS_CACHE[str_id]:
                    res["gemini_analysis"] = _RECENT_PREDICTIONS_CACHE[str_id]["gemini_analysis"]

                return res, False
        except Exception as err:
            logger.error("Error querying prediction id=%s: %s", str_id, err)

    # 2. Check memory fallback cache
    if str_id in _RECENT_PREDICTIONS_CACHE:
        cached = _RECENT_PREDICTIONS_CACHE[str_id]
        rec_user_id = cached.get("user_id")
        if rec_user_id is not None:
            if current_user_id is None or int(current_user_id) != int(rec_user_id):
                return None, True
        return cached, False

    return None, False


def run_prediction_pipeline(
    image_source: Union[str, Path, bytes, Any],
    stored_relative_path: str,
    upload_id: Optional[str] = None,
    user_id: Optional[int] = None,
    top_k: int = 3,
) -> Dict[str, Any]:
    """
    End-to-End Orchestrator:
      1. Executes inference using ModelService.
      2. Retrieves agronomic recommendations from MySQL.
      3. Persists prediction to MySQL 'predictions' table.
      4. Caches result in memory fallback cache.

    Returns:
        Structured result dictionary suitable for rendering in templates/results.html.
    """
    model_service = get_model_service()

    # Step 1: Forward-pass inference
    inference_result = model_service.predict_image(image_source, top_k=top_k)

    crop_name = inference_result["crop_name"]
    disease_code = inference_result["disease_code"]
    disease_name = inference_result["disease_name"]
    confidence = inference_result["confidence"]
    model_version = inference_result.get("model_version", "MobileNetV2-v1.0")

    # Step 2: Retrieve recommendations
    recommendation_bundle = get_recommendations_for_prediction(inference_result)

    # Step 3: Persist to MySQL
    persisted, prediction_id, persist_msg = save_prediction_record(
        image_path=stored_relative_path,
        crop_name=crop_name,
        disease_code=disease_code,
        disease_name=disease_name,
        confidence=confidence,
        model_version=model_version,
        user_id=user_id,
        upload_id=upload_id,
    )

    # Step 4: Supplementary Gemini Multimodal Analysis (non-blocking, failure-shielded)
    gemini_service = get_gemini_service()
    gemini_analysis: Dict[str, Any]
    try:
        gemini_analysis = gemini_service.analyze_leaf_image(
            image_source=image_source,
            cnn_context=inference_result,
        )
    except Exception as gem_err:
        logger.warning("Gemini supplementary analysis encountered error: %s", gem_err)
        gemini_analysis = {
            "status": "error",
            "source": "Gemini Multimodal AI (Supplementary)",
            "identified_plant_species": "Analysis Unavailable",
            "visual_symptoms": "Supplementary analysis encountered an unexpected exception.",
            "suspected_condition": "Inconclusive",
            "certainty_level": "Unavailable",
            "is_leaf_clear": False,
            "botanical_observations": f"Service exception: {gem_err}",
            "recommended_next_step": "Rely on the CNN primary diagnosis.",
            "disclaimer": "Local CNN (MobileNetV2) classification remains active.",
        }

    # Combined composite payload
    final_result: Dict[str, Any] = {
        **inference_result,
        **recommendation_bundle,
        "image_path": stored_relative_path,
        "upload_id": upload_id,
        "user_id": user_id,
        "persisted": persisted,
        "prediction_id": prediction_id,
        "persist_message": persist_msg,
        "gemini_analysis": gemini_analysis,
    }

    # Cache for direct view retrieval
    cache_key = str(prediction_id) if prediction_id is not None else (upload_id or stored_relative_path)
    if len(_RECENT_PREDICTIONS_CACHE) >= MAX_CACHE_ENTRIES:
        # Evict oldest entry
        first_key = next(iter(_RECENT_PREDICTIONS_CACHE))
        _RECENT_PREDICTIONS_CACHE.pop(first_key, None)
    _RECENT_PREDICTIONS_CACHE[str(cache_key)] = final_result

    if upload_id and str(upload_id) != str(cache_key):
        _RECENT_PREDICTIONS_CACHE[str(upload_id)] = final_result

    return final_result


def run_gemini_analysis_for_record(
    prediction_id: Union[int, str],
    current_user_id: Optional[int] = None,
) -> Dict[str, Any]:
    """
    On-demand or asynchronous trigger for Gemini supplementary analysis on an
    existing prediction record. Thread-safe and updates the in-memory cache.
    """
    record, is_forbidden = get_prediction_record(prediction_id, current_user_id=current_user_id)
    if is_forbidden:
        return {
            "status": "forbidden",
            "error": "You do not have authorization to analyze this diagnostic report.",
        }

    if not record:
        return {
            "status": "not_found",
            "error": f"Prediction reference '{prediction_id}' was not found.",
        }

    # If already cached and successful, return cached analysis
    if "gemini_analysis" in record and record["gemini_analysis"].get("status") == "success":
        return record["gemini_analysis"]

    # Locate image file
    rel_path = record.get("image_path", "")
    full_path = Path(Config.UPLOAD_FOLDER) / Path(rel_path).name

    if not full_path.is_file():
        return {
            "status": "file_not_found",
            "error": "The image file for this prediction could not be located on disk.",
        }

    gemini_service = get_gemini_service()
    gemini_result = gemini_service.analyze_leaf_image(
        image_source=full_path,
        cnn_context=record,
    )

    # Update in-memory record cache
    record["gemini_analysis"] = gemini_result
    cache_key = str(prediction_id)
    _RECENT_PREDICTIONS_CACHE[cache_key] = record

    return gemini_result
