"""
Gemini Multimodal Image Analysis Service (services/gemini_service.py)
=====================================================================
Integrates Google Gemini multimodal vision capabilities as a supplementary
image analysis layer for the Plant Disease Detection and Smart Fertilizer
Recommendation System.

Architectural Principles:
- Supplementary Role: The local CNN (MobileNetV2) is the primary classifier
  for its supported 38 classes. Gemini provides supplementary botanical
  identification and physical symptom descriptions.
- No Fabricated Probabilities: Gemini certainty is strictly qualitative
  ("High", "Moderate", "Low / Inconclusive") and is NEVER combined with CNN
  softmax probabilities.
- Safety & Agronomic Guardrails: Gemini is constrained from prescribing
  chemical dosages, active ingredient rates, or regulatory approvals. All
  chemical management and fertilizer guidance remain grounded in the vetted
  MySQL database and agronomic catalog.
- Security: GEMINI_API_KEY is loaded from .env and NEVER printed, logged, or
  transmitted to the client frontend.
- Robustness: Network timeouts, rate limits (429), server load (503), and
  corrupt image handling with graceful fallback.
"""

import base64
import json
import logging
import re
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import requests
from PIL import Image, UnidentifiedImageError

from config import Config

logger = logging.getLogger(__name__)

# Fallback Gemini models in order of preference if primary experiences high demand
FALLBACK_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.8-flash",
    "gemini-3.1-flash-lite",
    "gemini-flash-latest",
]

_GEMINI_SERVICE_INSTANCE: Optional["GeminiService"] = None


class GeminiService:
    """
    Manages multimodal leaf image analysis via Google Gemini API.
    Thread-safe and decoupled from the local CNN inference pipeline.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: Optional[int] = None,
        enabled: Optional[bool] = None,
    ):
        if api_key is not None:
            self._api_key = api_key.strip()
        else:
            self._api_key = (Config.GEMINI_API_KEY or "").strip()

        if model_name is not None:
            self._model_name = model_name.strip()
        else:
            self._model_name = (Config.GEMINI_MODEL or "gemini-3.5-flash-lite").strip()

        self._timeout = timeout if timeout is not None else Config.GEMINI_TIMEOUT_SECONDS
        self._enabled = enabled if enabled is not None else Config.GEMINI_ENABLED

    @property
    def is_configured(self) -> bool:
        """Checks if the service is enabled and has a non-empty API key."""
        return bool(self._enabled and self._api_key and len(self._api_key) > 10)

    @property
    def model_name(self) -> str:
        """Returns the configured primary Gemini model identifier."""
        return self._model_name

    def _prepare_image_payload(
        self, image_source: Union[str, Path, bytes, Image.Image], max_dimension: int = 800
    ) -> Tuple[str, str]:
        """
        Validates, downsamples, and encodes the image into base64 JPEG format.
        Downsampling prevents oversized HTTP payloads and connection timeouts.

        Returns:
            Tuple of (base64_encoded_data, mime_type)
        """
        if isinstance(image_source, Image.Image):
            img = image_source.copy()
        elif isinstance(image_source, bytes):
            img = Image.open(BytesIO(image_source))
        elif isinstance(image_source, (str, Path)):
            p = Path(image_source)
            if not p.is_file():
                raise FileNotFoundError(f"Image file not found: {p}")
            img = Image.open(p)
        else:
            raise ValueError(f"Unsupported image source type: {type(image_source)}")

        img.verify()

        # Re-open after verify to perform operations
        if isinstance(image_source, (str, Path)):
            img = Image.open(image_source)
        elif isinstance(image_source, bytes):
            img = Image.open(BytesIO(image_source))

        # Convert to RGB (handles RGBA, Palette, Grayscale)
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Downscale proportionally to fit within max_dimension
        img.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)

        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=85, optimize=True)
        b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return b64_str, "image/jpeg"

    def _build_prompt(self, cnn_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Builds the strict botanical instruction prompt for Gemini.
        Explicitly instructs the model to describe visible symptoms and identify
        the plant species while strictly withholding chemical dosage advice.
        """
        context_str = ""
        if cnn_context:
            crop = cnn_context.get("crop_name")
            cond = cnn_context.get("disease_name")
            is_supp = cnn_context.get("is_supported", True)
            if is_supp and crop and cond:
                context_str = f" The primary CNN statistical classifier noted possible correlation with {crop} ({cond})."
            else:
                context_str = " Note: The primary CNN classifier flagged this specimen as out-of-distribution or inconclusive."

        return (
            "You are a certified agricultural botanical analyst assisting an educational "
            "Plant Disease Detection System."
            f"{context_str}\n"
            "Examine the provided plant leaf photograph and return a structured JSON evaluation matching these exact fields:\n"
            "{\n"
            '  "identified_plant_species": "Botanical and common name of the plant species (e.g. Azadirachta indica (Neem), Malus domestica (Apple), or Unknown / Unclear)",\n'
            '  "visual_symptoms": "Detailed factual description of visible foliar symptoms (e.g. necrotic lesions, chlorotic halo, pustules, leaf curling, fungal sporulation, or healthy tissue)",\n'
            '  "suspected_condition": "Suspected foliar pathology, physiological disorder, pest damage, or Healthy foliage",\n'
            '  "certainty_level": "High", "Moderate", or "Low / Inconclusive",\n'
            '  "is_leaf_clear": true or false (false if blurry, dark, ambiguous, or no leaf detected),\n'
            '  "botanical_observations": "1-2 concise, factual sentences summarizing the morphological and pathology observations.",\n'
            '  "recommended_next_step": "Agronomic advice such as Clearer photo required, Consult local agricultural extension, or Routine plant sanitation."\n'
            "}\n\n"
            "STRICT SAFETY & AGRONOMIC CONSTRAINTS:\n"
            "- Do NOT prescribe chemical pesticide dosages, fungicide chemical names, active ingredient percentages, application schedules, waiting periods, or regulatory safety claims.\n"
            "- Do NOT invent fertilizer mix numbers (e.g., NPK ratios).\n"
            "- If the plant or leaf is unfamiliar, out of focus, or ambiguous, explicitly set certainty_level to 'Low / Inconclusive' and is_leaf_clear to false.\n"
            "Respond ONLY with the raw JSON object, without introductory text or markdown formatting."
        )

    def _execute_api_call(
        self,
        model_name: str,
        b64_data: str,
        mime_type: str,
        prompt_text: str,
    ) -> requests.Response:
        """Executes the HTTP POST request to the Google Gemini generateContent REST endpoint."""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={self._api_key}"
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt_text},
                        {
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": b64_data,
                            }
                        },
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 800,
                "responseMimeType": "application/json",
            },
        }

        return requests.post(url, json=payload, timeout=self._timeout)

    def analyze_leaf_image(
        self,
        image_source: Union[str, Path, bytes, Image.Image],
        cnn_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Performs supplementary multimodal botanical analysis on a leaf image.

        Args:
            image_source: Image file path, raw bytes, or Pillow Image instance.
            cnn_context: Optional dictionary containing CNN primary classification findings.

        Returns:
            Dictionary containing:
              - status: "success", "not_configured", "error", "timeout", "rate_limited", or "unavailable"
              - source: "Gemini Multimodal AI (Supplementary)"
              - identified_plant_species: str
              - visual_symptoms: str
              - suspected_condition: str
              - certainty_level: str ("High", "Moderate", or "Low / Inconclusive")
              - is_leaf_clear: bool
              - botanical_observations: str
              - recommended_next_step: str
              - disclaimer: str
              - error_message: Optional[str]
        """
        if not self.is_configured:
            return {
                "status": "not_configured",
                "source": "Gemini Multimodal AI (Supplementary)",
                "identified_plant_species": "Supplementary AI Unconfigured",
                "visual_symptoms": "Gemini API key is not configured in .env.",
                "suspected_condition": "Analysis Unavailable",
                "certainty_level": "Unavailable",
                "is_leaf_clear": False,
                "botanical_observations": "To activate Gemini supplementary botanical analysis, add your GEMINI_API_KEY in the project .env file.",
                "recommended_next_step": "Rely on the local CNN model or add GEMINI_API_KEY to .env.",
                "disclaimer": "Local CNN (MobileNetV2) operates independently as the primary diagnostic classifier.",
            }

        # 1. Image preparation & validation
        try:
            b64_data, mime_type = self._prepare_image_payload(image_source)
        except (UnidentifiedImageError, ValueError) as err:
            logger.warning("Gemini image preparation rejected input: %s", err)
            return {
                "status": "invalid_image",
                "source": "Gemini Multimodal AI (Supplementary)",
                "identified_plant_species": "Invalid Image",
                "visual_symptoms": "The uploaded file could not be parsed as a valid botanical image.",
                "suspected_condition": "Inconclusive Specimen",
                "certainty_level": "Low / Inconclusive",
                "is_leaf_clear": False,
                "botanical_observations": f"Image processing failed: {err}",
                "recommended_next_step": "Upload a valid JPG, PNG, or WEBP leaf photograph.",
                "disclaimer": "Input verification failed prior to AI inspection.",
            }
        except FileNotFoundError as err:
            logger.error("Image file missing for Gemini analysis: %s", err)
            return {
                "status": "file_not_found",
                "source": "Gemini Multimodal AI (Supplementary)",
                "identified_plant_species": "File Not Found",
                "visual_symptoms": "Target image file was not found on disk.",
                "suspected_condition": "Unavailable",
                "certainty_level": "Unavailable",
                "is_leaf_clear": False,
                "botanical_observations": str(err),
                "recommended_next_step": "Re-upload the leaf image.",
                "disclaimer": "Diagnostic pipeline input missing.",
            }

        # 2. Build targeted prompt
        prompt = self._build_prompt(cnn_context)

        # 3. Call API with single fast fallback (fail fast on timeout)
        primary_model = self._model_name
        fallback_model = "gemini-3.8-flash" if primary_model != "gemini-3.8-flash" else "gemini-3.5-flash-lite"
        models_to_try = [primary_model, fallback_model]
        last_error = "Unknown error"

        for idx, model_candidate in enumerate(models_to_try):
            try:
                logger.info("Executing Gemini supplementary analysis with model: %s", model_candidate)
                resp = self._execute_api_call(model_candidate, b64_data, mime_type, prompt)

                if resp.status_code == 200:
                    data = resp.json()
                    candidates = data.get("candidates", [])
                    if not candidates:
                        last_error = "Gemini returned zero response candidates (possible content filter)."
                        continue

                    raw_text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                    clean_text = raw_text.strip()
                    if clean_text.startswith("```"):
                        clean_text = re.sub(r"^```(?:json)?\s*", "", clean_text)
                        clean_text = re.sub(r"\s*```$", "", clean_text)

                    parsed = json.loads(clean_text)

                    plant_species = str(parsed.get("identified_plant_species") or "Unknown Species").strip()
                    symptoms = str(parsed.get("visual_symptoms") or "No distinct foliar symptoms described.").strip()
                    suspected = str(parsed.get("suspected_condition") or "Inconclusive").strip()
                    certainty = str(parsed.get("certainty_level") or "Moderate").strip()
                    is_clear = bool(parsed.get("is_leaf_clear", True))
                    observations = str(parsed.get("botanical_observations") or "").strip()
                    next_step = str(parsed.get("recommended_next_step") or "Inspect with local extension expert.").strip()

                    # Safety check: sanitize against chemical recommendations
                    banned_terms = ["apply", "dose", "per liter", "spray rate", "mix ratio", "grams per"]
                    for term in banned_terms:
                        if term in symptoms.lower() or term in observations.lower():
                            logger.warning("Sanitizing chemical recommendation from Gemini text.")
                            observations = re.sub(r"(?i)apply.*?(\.|$)", "", observations).strip()

                    return {
                        "status": "success",
                        "source": "Gemini Multimodal AI (Supplementary)",
                        "model": model_candidate,
                        "identified_plant_species": plant_species,
                        "visual_symptoms": symptoms,
                        "suspected_condition": suspected,
                        "certainty_level": certainty,
                        "is_leaf_clear": is_clear,
                        "botanical_observations": observations,
                        "recommended_next_step": next_step,
                        "disclaimer": (
                            "Supplementary visual observation from Gemini Multimodal AI. Independent of "
                            "CNN statistical classification. Agronomic pesticide and fertilizer recommendations "
                            "are strictly derived from verified extension databases."
                        ),
                    }

                elif resp.status_code == 429:
                    logger.warning("Gemini API rate limit (429) on model %s", model_candidate)
                    return {
                        "status": "rate_limited",
                        "source": "Gemini Multimodal AI (Supplementary)",
                        "identified_plant_species": "Rate Limit Exceeded",
                        "visual_symptoms": "Gemini API request quota has been reached.",
                        "suspected_condition": "Analysis Temporarily Unavailable",
                        "certainty_level": "Unavailable",
                        "is_leaf_clear": True,
                        "botanical_observations": "The Gemini API quota is temporarily exhausted. Please wait a moment and try again.",
                        "recommended_next_step": "Rely on the CNN primary diagnosis, which is fully operational.",
                        "disclaimer": "Local CNN (MobileNetV2) classification remains active and unaffected.",
                    }

                elif resp.status_code in (503, 404):
                    logger.warning("Gemini API returned status %d on %s, trying fallback...", resp.status_code, model_candidate)
                    last_error = f"Gemini model {model_candidate} returned {resp.status_code}."
                    continue

                else:
                    err_msg = resp.json().get("error", {}).get("message", resp.text[:120]) if resp.text else f"HTTP {resp.status_code}"
                    logger.warning("Gemini API error (HTTP %d) on model %s: %s", resp.status_code, model_candidate, err_msg)
                    last_error = f"Gemini API returned status {resp.status_code}: {err_msg}"
                    continue

            except requests.exceptions.Timeout:
                logger.warning("Gemini request timed out on model %s (timeout=%ds)", model_candidate, self._timeout)
                last_error = f"Analysis request timed out after {self._timeout} seconds."
                # On timeout, do not block the pipeline with further attempts
                break
            except requests.exceptions.RequestException as net_err:
                logger.warning("Gemini network error on model %s: %s", model_candidate, net_err)
                last_error = f"Network connection failed: {net_err}"
                break
            except (json.JSONDecodeError, KeyError) as parse_err:
                logger.warning("Failed parsing Gemini JSON on model %s: %s", model_candidate, parse_err)
                last_error = "Failed to parse structured response from Gemini."
                continue

        # If all candidates failed:
        return {
            "status": "unavailable",
            "source": "Gemini Multimodal AI (Supplementary)",
            "identified_plant_species": "Service Unavailable",
            "visual_symptoms": "Supplementary visual analysis could not be completed at this time.",
            "suspected_condition": "Inconclusive",
            "certainty_level": "Low / Inconclusive",
            "is_leaf_clear": False,
            "botanical_observations": f"Supplementary AI notice: {last_error}",
            "recommended_next_step": "Refer to the CNN classification report and consult a local agricultural extension expert.",
            "disclaimer": "The CNN primary classifier remains fully operational and unaffected by external API status.",
        }


def get_gemini_service() -> GeminiService:
    """Returns the singleton instance of GeminiService."""
    global _GEMINI_SERVICE_INSTANCE
    if _GEMINI_SERVICE_INSTANCE is None:
        _GEMINI_SERVICE_INSTANCE = GeminiService()
    return _GEMINI_SERVICE_INSTANCE
