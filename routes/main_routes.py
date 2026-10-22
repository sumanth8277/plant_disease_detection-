"""
Main Application Routes Module (Phase 2)
Provides endpoints for homepage, diagnostic health probes, secure image upload,
upload success view, controlled image serving, and accessible error handling.
"""

import logging
import re
from pathlib import Path

from flask import (
    Flask,
    abort,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
    session,
    url_for,
)
from werkzeug.exceptions import RequestEntityTooLarge

from config import Config
from database.db import check_db_connection
from services.gemini_service import get_gemini_service
from services.model_service import get_model_service, ModelError
from services.prediction_service import (
    get_prediction_record,
    run_gemini_analysis_for_record,
    run_prediction_pipeline,
)
from services.upload_service import (
    DatabaseStorageError,
    UploadValidationError,
    get_upload_metadata,
    validate_and_process_image,
)

logger = logging.getLogger(__name__)

# Strict regex for controlled image retrieval (32 hex characters + .jpg or .png)
SAFE_FILENAME_REGEX = re.compile(r"^[a-f0-9]{32}\.(jpg|png)$")
# Alphanumeric validator for prediction/upload references
SAFE_ID_REGEX = re.compile(r"^[a-zA-Z0-9_\-]+$")


def register_routes(app: Flask) -> None:
    """Registers all web interface, API, and diagnostic endpoints on the Flask application."""

    @app.route("/", methods=["GET"])
    def index():
        """Renders the comprehensive project homepage with the embedded upload interface."""
        return render_template("index.html")

    @app.route("/health", methods=["GET"])
    def health():
        """
        Application-level health endpoint.
        Returns 200 OK if Flask is running. Never exposes secrets.
        """
        model_ready, _ = get_model_service().is_model_ready()
        return jsonify(
            {
                "status": "healthy",
                "service": "Plant Disease Detection System",
                "phase": "Phase 5 - Authentication, History Dashboard & Final Polish",
                "model_ready": model_ready,
                "gemini_configured": get_gemini_service().is_configured,
                "debug": app.config.get("DEBUG", False),
            }
        ), 200

    @app.route("/health/db", methods=["GET"])
    def health_db():
        """
        Database connectivity diagnostic endpoint.
        Executes a safe ping (SELECT 1) and returns safe status without secrets.
        Returns 200 if connected, 503 if MySQL is unavailable or misconfigured.
        """
        is_connected, message = check_db_connection()
        response_payload = {
            "status": "healthy" if is_connected else "unavailable",
            "database": app.config.get("DB_NAME"),
            "host": app.config.get("DB_HOST"),
            "port": app.config.get("DB_PORT"),
            "message": message,
        }
        status_code = 200 if is_connected else 503
        return jsonify(response_payload), status_code

    @app.route("/health/model", methods=["GET"])
    def health_model():
        """
        Model readiness diagnostic endpoint (Phase 4).
        Returns model loading status, version, and number of supported classes.
        Never exposes filesystem secrets or paths.
        """
        model_service = get_model_service()
        is_ready, message = model_service.is_model_ready()
        meta = model_service.get_model_metadata()

        payload = {
            "status": "ready" if is_ready else "unavailable",
            "message": message,
            "num_classes": meta.get("num_classes", 0),
            "model_version": meta.get("metadata", {}).get("model_version", "MobileNetV2-v1.0"),
        }
        status_code = 200 if is_ready else 503
        return jsonify(payload), status_code

    @app.route("/health/gemini", methods=["GET"])
    def health_gemini():
        """
        Gemini supplementary service diagnostic endpoint.
        Returns configuration status and active model. Never exposes secrets or API keys.
        """
        gemini_service = get_gemini_service()
        payload = {
            "status": "configured" if gemini_service.is_configured else "not_configured",
            "model": gemini_service.model_name,
            "role": "supplementary_multimodal_analysis",
            "primary_classifier": "MobileNetV2 (CNN)",
        }
        return jsonify(payload), 200

    @app.route("/api/gemini/analyze/<path:prediction_id>", methods=["POST"])
    def api_gemini_analyze(prediction_id: str):
        """
        Triggers or fetches Gemini supplementary analysis for an existing prediction record.
        Validates ID format and permissions, returning structured analysis JSON.
        """
        clean_id = str(prediction_id).strip()
        if not SAFE_ID_REGEX.match(clean_id):
            return jsonify({"success": False, "error": "Invalid prediction reference format."}), 400

        result = run_gemini_analysis_for_record(clean_id, current_user_id=session.get("user_id"))
        if result.get("status") == "forbidden":
            return jsonify({"success": False, "error": result.get("error", "Access denied.")}), 403
        if result.get("status") in {"not_found", "file_not_found"}:
            return jsonify({"success": False, "error": result.get("error", "Record not found.")}), 404

        return jsonify({"success": True, "gemini_analysis": result}), 200

    @app.route("/upload", methods=["GET", "POST"])
    def upload():
        """
        Handles plant leaf image uploads.
        GET: Renders dedicated upload page.
        POST: Validates and saves leaf image.
              If model is loaded: runs inference, fetches recommendations, persists history, redirects to /results/<id>.
              If model is not loaded: saves upload, flashes informative guidance, redirects to /upload/success/<id>.
        """
        if request.method == "GET":
            return render_template("upload.html")

        # Determine if caller expects JSON (AJAX submission)
        is_ajax = (
            request.headers.get("X-Requested-With") == "XMLHttpRequest"
            or "application/json" in request.headers.get("Accept", "")
        )

        file_storage = request.files.get("leaf_image") or request.files.get("file")

        if not file_storage or not file_storage.filename:
            err_msg = "Please select a plant leaf image to upload."
            if is_ajax:
                return jsonify({"success": False, "error": err_msg}), 400
            flash(err_msg, "danger")
            return render_template("upload.html"), 400

        try:
            upload_folder = Path(app.config.get("UPLOAD_FOLDER", Config.UPLOAD_FOLDER))
            metadata = validate_and_process_image(file_storage, upload_folder=upload_folder)

            # Check Model Readiness
            model_service = get_model_service()
            is_ready, ready_msg = model_service.is_model_ready()
            if not is_ready:
                # Attempt lazy loading on-demand
                try:
                    model_service.load_model_artifacts()
                    is_ready, ready_msg = model_service.is_model_ready()
                except Exception:
                    pass

            if is_ready:
                leaf_path = upload_folder / metadata["stored_filename"]
                try:
                    pipeline_result = run_prediction_pipeline(
                        image_source=leaf_path,
                        stored_relative_path=metadata["relative_path"],
                        upload_id=metadata["upload_id"],
                        user_id=session.get("user_id"),
                    )

                    pred_ref = pipeline_result.get("prediction_id") or metadata["upload_id"]
                    redirect_target = url_for("view_result", prediction_id=pred_ref)

                    if is_ajax:
                        return jsonify(
                            {
                                "success": True,
                                "upload_id": metadata["upload_id"],
                                "prediction_id": pipeline_result.get("prediction_id"),
                                "redirect_url": redirect_target,
                                "diagnosis": {
                                    "crop_name": pipeline_result["crop_name"],
                                    "disease_name": pipeline_result["disease_name"],
                                    "confidence_percent": pipeline_result["confidence_percent"],
                                    "is_healthy": pipeline_result["is_healthy"],
                                },
                            }
                        ), 200

                    return redirect(redirect_target)

                except Exception as pred_err:
                    logger.error("Inference execution failed on upload %s: %s", metadata["upload_id"], pred_err)
                    # If inference fails unexpectedly, do not fake results; redirect to upload success with notice
                    flash(
                        f"Image uploaded successfully, but automated diagnosis could not complete: {pred_err}",
                        "warning",
                    )
                    redirect_target = url_for("upload_success", upload_id=metadata["upload_id"])
                    if is_ajax:
                        return jsonify({"success": True, "redirect_url": redirect_target}), 200
                    return redirect(redirect_target)

            else:
                # Model weights are not loaded (e.g. model offline)
                redirect_target = url_for("upload_success", upload_id=metadata["upload_id"])
                flash(
                    "Image uploaded successfully. Note: Automated AI diagnosis model is initializing or currently offline.",
                    "info",
                )

                if is_ajax:
                    return jsonify(
                        {
                            "success": True,
                            "upload_id": metadata["upload_id"],
                            "stored_filename": metadata["stored_filename"],
                            "model_ready": False,
                            "redirect_url": redirect_target,
                        }
                    ), 200

                return redirect(redirect_target)

        except UploadValidationError as val_err:
            logger.warning("Upload validation rejected file: %s", val_err)
            if is_ajax:
                return jsonify({"success": False, "error": str(val_err)}), 400
            flash(str(val_err), "danger")
            return render_template("upload.html"), 400

        except DatabaseStorageError as db_err:
            logger.error("Upload database storage error: %s", db_err)
            if is_ajax:
                return jsonify({"success": False, "error": str(db_err)}), 503
            flash(str(db_err), "warning")
            return render_template("upload.html"), 503

        except Exception as exc:
            logger.exception("Unexpected error during image upload: %s", exc)
            generic_err = "An unexpected error occurred while processing your upload. Please try again."
            if is_ajax:
                return jsonify({"success": False, "error": generic_err}), 500
            flash(generic_err, "danger")
            return render_template("upload.html"), 500

    @app.route("/results/<prediction_id>", methods=["GET"])
    def view_result(prediction_id: str):
        """
        Displays diagnostic prediction results, classification confidence,
        disease symptoms, IPM treatments, and fertilizer schedules (Phase 4).
        """
        clean_id = str(prediction_id).strip()
        if not SAFE_ID_REGEX.match(clean_id):
            abort(404)

        current_user_id = session.get("user_id")
        result_data, is_forbidden = get_prediction_record(clean_id, current_user_id=current_user_id)

        if is_forbidden:
            return render_template(
                "error.html",
                error_title="Access Denied",
                error_code=403,
                error_message="You do not have authorization to view this private diagnostic report.",
            ), 403

        if not result_data:
            return render_template(
                "error.html",
                error_title="Diagnosis Result Not Found",
                error_code=404,
                error_message=(
                    f"The requested diagnosis reference '{clean_id}' could not be found. "
                    "It may have expired or was never generated."
                ),
            ), 404

        return render_template("results.html", result=result_data)

    @app.route("/upload/success/<upload_id>", methods=["GET"])
    def upload_success(upload_id: str):
        """
        Displays the upload success view following Post/Redirect/Get pattern.
        Shows image preview, image dimensions, file size, and Phase 2 status.
        """
        metadata = get_upload_metadata(upload_id)
        if not metadata:
            return render_template(
                "error.html",
                error_title="Upload Not Found",
                error_code=404,
                error_message=(
                    f"The requested upload identifier '{upload_id}' could not be located. "
                    "It may have expired or was never created."
                ),
            ), 404

        return render_template("upload_success.html", upload=metadata)

    @app.route("/uploads/<filename>", methods=["GET"])
    def serve_upload(filename: str):
        """
        Controlled route for safely serving uploaded leaf images.
        Enforces strict alphanumeric filename patterns, security headers,
        and prevents directory traversal.
        """
        if not SAFE_FILENAME_REGEX.match(filename):
            abort(404)

        upload_dir = Path(app.config.get("UPLOAD_FOLDER", Config.UPLOAD_FOLDER))
        target_file = (upload_dir / filename).resolve()

        if not target_file.is_file() or not str(target_file).startswith(str(upload_dir.resolve())):
            abort(404)

        response = send_from_directory(str(upload_dir), filename)
        # Add security and cache control headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Cache-Control"] = "public, max-age=86400"
        return response

    # --------------------------------------------------------------------------
    # Error Handlers
    # --------------------------------------------------------------------------

    @app.errorhandler(413)
    @app.errorhandler(RequestEntityTooLarge)
    def handle_large_file(error):
        """Handles files exceeding MAX_CONTENT_LENGTH (5 MB)."""
        max_mb = app.config.get("MAX_CONTENT_LENGTH", 5 * 1024 * 1024) / (1024 * 1024)
        err_msg = f"The uploaded file is too large. Maximum allowed file size is {int(max_mb)} MB."

        if (
            request.headers.get("X-Requested-With") == "XMLHttpRequest"
            or "application/json" in request.headers.get("Accept", "")
        ):
            return jsonify({"success": False, "error": err_msg}), 413

        flash(err_msg, "danger")
        return (
            render_template(
                "error.html",
                error_title="File Size Exceeded",
                error_code=413,
                error_message=err_msg,
            ),
            413,
        )

    @app.errorhandler(404)
    def handle_not_found(error):
        """Handles 404 Not Found cleanly."""
        return (
            render_template(
                "error.html",
                error_title="Page Not Found",
                error_code=404,
                error_message="The requested page or image could not be found.",
            ),
            404,
        )

    @app.errorhandler(500)
    def handle_server_error(error):
        """Handles unexpected 500 errors without exposing tracebacks."""
        logger.error("Internal Server Error encountered: %s", error)
        return (
            render_template(
                "error.html",
                error_title="Internal Server Error",
                error_code=500,
                error_message="An unexpected system error occurred. Please try again later.",
            ),
            500,
        )
