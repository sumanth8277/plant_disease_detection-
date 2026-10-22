"""
Upload Service Module
Handles secure image validation, Pillow verification, safe filesystem storage,
and MySQL metadata persistence with automatic rollback on database failure.
"""

import logging
import os
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from PIL import Image, UnidentifiedImageError
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

from config import Config
from database.db import check_db_connection, execute_query, get_db_cursor

logger = logging.getLogger(__name__)

# Valid image MIME types and corresponding extensions
MIME_MAP = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
}


class UploadValidationError(Exception):
    """Raised when an uploaded file fails validation."""
    pass


class DatabaseStorageError(Exception):
    """Raised when saving upload metadata to MySQL fails."""
    pass


def validate_and_process_image(
    file_storage: FileStorage,
    upload_folder: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Validates the uploaded image file on the server side:
      1. Confirms file existence and non-empty filename.
      2. Validates extension against ALLOWED_EXTENSIONS.
      3. Verifies image integrity and format using Pillow.
      4. Validates dimensions against decompression bomb limits.
      5. Normalizes image to RGB mode.
      6. Generates a cryptographically random, safe filename.
      7. Saves image to upload directory safely.
      8. Persists metadata in MySQL 'uploads' table. If DB fails, cleans up the file.

    Returns:
        dict containing upload metadata (upload_id, stored_filename, original_filename,
        relative_path, mime_type, file_size, image_width, image_height).

    Raises:
        UploadValidationError: on any validation or format issue.
        DatabaseStorageError: on MySQL persistence failure (file is safely deleted).
    """
    if upload_folder is None:
        upload_folder = Path(Config.UPLOAD_FOLDER)

    # 1. Check file object
    if not file_storage or not file_storage.filename:
        raise UploadValidationError("No file was selected for upload.")

    raw_filename = file_storage.filename.strip()
    if not raw_filename:
        raise UploadValidationError("Empty filename is not allowed.")

    # 2. Check extension
    if "." not in raw_filename:
        raise UploadValidationError(
            "File has no extension. Only JPG, JPEG, and PNG images are supported."
        )

    ext = raw_filename.rsplit(".", 1)[1].lower()
    if ext not in Config.ALLOWED_EXTENSIONS:
        raise UploadValidationError(
            f"Unsupported file format '.{ext}'. Supported formats are: "
            f"{', '.join(sorted(Config.ALLOWED_EXTENSIONS)).upper()}."
        )

    # Normalize extension (e.g. .jpeg -> .jpg for uniform storage)
    target_ext = "jpg" if ext in ("jpg", "jpeg") else "png"

    # 3. Pillow Image Verification
    # Ensure stream is at position 0
    file_storage.stream.seek(0)

    try:
        # Step A: Quick integrity check
        with Image.open(file_storage.stream) as img:
            img.verify()
    except (UnidentifiedImageError, OSError, SyntaxError) as err:
        logger.warning("Corrupt or invalid image uploaded (%s): %s", raw_filename, err)
        raise UploadValidationError(
            "The uploaded file is not a valid or readable image. Please upload an intact image."
        )

    # Step B: Reset stream and re-open to inspect format and dimensions
    file_storage.stream.seek(0)
    try:
        with Image.open(file_storage.stream) as img:
            detected_format = img.format
            if detected_format not in MIME_MAP:
                raise UploadValidationError(
                    f"Unsupported image format '{detected_format}'. Only JPEG and PNG images are accepted."
                )

            mime_type = MIME_MAP[detected_format]
            width, height = img.size

            # 4. Limit dimensions and total pixels to protect against decompression bombs
            if width > Config.MAX_IMAGE_DIMENSION or height > Config.MAX_IMAGE_DIMENSION:
                raise UploadValidationError(
                    f"Image dimensions ({width}x{height}) exceed maximum allowed dimension "
                    f"({Config.MAX_IMAGE_DIMENSION}px)."
                )

            total_pixels = width * height
            if total_pixels > Config.MAX_IMAGE_PIXELS:
                raise UploadValidationError(
                    f"Image total pixels ({total_pixels}) exceed maximum allowed limit."
                )

            # Convert to RGB mode if needed (RGBA or Palette to RGB for JPEG)
            if img.mode in ("RGBA", "P", "LA", "CMYK"):
                rgb_image = img.convert("RGB")
            else:
                rgb_image = img.copy()

    except UploadValidationError:
        raise
    except Exception as err:
        logger.error("Error inspecting image with Pillow: %s", err)
        raise UploadValidationError(
            "Unable to process the image. The file may be damaged."
        )

    # 5. Generate secure, random filename (UUID-based)
    upload_id = uuid.uuid4().hex
    stored_filename = f"{upload_id}.{target_ext}"

    # Sanitized original filename for display purposes only
    safe_orig_name = secure_filename(raw_filename)
    if not safe_orig_name:
        safe_orig_name = f"unnamed_image.{target_ext}"
    if len(safe_orig_name) > 100:
        safe_orig_name = safe_orig_name[:96] + f".{target_ext}"

    # 6. Safe Path resolution and Path Traversal prevention
    upload_folder.mkdir(parents=True, exist_ok=True)
    target_path = (upload_folder / stored_filename).resolve()
    resolved_folder = upload_folder.resolve()

    if not str(target_path).startswith(str(resolved_folder)):
        raise UploadValidationError("Invalid destination path (path traversal prevented).")

    # 7. Save file to disk
    try:
        if target_ext == "jpg":
            rgb_image.save(target_path, format="JPEG", quality=90, optimize=True)
        else:
            rgb_image.save(target_path, format="PNG", optimize=True)
        file_size = target_path.stat().st_size
    except Exception as err:
        logger.error("Filesystem write error: %s", err)
        if target_path.exists():
            try:
                target_path.unlink()
            except OSError:
                pass
        raise UploadValidationError(
            "An error occurred while saving the image to disk. Please try again."
        )

    relative_path = f"uploads/{stored_filename}"

    metadata = {
        "upload_id": upload_id,
        "stored_filename": stored_filename,
        "original_filename": safe_orig_name,
        "relative_path": relative_path,
        "mime_type": mime_type,
        "file_size": file_size,
        "image_width": width,
        "image_height": height,
        "status": "pending_analysis",
    }

    # 8. Save upload metadata into MySQL (if DB is configured & online)
    # If DB is offline, raise clean error or handle according to requirement
    is_connected, _ = check_db_connection()
    if is_connected:
        insert_sql = """
            INSERT INTO uploads (
                upload_id, stored_filename, original_filename, relative_path,
                mime_type, file_size, image_width, image_height, status
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
        """
        try:
            with get_db_cursor(commit=True) as cursor:
                cursor.execute(
                    insert_sql,
                    (
                        metadata["upload_id"],
                        metadata["stored_filename"],
                        metadata["original_filename"],
                        metadata["relative_path"],
                        metadata["mime_type"],
                        metadata["file_size"],
                        metadata["image_width"],
                        metadata["image_height"],
                        metadata["status"],
                    ),
                )
        except Exception as err:
            logger.error("Failed to insert upload metadata into MySQL: %s", err)
            # Safe cleanup: Roll back file write so no orphaned file remains
            if target_path.exists():
                try:
                    target_path.unlink()
                except OSError:
                    pass
            raise DatabaseStorageError(
                "Database error: could not save upload metadata. Image storage was rolled back."
            )
    else:
        logger.info(
            "MySQL database offline; upload %s saved to disk in standby mode.",
            upload_id,
        )

    return metadata


def get_upload_metadata(
    upload_id: str, upload_folder: Optional[Path] = None
) -> Optional[Dict[str, Any]]:
    """
    Retrieves upload metadata by its unique upload_id.
    First attempts MySQL lookup from the 'uploads' table.
    If MySQL is unavailable, looks up the file on disk in the upload directory.
    """
    if not upload_id or not upload_id.isalnum():
        return None

    if upload_folder is None:
        try:
            from flask import current_app, has_request_context
            if has_request_context() and current_app and "UPLOAD_FOLDER" in current_app.config:
                upload_folder = Path(current_app.config["UPLOAD_FOLDER"])
            else:
                upload_folder = Path(Config.UPLOAD_FOLDER)
        except Exception:
            upload_folder = Path(Config.UPLOAD_FOLDER)
    else:
        upload_folder = Path(upload_folder)

    # Check MySQL if online
    is_connected, _ = check_db_connection()
    if is_connected:
        sql = """
            SELECT upload_id, stored_filename, original_filename, relative_path,
                   mime_type, file_size, image_width, image_height, status, created_at
            FROM uploads
            WHERE upload_id = %s
            LIMIT 1;
        """
        try:
            row = execute_query(sql, (upload_id,), fetchone=True)
            if row:
                return dict(row)
        except Exception as err:
            logger.warning("Failed to query upload from database: %s", err)

    # Standby disk-based lookup
    for ext in ("jpg", "png"):
        candidate_filename = f"{upload_id}.{ext}"
        candidate_path = upload_folder / candidate_filename
        if candidate_path.exists():
            stat = candidate_path.stat()
            width, height = 0, 0
            try:
                with Image.open(candidate_path) as img:
                    width, height = img.size
            except Exception:
                pass
            return {
                "upload_id": upload_id,
                "stored_filename": candidate_filename,
                "original_filename": candidate_filename,
                "relative_path": f"uploads/{candidate_filename}",
                "mime_type": "image/jpeg" if ext == "jpg" else "image/png",
                "file_size": stat.st_size,
                "image_width": width,
                "image_height": height,
                "status": "pending_analysis",
                "created_at": None,
            }

    return None
