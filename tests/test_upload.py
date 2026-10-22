"""
Unit and integration tests for secure plant leaf image upload (tests/test_upload.py).
"""

import io
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

from app import create_app
from config import TestingConfig
from services.upload_service import (
    DatabaseStorageError,
    UploadValidationError,
    validate_and_process_image,
)
from werkzeug.datastructures import FileStorage


def create_mock_image(format="JPEG", size=(200, 200), color=(34, 139, 34)) -> io.BytesIO:
    """Helper to generate in-memory valid image bytes using Pillow."""
    buf = io.BytesIO()
    mode = "RGB" if format == "JPEG" else "RGBA"
    img = Image.new(mode, size, color=color)
    img.save(buf, format=format)
    buf.seek(0)
    return buf


@pytest.fixture
def temp_upload_dir(tmp_path):
    """Provides a temporary upload directory isolated for each test."""
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    return upload_dir


@pytest.fixture
def client(temp_upload_dir):
    """Provides a Flask test client configured with temp upload folder."""
    app = create_app(TestingConfig)
    app.config["UPLOAD_FOLDER"] = temp_upload_dir
    with app.test_client() as test_client:
        yield test_client


def test_valid_jpg_upload(client, temp_upload_dir):
    """Verify that a valid JPG image is successfully uploaded and stored."""
    img_bytes = create_mock_image(format="JPEG")
    data = {
        "leaf_image": (img_bytes, "healthy_tomato_leaf.jpg", "image/jpeg"),
    }
    response = client.post("/upload", data=data, content_type="multipart/form-data")

    # Follows Post/Redirect/Get pattern
    assert response.status_code == 302
    assert "/results/" in response.headers["Location"] or "/upload/success/" in response.headers["Location"]

    # Verify a single image file was created in upload directory
    stored_files = list(temp_upload_dir.glob("*.jpg"))
    assert len(stored_files) == 1
    # Check that filename is safe 32-character hex + .jpg
    assert len(stored_files[0].stem) == 32


def test_valid_jpeg_upload(client, temp_upload_dir):
    """Verify that .jpeg extension is normalized and stored successfully."""
    img_bytes = create_mock_image(format="JPEG")
    data = {
        "leaf_image": (img_bytes, "potato_early_blight.jpeg", "image/jpeg"),
    }
    response = client.post("/upload", data=data, content_type="multipart/form-data")
    assert response.status_code == 302
    assert "/results/" in response.headers["Location"] or "/upload/success/" in response.headers["Location"]
    stored_files = list(temp_upload_dir.glob("*.jpg"))
    assert len(stored_files) == 1


def test_valid_png_upload(client, temp_upload_dir):
    """Verify that a valid PNG image is successfully uploaded."""
    img_bytes = create_mock_image(format="PNG")
    data = {
        "leaf_image": (img_bytes, "corn_rust_sample.png", "image/png"),
    }
    response = client.post("/upload", data=data, content_type="multipart/form-data")
    assert response.status_code == 302
    stored_files = list(temp_upload_dir.glob("*.png"))
    assert len(stored_files) == 1


def test_ajax_upload_returns_json(client):
    """Verify that AJAX submission returns JSON with redirect URL."""
    img_bytes = create_mock_image(format="JPEG")
    data = {"leaf_image": (img_bytes, "leaf.jpg", "image/jpeg")}
    headers = {"X-Requested-With": "XMLHttpRequest", "Accept": "application/json"}
    response = client.post("/upload", data=data, content_type="multipart/form-data", headers=headers)
    assert response.status_code == 200
    res_json = response.get_json()
    assert res_json["success"] is True
    assert "upload_id" in res_json
    assert "/results/" in res_json["redirect_url"] or "/upload/success/" in res_json["redirect_url"]


def test_missing_file_rejected(client):
    """Verify that submitting an empty form is rejected with 400 Bad Request."""
    response = client.post("/upload", data={}, content_type="multipart/form-data")
    assert response.status_code == 400


def test_empty_filename_rejected(client):
    """Verify that an empty filename is rejected with 400 Bad Request."""
    data = {"leaf_image": (io.BytesIO(b""), "", "image/jpeg")}
    response = client.post("/upload", data=data, content_type="multipart/form-data")
    assert response.status_code == 400


def test_unsupported_extension_rejected(client):
    """Verify that unsupported extensions (.txt, .exe, .gif, .pdf) are rejected."""
    for bad_name in ["script.py", "exploit.exe", "document.pdf", "image.gif"]:
        data = {"leaf_image": (io.BytesIO(b"Fake content"), bad_name, "text/plain")}
        response = client.post("/upload", data=data, content_type="multipart/form-data")
        assert response.status_code == 400


def test_spoofed_mime_type_rejected(client):
    """Verify that a non-image file with a fake .jpg extension is detected and rejected."""
    text_content = io.BytesIO(b"This is just plain text, not a real JPEG image.")
    data = {"leaf_image": (text_content, "fake_image.jpg", "image/jpeg")}
    response = client.post("/upload", data=data, content_type="multipart/form-data")
    assert response.status_code == 400


def test_corrupted_image_rejected(client):
    """Verify that a corrupted image stream fails Pillow integrity check."""
    corrupted_bytes = io.BytesIO(b"\xff\xd8\xff\xe0" + b"Corrupted bytes" * 20)
    data = {"leaf_image": (corrupted_bytes, "broken.jpg", "image/jpeg")}
    response = client.post("/upload", data=data, content_type="multipart/form-data")
    assert response.status_code == 400


def test_excessive_dimensions_rejected(temp_upload_dir):
    """Verify that images exceeding dimension limits (> 5000px) are rejected safely."""
    # Test service level validation with large dimensions
    fs = FileStorage(
        stream=create_mock_image(format="JPEG", size=(5001, 100)),
        filename="huge_dimensions.jpg",
        content_type="image/jpeg",
    )
    with pytest.raises(UploadValidationError, match="dimensions"):
        validate_and_process_image(fs, upload_folder=temp_upload_dir)


def test_database_failure_cleans_up_uploaded_file(temp_upload_dir):
    """Verify that if MySQL fails, newly written image is removed and unrelated files remain untouched."""
    # Place an unrelated pre-existing file in the directory
    unrelated_file = temp_upload_dir / "unrelated_existing.jpg"
    unrelated_file.write_bytes(b"preserved-content")

    fs = FileStorage(
        stream=create_mock_image(format="JPEG"),
        filename="test_rollback.jpg",
        content_type="image/jpeg",
    )

    # Mock database to simulate connectivity error on insert
    with patch("services.upload_service.check_db_connection", return_value=(True, "OK")):
        with patch("services.upload_service.get_db_cursor", side_effect=RuntimeError("Simulated MySQL DB crash")):
            with pytest.raises(DatabaseStorageError):
                validate_and_process_image(fs, upload_folder=temp_upload_dir)

    # Confirm the unrelated file was not touched
    assert unrelated_file.exists()
    assert unrelated_file.read_bytes() == b"preserved-content"

    # Confirm no orphaned image was left from the failed upload
    files_in_dir = list(temp_upload_dir.glob("*.jpg"))
    assert files_in_dir == [unrelated_file]


def test_upload_success_page_rendering(client, temp_upload_dir):
    """Verify that /upload/success/<upload_id> renders metadata and status properly."""
    img_bytes = create_mock_image(format="JPEG")
    fs = FileStorage(stream=img_bytes, filename="leaf_sample.jpg", content_type="image/jpeg")
    metadata = validate_and_process_image(fs, upload_folder=temp_upload_dir)

    response = client.get(f"/upload/success/{metadata['upload_id']}")
    assert response.status_code == 200
    html = response.data.decode("utf-8")
    assert "Image Uploaded Successfully" in html
    assert "Ready for AI analysis" in html
    assert metadata["upload_id"] in html
    assert "Leaf Image Upload Verified" in html or "disease detection" in html.lower()


def test_upload_success_page_404_for_unknown_id(client):
    """Verify that /upload/success/<unknown_id> returns a 404 page."""
    response = client.get("/upload/success/unknown999999999999999999999999")
    assert response.status_code == 404
    html = response.data.decode("utf-8")
    assert "Upload Not Found" in html
