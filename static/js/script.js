/**
 * Plant Disease Detection System - Phase 2 Client Script
 * Handles drag-and-drop, client-side validation, object URL previewing,
 * loading states, accessible feedback, and diagnostic health polling.
 */

document.addEventListener("DOMContentLoaded", () => {
    // -------------------------------------------------------------------------
    // 1. Diagnostic Health Check Polling
    // -------------------------------------------------------------------------
    const dbStatusText = document.getElementById("db-status-text");
    const liveStatusPill = document.getElementById("live-status-pill");

    if (dbStatusText) {
        fetch("/health/db")
            .then(async (response) => {
                const data = await response.json();
                if (response.ok && data.status === "healthy") {
                    dbStatusText.textContent = "Connected & Healthy";
                    dbStatusText.style.color = "var(--primary)";
                    if (liveStatusPill) {
                        liveStatusPill.textContent = "Online";
                        liveStatusPill.className = "tag tag-active";
                    }
                } else {
                    const message = data.message || "Database unreachable";
                    dbStatusText.textContent = `Unavailable (${message})`;
                    dbStatusText.style.color = "var(--danger-text)";
                    if (liveStatusPill) {
                        liveStatusPill.textContent = "Degraded (No DB)";
                        liveStatusPill.className = "tag tag-danger";
                    }
                }
            })
            .catch(() => {
                dbStatusText.textContent = "Diagnostic endpoint unreachable";
                dbStatusText.style.color = "var(--danger-text)";
                if (liveStatusPill) {
                    liveStatusPill.textContent = "Offline";
                    liveStatusPill.className = "tag tag-danger";
                }
            });
    }

    // -------------------------------------------------------------------------
    // 2. Leaf Image Upload & Preview Handler
    // -------------------------------------------------------------------------
    const uploadForm = document.getElementById("upload-form");
    const dropzone = document.getElementById("dropzone");
    const fileInput = document.getElementById("leaf-image-input");
    const dropzonePrompt = document.getElementById("dropzone-prompt");
    const previewContainer = document.getElementById("preview-container");
    const imagePreview = document.getElementById("image-preview");
    const previewFilename = document.getElementById("preview-filename");
    const previewFilesize = document.getElementById("preview-filesize");
    const removeImageBtn = document.getElementById("remove-image-btn");
    const submitBtn = document.getElementById("submit-btn");
    const clientErrorBox = document.getElementById("client-error-box");

    if (!uploadForm || !fileInput) {
        return; // Page does not have an upload form
    }

    const MAX_FILE_SIZE = 5 * 1024 * 1024; // 5 Megabytes
    const ALLOWED_EXTENSIONS = ["jpg", "jpeg", "png"];
    const ALLOWED_MIME_TYPES = ["image/jpeg", "image/png"];

    let currentObjectUrl = null;

    /**
     * Shows a client-side error message.
     */
    function showError(message) {
        if (!clientErrorBox) return;
        clientErrorBox.textContent = message;
        clientErrorBox.style.display = "block";
    }

    /**
     * Clears client-side error messages.
     */
    function clearError() {
        if (!clientErrorBox) return;
        clientErrorBox.textContent = "";
        clientErrorBox.style.display = "none";
    }

    /**
     * Revokes active object URL to free memory.
     */
    function cleanupObjectUrl() {
        if (currentObjectUrl) {
            URL.revokeObjectURL(currentObjectUrl);
            currentObjectUrl = null;
        }
    }

    /**
     * Resets the dropzone to the initial empty state.
     */
    function resetUploadState() {
        cleanupObjectUrl();
        fileInput.value = "";
        if (imagePreview) imagePreview.src = "";
        if (previewContainer) previewContainer.style.display = "none";
        if (dropzonePrompt) dropzonePrompt.style.display = "block";
        if (submitBtn) submitBtn.disabled = true;
        clearError();
    }

    /**
     * Formats bytes into a human-readable string.
     */
    function formatBytes(bytes) {
        if (bytes === 0) return "0 Bytes";
        const k = 1024;
        const sizes = ["Bytes", "KB", "MB"];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + " " + sizes[i];
    }

    /**
     * Validates and displays the selected image file.
     */
    function handleSelectedFile(file) {
        clearError();

        if (!file) {
            resetUploadState();
            return;
        }

        // Validate File Extension
        const filename = file.name || "";
        const ext = filename.split(".").pop().toLowerCase();
        if (!ALLOWED_EXTENSIONS.includes(ext)) {
            showError(`Unsupported file format '.${ext}'. Please choose a JPG, JPEG, or PNG image.`);
            resetUploadState();
            return;
        }

        // Validate MIME type (if provided by browser)
        if (file.type && !ALLOWED_MIME_TYPES.includes(file.type)) {
            showError(`Invalid file type (${file.type}). Only JPEG and PNG images are allowed.`);
            resetUploadState();
            return;
        }

        // Validate File Size (5 MB limit)
        if (file.size > MAX_FILE_SIZE) {
            showError(`The selected file is too large (${formatBytes(file.size)}). Maximum upload size is 5 MB.`);
            resetUploadState();
            return;
        }

        // Generate safe Object URL preview
        cleanupObjectUrl();
        currentObjectUrl = URL.createObjectURL(file);

        if (imagePreview) {
            imagePreview.src = currentObjectUrl;
            imagePreview.onload = () => {
                // Free URL when preview DOM has finished loading
                // (kept active until replacement or form reset)
            };
        }

        if (previewFilename) previewFilename.textContent = filename;
        if (previewFilesize) previewFilesize.textContent = formatBytes(file.size);

        if (dropzonePrompt) dropzonePrompt.style.display = "none";
        if (previewContainer) previewContainer.style.display = "block";
        if (submitBtn) submitBtn.disabled = false;
    }

    // -------------------------------------------------------------------------
    // Event Listeners
    // -------------------------------------------------------------------------

    // Native file input change
    fileInput.addEventListener("change", (e) => {
        const file = e.target.files && e.target.files[0];
        handleSelectedFile(file);
    });

    // Remove / Replace button
    if (removeImageBtn) {
        removeImageBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            resetUploadState();
        });
    }

    // Drag and drop event handlers
    if (dropzone) {
        ["dragenter", "dragover"].forEach((eventName) => {
            dropzone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropzone.classList.add("dragover");
            });
        });

        ["dragleave", "drop"].forEach((eventName) => {
            dropzone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropzone.classList.remove("dragover");
            });
        });

        dropzone.addEventListener("drop", (e) => {
            const dt = e.dataTransfer;
            const files = dt.files;
            if (files && files.length > 0) {
                fileInput.files = files; // Bind dropped file to form input
                handleSelectedFile(files[0]);
            }
        });
    }

    // Form submission handler (prevents double submits and handles AJAX/standard submit)
    uploadForm.addEventListener("submit", async (e) => {
        if (!fileInput.files || fileInput.files.length === 0) {
            e.preventDefault();
            showError("Please select a leaf image file first.");
            return;
        }

        // Prevent duplicate clicks & show loading state
        submitBtn.disabled = true;
        const btnText = submitBtn.querySelector(".btn-text");
        const btnSpinner = submitBtn.querySelector(".btn-spinner");
        if (btnText) btnText.style.display = "none";
        if (btnSpinner) btnSpinner.style.display = "inline-flex";

        // If standard form submit is preferred, allow event to propagate.
        // Let's use fetch for smooth Post/Redirect/Get handling with immediate error display:
        e.preventDefault();
        clearError();

        const formData = new FormData(uploadForm);
        const csrfMeta = document.querySelector('meta[name="csrf-token"]');
        const headers = {
            "X-Requested-With": "XMLHttpRequest",
            "Accept": "application/json",
        };
        if (csrfMeta && csrfMeta.content) {
            headers["X-CSRFToken"] = csrfMeta.content;
        }

        try {
            const response = await fetch(uploadForm.action, {
                method: "POST",
                body: formData,
                headers: headers,
            });

            const result = await response.json();

            if (response.ok && result.success && result.redirect_url) {
                // Navigate via PRG pattern to the success page
                window.location.href = result.redirect_url;
            } else {
                const errMessage = result.error || "An error occurred during upload. Please check your image.";
                showError(errMessage);
                submitBtn.disabled = false;
                if (btnText) btnText.style.display = "inline";
                if (btnSpinner) btnSpinner.style.display = "none";
            }
        } catch (fetchErr) {
            // If AJAX failed or returned HTML, fall back to native submission
            uploadForm.submit();
        }
    });

    // Cleanup object URL on page unload
    window.addEventListener("beforeunload", () => {
        cleanupObjectUrl();
    });
});
