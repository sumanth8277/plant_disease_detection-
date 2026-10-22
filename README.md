# Agrotech: Plant Disease Detection and Smart Pesticide & Fertilizer Recommendation System

An intelligent agronomic advisory and diagnostic platform combining deep learning computer vision for foliar disease detection with targeted, sustainable pesticide recommendations, growth-stage-specific fertilizer schedules, enterprise security, and personalized farm diagnostic management.

---

## 📌 Project Phases & Implementation Status

* **Phase 1: Architecture & MySQL Relational Database** `[Completed]`
  * Enterprise modular structure, configuration via `.env`, normalized MySQL schema (`users`, `predictions`, `disease_recommendations`, `fertilizer_recommendations`), connection pool abstraction, safe error sanitization, and repeatable data seeder.
* **Phase 2: Web Interface, Leaf Upload, & Dataset Tooling** `[Completed]`
  * Responsive agriculture-themed web interface (HTML5/CSS3/JavaScript).
  * Secure plant leaf image upload pipeline with client-side & server-side validation (Pillow integrity, decompression bomb guards, path traversal prevention, cryptographic UUID filenames).
  * MySQL upload metadata logging (`uploads` table) with automatic file cleanup on database rollback.
  * Controlled image serving route (`/uploads/<filename>`) with security headers (`nosniff`).
  * Post/Redirect/Get upload workflow with detailed success preview.
  * Dataset acquisition instructions, image integrity audit, duplicate detection, and stratified split tools.
* **Phase 3: Convolutional Neural Network (MobileNetV2) Training & Evaluation Pipeline** `[Completed]`
  * Transfer Learning architecture (`models/disease_classifier.py`) using MobileNetV2 with frozen backbone, custom classification head, and selective fine-tuning stage.
  * Standardized image preprocessing service (`services/image_preprocessing.py`) handling EXIF orientation, RGBA/grayscale-to-RGB conversion, 224×224 bilinear resizing, MobileNetV2 `[-1, 1]` scaling, and Keras data augmentation (flips, rotations, zoom).
  * Reproducible dataset preparation pipeline (`utils/prepare_dataset.py`) generating stratified 70/15/15 train/val/test splits, SHA-256 duplicate image leakage prevention, and machine-readable manifests (`reports/training/dataset_manifest.json`, `model/class_indices.json`).
  * Robust training engine (`utils/train_model.py`) with `tf.data` pipeline, caching, prefetching, multi-metric monitoring, callbacks (`ModelCheckpoint`, `EarlyStopping`, `ReduceLROnPlateau`, `CSVLogger`), and SHA-256 artifact checksum verification.
  * Independent model evaluator (`utils/evaluate_model.py`) generating test set metrics (`evaluation_report.json`), classification reports, confusion matrix heatmaps, and training curve plots.
  * Single-image inference smoke test CLI (`utils/predict_image.py`) with top-k ranking and diagnostic disclaimers.
* **Phase 4: ML Inference & Dynamic Recommendation Integration** `[Completed]`
  * Thread-safe, cached Model Service (`services/model_service.py`) with lazy loading, artifact validation, probability vector verification, and readiness checks.
  * Deterministic disease label normalization and mapping (`utils/disease_mapping.py` and `model/disease_class_mapping.json`) translating model outputs to standardized crop names, disease codes, and healthy vs. diseased flags.
  * Parameterized MySQL recommendation lookup (`services/recommendation_service.py`) for symptoms, Integrated Pest Management (IPM), verified chemical guidance, and crop nutrition.
  * End-to-end prediction pipeline and audit persistence (`services/prediction_service.py`) logging to MySQL `predictions` table with safe relative paths and numeric confidence constraints.
  * Diagnostic results web interface (`templates/results.html`) with leaf preview, confidence meter, top-3 alternate diagnoses, agronomic treatment blocks, growth-stage fertilizer schedule, and print stylesheet.
  * Dedicated model health diagnostic endpoint (`GET /health/model`).
* **Phase 5: User Authentication, Security Hardening, Prediction History & College Viva Deliverables** `[Completed]`
  * **Branding:** Modern **Agrotech** theme across all pages, dynamic navigation header, and responsive cards.
  * **Authentication:** User registration (`/register`), login (`/login`), and POST-only logout (`/logout`) backed by `services/auth_service.py`.
  * **Password Security:** Cryptographic password hashing and salting using Werkzeug (`scrypt`/`pbkdf2`).
  * **Session Hardening & CSRF Protection:** Cryptographic CSRF protection via `Flask-WTF` on all forms and AJAX requests; hardened session cookies (`HTTPOnly`, `SameSite=Lax`).
  * **Access Control & Ownership Isolation:** Strict Insecure Direct Object Reference (IDOR) prevention—viewing another user's scan returns **HTTP 403 Forbidden**.
  * **Prediction History Dashboard:** Paginated personal scan audit (`/history`) displaying past diagnoses, thumbnail previews, confidence levels, and timestamps.
  * **Database Index Optimization:** Applied composite index `idx_predictions_user_created (user_id, created_at DESC)` via `database/migrations/001_phase5_auth_indexes.sql`.
  * **Dual Workflow Support:** Frictionless guest scans (`user_id = NULL`) alongside personalized authenticated farmer history.
  * **Academic College Deliverables:**
    * [`reports/PROJECT_REPORT.md`](reports/PROJECT_REPORT.md) — Comprehensive academic mini-project report.
    * [`reports/VIVA_QUESTIONS.md`](reports/VIVA_QUESTIONS.md) — 30 detailed viva voce questions and exam-ready answers.
    * [`reports/PRESENTATION_OUTLINE.md`](reports/PRESENTATION_OUTLINE.md) — 12-slide presentation structure with speaking notes.
    * [`reports/DEMO_SCRIPT.md`](reports/DEMO_SCRIPT.md) — 5–7 minute live examiner walkthrough script.
    * [`reports/FINAL_TEST_CHECKLIST.md`](reports/FINAL_TEST_CHECKLIST.md) — 23-scenario E2E manual verification matrix.

* **Phase 6 / Architecture Upgrade: Multi-Dataset Training Pipeline & Hierarchical Plant Identification** `[Completed]`
  * **5-Dataset Pipeline Tools:** Scripts for inspection (`scripts/inspect_datasets.py`), standardization & anti-leakage splitting (`scripts/prepare_dataset.py`), model training (`scripts/train_model.py`), and test evaluation (`scripts/evaluate_model.py`) supporting PlantVillage, New Plant Diseases, PlantDoc, Plant Pathology 2020 FGVC7, and NeemLeaf.
  * **Hierarchical Classification Architecture:** Decomposes vision processing into Stage 1 (Plant Species Identification) and Stage 2 (Foliar Disease Diagnosis) via `services/plant_identifier_service.py`.
  * **Native Neem Handling & Recommendation Shielding:** Accurately recognizes Neem (*Azadirachta indica*) leaves while declaring neem disease detection unsupported; displays a dedicated notification banner and strictly suppresses false pesticide and fertilizer recommendations.
  * **Calibrated OOD Detection:** Multi-metric uncertainty gates combining green foliage coverage ($> 15\%$), crop-family probability aggregation ($> 40\%$), class margin ($> 12\%$), and normalized Shannon entropy ($< 0.65$).
  * **Automated Testing Suite:** Expanded to **129 tests** with 100% pass rate (`128 passed, 1 skipped`).

---

## 🛠️ Technology Stack

| Layer | Component | Specification |
|---|---|---|
| **Branding & UI** | Agrotech Portal | HTML5, CSS3, Vanilla JS, Responsive Grid & Flexbox |
| **Language** | Python | 3.11+ (tested on Python 3.11.9) |
| **Deep Learning Framework** | TensorFlow & Keras | TensorFlow 2.17.1, Keras 3.15.1 |
| **Model Architecture** | MobileNetV2 | Transfer Learning (224×224×3 RGB input) |
| **Evaluation & Metrics** | scikit-learn & matplotlib | scikit-learn 1.9.1, matplotlib 3.11.2 |
| **Array & Image Processing**| NumPy & Pillow | NumPy 1.26.4, Pillow 13.0.0 |
| **Web Framework & Security** | Flask & Flask-WTF | Flask 3.0.3, Flask-WTF 1.2.1, email-validator 2.2.0 |
| **Relational Database** | MySQL Server | 8.0+ (InnoDB, `utf8mb4`) |
| **Database Connector** | mysql-connector-python & PyMySQL | Official connector & PyMySQL drivers |
| **Automated Testing** | pytest | pytest 8.2.2+ (102 test cases) |
| **Target Dataset** | PlantVillage | Foliar leaf disease collection (38 classes) |

---

## 📁 Directory Structure

```text
plant desiease setection mini project/
├── .env                          # Local environment variables & credentials (Git-ignored)
├── .env.example                  # Template configuration with safe placeholders
├── .gitignore                    # Version control exclusion rules
├── app.py                        # Flask application entry point & factory (CSRF, blueprints)
├── config.py                     # Configuration classes (Development, Testing, Production)
├── requirements.txt              # Pinned Python dependencies for Phases 1-5
├── README.md                     # Comprehensive documentation and setup guide
│
├── database/
│   ├── __init__.py               # Exports database utilities
│   ├── db.py                     # Connection pooling, context managers & health check
│   ├── schema.sql                # Complete MySQL DDL (users, predictions, recommendations, uploads)
│   ├── seed.py                   # Idempotent seeder for demonstration guidance data
│   └── migrations/
│       └── 001_phase5_auth_indexes.sql # Composite index on predictions(user_id, created_at)
│
├── dataset/
│   ├── README.md                 # Dataset layout, classes, download instructions
│   ├── raw/                      # Raw class-organized subfolders (<class_name>/*.jpg)
│   └── processed/                # Manifest-governed prepared dataset artifacts
│
├── model/
│   ├── README.md                 # Artifact guide, class mappings & versioning notes
│   ├── checkpoints/              # Best checkpoint files saved during training
│   ├── saved_model/              # Production-ready exported models (.keras format)
│   ├── class_indices.json        # Class-name to integer index mappings
│   ├── disease_class_mapping.json# Canonical PlantVillage to MySQL code mapping
│   └── model_metadata.json       # Training metadata, input shape, SHA-256 checksum
│
├── models/
│   ├── __init__.py               # Package exports
│   └── disease_classifier.py     # MobileNetV2 model architecture & fine-tuning builder
│
├── reports/
│   ├── PROJECT_REPORT.md         # Comprehensive academic college project report
│   ├── VIVA_QUESTIONS.md         # 30 viva voce questions with technical answers
│   ├── PRESENTATION_OUTLINE.md   # 12-slide presentation outline and speaker notes
│   ├── DEMO_SCRIPT.md            # 5-7 minute live demonstration and examiner script
│   ├── FINAL_TEST_CHECKLIST.md   # 23-scenario E2E manual testing checklist
│   ├── dataset_inspection_report.json # Dataset audit report
│   ├── dataset_split_summary.json     # Split summary
│   └── training/
│       ├── dataset_manifest.json      # Stratified 70/15/15 image manifest
│       ├── training_log.csv           # Epoch-by-epoch loss and accuracy metrics
│       ├── training_history.png       # Loss and accuracy curves visualization
│       ├── evaluation_report.json     # Test set loss, accuracy, macro/weighted metrics
│       ├── classification_report.txt  # Precision, recall, f1-score per class
│       └── confusion_matrix.png       # Multi-class confusion matrix heatmap
│
├── routes/
│   ├── __init__.py               # Blueprint registration package (main_bp, auth_bp)
│   ├── main_routes.py            # Endpoints: /, /upload, /results/<id>, /health
│   └── auth_routes.py            # Endpoints: /register, /login, /logout, /history
│
├── services/
│   ├── __init__.py               # Services package
│   ├── auth_service.py           # User registration, login, scrypt hashing, user history
│   ├── image_preprocessing.py    # Robust RGB normalization, resizing & data augmentation
│   ├── model_service.py          # Cached model loading, readiness & inference
│   ├── prediction_service.py     # End-to-end prediction, user assignment & IDOR checks
│   ├── recommendation_service.py # Parameterized disease & fertilizer lookup
│   └── upload_service.py         # Pillow image validation, safe storage & DB logging
│
├── static/
│   ├── css/
│   │   └── style.css             # Agrotech tech styling, auth cards, history table & meters
│   ├── js/
│   │   └── script.js             # Drag-and-drop, preview, CSRF header injection
│   └── uploads/
│       └── .gitkeep              # Secure filesystem storage directory for leaf images
│
├── templates/
│   ├── base.html                 # Base layout with Agrotech branding, dynamic auth navbar, CSRF meta
│   ├── index.html                # Agrotech homepage with hero, how-it-works, upload zone
│   ├── upload.html               # Dedicated diagnosis upload workflow
│   ├── results.html              # Diagnostic report with confidence meter, IPM & fertilizer tabs
│   ├── register.html             # User registration form with validation feedback
│   ├── login.html                # User login form with redirect support
│   ├── history.html              # Personal prediction history dashboard with pagination
│   └── error.html                # Friendly, accessible error pages (400, 403, 404, 500, 503)
│
├── tests/
│   ├── __init__.py
│   ├── test_auth_service.py      # Registration, scrypt hashing, and credential validation tests
│   ├── test_auth_routes.py       # Auth routes, redirects, CSRF, and POST-only logout tests
│   ├── test_authorization.py     # IDOR defense and HTTP 403 authorization boundary tests
│   ├── test_app.py               # Route, health, and error handling tests
│   ├── test_config.py            # Environment validation tests
│   ├── test_database_schema.py   # Schema file structure and live MySQL tests
│   ├── test_dataset.py           # Dataset inspection & split tests
│   ├── test_dataset_preparation.py # Stratified split & duplicate prevention tests
│   ├── test_image_preprocessing.py # Image scaling, conversion, augmentation tests
│   ├── test_model_artifacts.py   # MobileNetV2 build, fine-tuning, inference tests
│   ├── test_model_service.py     # Model loading, error handling & mock tests
│   ├── test_phase4_routes.py     # Upload, model health & results routes tests
│   ├── test_predictions.py       # Prediction persistence, ownership & confidence tests
│   ├── test_recommendations.py   # MySQL disease & fertilizer lookup tests
│   ├── test_training_utils.py    # Training CLI validation tests
│   └── test_upload.py            # Upload verification, spoofing, Pillow integrity tests
│
└── utils/
    ├── __init__.py               # Package exports
    ├── auth.py                   # @login_required, is_safe_url, and password validation helpers
    ├── disease_mapping.py        # Label normalization & mapping engine
    ├── inspect_dataset.py        # Dataset audit CLI
    ├── prepare_dataset.py        # Stratified manifest & class index CLI
    ├── train_model.py            # Two-stage MobileNetV2 training engine
    ├── evaluate_model.py         # Independent test set evaluator
    └── predict_image.py          # Single-image smoke test CLI
```

---

## 🚀 Setup & Execution Guide

### 1. Virtual Environment Activation

In **Windows PowerShell**:

```powershell
# Activate virtual environment
.\venv\Scripts\Activate.ps1
```
*(On Linux/macOS: `source venv/bin/activate`)*

---

### 2. Dependency Installation

Install or verify required packages:

```powershell
pip install -r requirements.txt
```

---

### 3. MySQL Database Setup & Migration

If MySQL Server is installed:

1. **Connect as Administrator**:
   ```bash
   mysql -u root -p
   ```
2. **Execute Administrative Setup**:
   ```sql
   CREATE DATABASE IF NOT EXISTS plant_disease_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
   CREATE USER IF NOT EXISTS 'plant_app'@'localhost' IDENTIFIED BY 'YourStrongPassword123!';
   GRANT SELECT, INSERT, UPDATE, DELETE ON plant_disease_db.* TO 'plant_app'@'localhost';
   FLUSH PRIVILEGES;
   EXIT;
   ```
3. **Apply Schema DDL & Phase 5 Migration**:
   ```powershell
   Get-Content database/schema.sql | mysql -u root -p
   Get-Content database/migrations/001_phase5_auth_indexes.sql | mysql -u root -p
   ```
4. **Seed Demonstration Records**:
   ```powershell
   python database/seed.py
   ```

---

### 4. Running the Web Application

Start the Flask application:

```powershell
python app.py
```

Access the application in your browser:
* **Agrotech Homepage**: [http://127.0.0.1:5000/](http://127.0.0.1:5000/)
* **Leaf Diagnosis Screen**: [http://127.0.0.1:5000/upload](http://127.0.0.1:5000/upload)
* **User Registration**: [http://127.0.0.1:5000/register](http://127.0.0.1:5000/register)
* **User Login**: [http://127.0.0.1:5000/login](http://127.0.0.1:5000/login)
* **Prediction History Dashboard**: [http://127.0.0.1:5000/history](http://127.0.0.1:5000/history)
* **Application Health Probe**: [http://127.0.0.1:5000/health](http://127.0.0.1:5000/health)
* **Database Health Diagnostic**: [http://127.0.0.1:5000/health/db](http://127.0.0.1:5000/health/db)
* **Model Readiness Diagnostic**: [http://127.0.0.1:5000/health/model](http://127.0.0.1:5000/health/model)

---

## 🧪 Automated Testing

Run the comprehensive 102-test suite:

```powershell
pytest -v
```

Expected result:
```text
tests/test_app.py ......................... [ 11%]
tests/test_auth_routes.py ................ [ 20%]
tests/test_auth_service.py ............... [ 28%]
tests/test_authorization.py .............. [ 32%]
tests/test_config.py ...................... [ 36%]
tests/test_database_schema.py ............. [ 40%]
tests/test_dataset.py ..................... [ 44%]
tests/test_dataset_preparation.py ......... [ 48%]
tests/test_image_preprocessing.py ......... [ 56%]
tests/test_model_artifacts.py ............. [ 63%]
tests/test_model_service.py ............... [ 70%]
tests/test_phase4_routes.py ............... [ 76%]
tests/test_predictions.py ................. [ 83%]
tests/test_recommendations.py ............. [ 89%]
tests/test_training_utils.py .............. [ 94%]
tests/test_upload.py ...................... [100%]

====================== 101 passed, 1 skipped in 27.27s ======================
```
*(The single skipped test is `test_live_mysql_connectivity`, which skips gracefully when MySQL Server is not currently running on localhost).*

---

## 📚 College Project Documentation & Viva Guides

The following documents have been prepared specifically for university examinations, external viva reviews, and project presentations:

* **[Comprehensive Project Report](reports/PROJECT_REPORT.md)**: Academic project report covering abstract, literature survey, system design, transfer learning methodology, MySQL architecture, security controls, experimental evaluation, and future scope.
* **[Viva Voce Question Bank](reports/VIVA_QUESTIONS.md)**: 30 in-depth technical questions with concise, code-accurate answers spanning Deep Learning, Flask, MySQL, Web Security (CSRF, IDOR), and Agronomy.
* **[Presentation Slide Outline](reports/PRESENTATION_OUTLINE.md)**: 12-slide presentation structure with speaking notes and visual suggestions.
* **[Live Demonstration Script](reports/DEMO_SCRIPT.md)**: 5–7 minute timed demonstration script with examiner walkthroughs and fallback plans.
* **[Final Testing Checklist](reports/FINAL_TEST_CHECKLIST.md)**: 23-scenario verification matrix spanning authentication, scanning, history, and authorization.

---

## ⚠️ Responsible Use & Agronomic Limitations

* **Diagnostic Support Tool**: Predictions generated by this system are based on statistical deep learning visual patterns. They do not constitute certified biological assays or legal plant pathology guarantees.
* **Environmental Factors**: Lighting, leaf angle, camera focus, and background soil can affect model confidence.
* **Chemical Safety**: Never apply toxic fungicides, restricted-use pesticides, or synthetic chemicals without consulting an official regional agricultural extension officer or certified agronomist. Always read and adhere to pesticide manufacturer product labels.
