# Trained Model Artifacts & Architecture Specifications

This directory holds trained model weights, checkpoints, class index dictionaries, and training metadata produced by the Phase 3 deep learning pipeline.

---

## 📁 Artifact Hierarchy

```text
model/
├── README.md                 # Artifact management guide
├── plant_disease_model.keras # Primary best-performing Keras model (saved by ModelCheckpoint)
├── class_indices.json        # Class-to-index mapping dictionary { "Tomato___Early_blight": 0, ... }
├── model_metadata.json       # Hyperparameters, MobileNetV2 architecture metadata, sha256 checksum
│
├── checkpoints/              # Epoch-level checkpoint backups during active training
│   └── best_checkpoint.keras
│
└── saved_model/              # Exported SavedModel format (for high-performance serving)
```

---

## 🧠 Model Architecture & Hyperparameters

* **Base Architecture**: MobileNetV2 (Pre-trained on ImageNet, `include_top=False`).
* **Input Resolution**: `224 × 224 × 3` (RGB).
* **Feature Aggregator**: `GlobalAveragePooling2D`.
* **Regularization**: `Dropout(0.3)`.
* **Classification Head**: `Dense(num_classes, activation="softmax")`.
* **Loss Function**: `sparse_categorical_crossentropy` (or `categorical_crossentropy`).
* **Optimizer**: Adam (`learning_rate=0.001` initial; `learning_rate=0.00001` fine-tuning).
* **Callbacks**: `ModelCheckpoint`, `EarlyStopping(patience=3, restore_best_weights=True)`, `ReduceLROnPlateau`, `CSVLogger`.

---

## 🔒 Version Control & Data Integrity

* Large binary weights (`*.h5`, `*.keras`, `checkpoints/`) are strictly excluded from Git via `.gitignore`.
* Lightweight configuration dictionaries (`class_indices.json`, `model_metadata.json`) remain tracked to ensure reproducible inference.
