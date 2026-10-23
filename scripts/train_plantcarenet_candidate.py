"""
PlantCareNet Candidate Model Training & Benchmarking Pipeline (scripts/train_plantcarenet_candidate.py)
======================================================================================================
Trains the custom PlantCareNet CNN architecture from Muhaimin008/Plant-disease-classification-and-recommendations
on the unified foliar dataset (PlantVillage lab + PlantDoc field images), evaluates against independent test sets,
and saves the candidate model separately to models/candidates/plantcarenet_candidate.keras.
"""

import csv
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from services.image_preprocessing import get_augmentation_pipeline, load_and_preprocess_image

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("plantcarenet_pipeline")

RANDOM_SEED = 42
CANDIDATES_DIR = BASE_DIR / "models" / "candidates"
REPORTS_DIR = BASE_DIR / "reports" / "candidates"
CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def compute_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_combined_manifests() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    pv_manifest_path = BASE_DIR / "dataset" / "plantvillage_candidate" / "manifest.json"
    pd_manifest_path = BASE_DIR / "dataset" / "plantdoc" / "manifest.json"

    with open(pv_manifest_path, "r", encoding="utf-8") as f:
        pv_data = json.load(f)
    with open(pd_manifest_path, "r", encoding="utf-8") as f:
        pd_data = json.load(f)

    with open(BASE_DIR / "models" / "class_names.json", "r", encoding="utf-8") as f:
        class_names = json.load(f)

    train_recs = [r for r in pv_data["records"] if r["split"] == "train"] + [r for r in pd_data["records"] if r["split"] == "train"]
    val_recs = [r for r in pv_data["records"] if r["split"] == "val"] + [r for r in pd_data["records"] if r["split"] == "val"]
    pv_test_recs = [r for r in pv_data["records"] if r["split"] == "test"]
    pd_test_recs = [r for r in pd_data["records"] if r["split"] == "test"]

    logger.info("Loaded Combined Dataset: %d train, %d val, %d test (PV: %d, PD: %d)",
                len(train_recs), len(val_recs), len(pv_test_recs) + len(pd_test_recs), len(pv_test_recs), len(pd_test_recs))
    return train_recs, val_recs, pv_test_recs, pd_test_recs, class_names


def build_plantcarenet_model(num_classes: int = 38, input_shape: Tuple[int, int, int] = (128, 128, 3)):
    """
    Builds the custom PlantCareNet CNN architecture from Muhaimin008/plantcarenet.ipynb.
    Features: 3 Conv blocks with BatchNorm, Dropout, MaxPooling, and Dense classification head.
    """
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, Activation, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense

    model = Sequential([
        # Block 1
        Conv2D(32, (3, 3), padding="same", input_shape=input_shape),
        Activation("relu"),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Block 2
        Conv2D(64, (3, 3), padding="same"),
        Activation("relu"),
        BatchNormalization(),
        Conv2D(64, (3, 3), padding="same"),
        Activation("relu"),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Block 3
        Conv2D(128, (3, 3), padding="same"),
        Activation("relu"),
        BatchNormalization(),
        Dropout(0.3),
        Conv2D(128, (3, 3), padding="same"),
        Activation("relu"),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Head
        Flatten(),
        Dense(512),
        Activation("relu"),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes),
        Activation("softmax")
    ])

    opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=opt,
        metrics=["accuracy"]
    )
    return model


def train_plantcarenet(train_recs: List[Dict[str, Any]], val_recs: List[Dict[str, Any]], class_names: List[str]) -> Path:
    import tensorflow as tf

    logger.info("Initializing PlantCareNet Model...")
    tf.keras.utils.set_random_seed(RANDOM_SEED)

    batch_size = 32
    target_size = (128, 128)
    num_classes = len(class_names)

    def _build_ds(recs, is_train=False):
        paths = [r["filepath"] for r in recs]
        labels = [int(r["class_index"]) for r in recs]
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        if is_train:
            ds = ds.shuffle(buffer_size=len(recs), seed=RANDOM_SEED)

        def _loader(path_t, label_t):
            def _py_read(p):
                return load_and_preprocess_image(p.decode("utf-8"), target_size=target_size)
            img = tf.numpy_function(_py_read, [path_t], tf.float32)
            img.set_shape([128, 128, 3])
            return img, label_t

        ds = ds.map(_loader, num_parallel_calls=tf.data.AUTOTUNE)
        if is_train:
            aug = tf.keras.Sequential([
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomRotation(0.1),
                tf.keras.layers.RandomZoom(0.1)
            ])
            ds = ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
        return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    train_ds = _build_ds(train_recs, is_train=True)
    val_ds = _build_ds(val_recs, is_train=False)

    model = build_plantcarenet_model(num_classes=num_classes, input_shape=(128, 128, 3))
    candidate_path = CANDIDATES_DIR / "plantcarenet_candidate.keras"

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(filepath=str(candidate_path), monitor="val_accuracy", save_best_only=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5)
    ]

    logger.info("Training PlantCareNet for up to 6 epochs...")
    start_t = time.time()
    hist = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=6,
        callbacks=callbacks
    )
    train_time = time.time() - start_t
    logger.info("PlantCareNet training completed in %.1f seconds.", train_time)

    # Save training history
    history_csv = REPORTS_DIR / "plantcarenet_candidate_history.csv"
    with open(history_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss", "accuracy", "val_loss", "val_accuracy"])
        for ep in range(len(hist.history["loss"])):
            writer.writerow([
                ep + 1,
                f"{hist.history['loss'][ep]:.4f}",
                f"{hist.history['accuracy'][ep]:.4f}",
                f"{hist.history['val_loss'][ep]:.4f}",
                f"{hist.history['val_accuracy'][ep]:.4f}"
            ])

    return candidate_path


def evaluate_plantcarenet(candidate_path: Path, pv_test_recs: List[Dict[str, Any]], pd_test_recs: List[Dict[str, Any]], class_names: List[str]):
    import tensorflow as tf

    logger.info("Loading PlantCareNet candidate for evaluation: %s", candidate_path)
    model = tf.keras.models.load_model(str(candidate_path))
    target_size = (128, 128)

    def _eval_dataset(recs, desc="Test Set"):
        y_true, y_pred = [], []
        for r in recs:
            try:
                img = load_and_preprocess_image(r["filepath"], target_size=target_size)
                pred = model.predict(np.expand_dims(img, axis=0), verbose=0)
                y_true.append(int(r["class_index"]))
                y_pred.append(int(np.argmax(pred[0])))
            except Exception as e:
                logger.warning("Error processing %s: %s", r["filepath"], e)

        acc = float(np.mean(np.array(y_true) == np.array(y_pred)))
        rep = classification_report(y_true, y_pred, target_names=[class_names[i] for i in sorted(list(set(y_true)))], output_dict=True, zero_division=0)
        return acc, rep, y_true, y_pred

    pv_acc, pv_rep, _, _ = _eval_dataset(pv_test_recs, "PlantVillage Lab Test")
    pd_acc, pd_rep, _, _ = _eval_dataset(pd_test_recs, "PlantDoc Field Test")
    comb_recs = pv_test_recs + pd_test_recs
    comb_acc, comb_rep, comb_true, comb_pred = _eval_dataset(comb_recs, "Combined Test")

    logger.info("=== PlantCareNet Candidate Benchmark Results ===")
    logger.info(" - PlantVillage Lab Test Accuracy: %.2f%%", pv_acc * 100)
    logger.info(" - PlantDoc Field Test Accuracy:    %.2f%%", pd_acc * 100)
    logger.info(" - Overall Combined Test Accuracy:  %.2f%% (Macro F1: %.4f)", comb_acc * 100, comb_rep["macro avg"]["f1-score"])

    # Plot confusion matrix
    cm = confusion_matrix(comb_true, comb_pred)
    fig, ax = plt.subplots(figsize=(14, 12))
    cax = ax.matshow(cm, cmap="Blues")
    fig.colorbar(cax)
    ax.set_title("PlantCareNet Candidate Model - Confusion Matrix", fontsize=14, pad=20)
    ax.set_xlabel("Predicted Class Index", fontsize=11)
    ax.set_ylabel("True Class Index", fontsize=11)
    plt.tight_layout()
    cm_path = REPORTS_DIR / "plantcarenet_candidate_confusion_matrix.png"
    plt.savefig(cm_path, dpi=200)
    plt.close()

    eval_summary = {
        "candidate_model": "plantcarenet_candidate.keras",
        "architecture": "Custom PlantCareNet CNN (Muhaimin008)",
        "input_shape": [128, 128, 3],
        "lab_test_accuracy_percent": round(pv_acc * 100, 2),
        "field_test_accuracy_percent": round(pd_acc * 100, 2),
        "combined_test_accuracy_percent": round(comb_acc * 100, 2),
        "combined_macro_f1": round(comb_rep["macro avg"]["f1-score"], 4),
        "plantvillage_samples_evaluated": len(pv_test_recs),
        "plantdoc_samples_evaluated": len(pd_test_recs),
        "total_test_samples": len(comb_recs)
    }

    eval_json_path = REPORTS_DIR / "plantcarenet_candidate_eval.json"
    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump(eval_summary, f, indent=2)

    metadata = {
        "model_version": "PlantCareNet-Candidate-20261023",
        "architecture": "Custom PlantCareNet CNN (Conv2D + BatchNorm + Dropout)",
        "source_repo": "Muhaimin008/Plant-disease-classification-and-recommendations",
        "input_shape": [128, 128, 3],
        "num_classes": len(class_names),
        "sha256": compute_sha256(candidate_path),
        "file_size_bytes": candidate_path.stat().st_size,
        "metrics": eval_summary
    }
    with open(CANDIDATES_DIR / "plantcarenet_candidate_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Saved evaluation JSON to %s and metadata to %s", eval_json_path, CANDIDATES_DIR / "plantcarenet_candidate_metadata.json")


if __name__ == "__main__":
    train_recs, val_recs, pv_test_recs, pd_test_recs, class_names = load_combined_manifests()
    cand_path = train_plantcarenet(train_recs, val_recs, class_names)
    evaluate_plantcarenet(cand_path, pv_test_recs, pd_test_recs, class_names)
    logger.info("PlantCareNet Training & Evaluation Successfully Completed!")
