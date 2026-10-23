"""
Unified Multi-Dataset Foliar Disease Training & Evaluation Pipeline (scripts/train_unified_model.py)
===================================================================================================
Combines laboratory-controlled images (PlantVillage, spMohanty) and natural field images
(PlantDoc, pratikkayal) to train a unified, domain-invariant MobileNetV2 foliar disease classifier.
Evaluates on both lab and field test sets and preserves models/best_model.keras.
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

# Windows stdout encoding
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
logger = logging.getLogger("unified_pipeline")

RANDOM_SEED = 42
CANDIDATES_DIR = BASE_DIR / "models" / "candidates"
REPORTS_DIR = BASE_DIR / "reports" / "candidates"


def compute_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_combined_manifests() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    """Loads and unifies records from PlantVillage (lab) and PlantDoc (field) manifests."""
    pv_manifest_path = BASE_DIR / "dataset" / "plantvillage_candidate" / "manifest.json"
    pd_manifest_path = BASE_DIR / "dataset" / "plantdoc" / "manifest.json"

    with open(pv_manifest_path, "r", encoding="utf-8") as f:
        pv_data = json.load(f)
    with open(pd_manifest_path, "r", encoding="utf-8") as f:
        pd_data = json.load(f)

    with open(BASE_DIR / "models" / "class_names.json", "r", encoding="utf-8") as f:
        class_names = json.load(f)

    # Label sources
    for r in pv_data["records"]:
        r["domain"] = "PlantVillage-Lab"
    for r in pd_data["records"]:
        r["domain"] = "PlantDoc-Field"

    train_recs = [r for r in pv_data["records"] if r["split"] == "train"] + [r for r in pd_data["records"] if r["split"] == "train"]
    val_recs = [r for r in pv_data["records"] if r["split"] == "val"] + [r for r in pd_data["records"] if r["split"] == "val"]
    pv_test_recs = [r for r in pv_data["records"] if r["split"] == "test"]
    pd_test_recs = [r for r in pd_data["records"] if r["split"] == "test"]

    logger.info("Unified Dataset Composition:")
    logger.info(" - Total Training Images:   %d (PV Lab: %d, PD Field: %d)", len(train_recs), len(pv_data["records"]) - 383 - 405, len(pd_data["records"]) - 337 - 236)
    logger.info(" - Total Validation Images: %d (PV Lab: 383, PD Field: 337)", len(val_recs))
    logger.info(" - PlantVillage Test Set:   %d images", len(pv_test_recs))
    logger.info(" - PlantDoc Test Set:       %d images", len(pd_test_recs))
    logger.info(" - Combined Test Set:       %d images", len(pv_test_recs) + len(pd_test_recs))

    return train_recs, val_recs, pv_test_recs, pd_test_recs, class_names


def train_unified_model(train_recs: List[Dict[str, Any]], val_recs: List[Dict[str, Any]], class_names: List[str]) -> Path:
    """Trains MobileNetV2 on combined multi-domain foliar dataset."""
    import tensorflow as tf
    from models.disease_classifier import build_disease_classifier, unfreeze_for_fine_tuning

    logger.info("Building and compiling Unified MobileNetV2 Model...")
    tf.keras.utils.set_random_seed(RANDOM_SEED)

    batch_size = 32
    image_size = (224, 224)
    num_classes = len(class_names)

    def _build_ds(recs, is_train=False):
        paths = [r["filepath"] for r in recs]
        labels = [int(r["class_index"]) for r in recs]
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        if is_train:
            ds = ds.shuffle(buffer_size=len(recs), seed=RANDOM_SEED)

        def _loader(path_t, label_t):
            def _py_read(p):
                return load_and_preprocess_image(p.decode("utf-8"), target_size=image_size)
            img = tf.numpy_function(_py_read, [path_t], tf.float32)
            img.set_shape([224, 224, 3])
            return img, label_t

        ds = ds.map(_loader, num_parallel_calls=tf.data.AUTOTUNE)
        if is_train:
            aug = get_augmentation_pipeline()
            ds = ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
        return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    train_ds = _build_ds(train_recs, is_train=True)
    val_ds = _build_ds(val_recs, is_train=False)

    CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = CANDIDATES_DIR / "unified_multidataset_model.keras"

    model = build_disease_classifier(
        num_classes=num_classes,
        input_shape=(224, 224, 3),
        weights="imagenet",
        dropout_rate=0.3,
        initial_learning_rate=1e-3,
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]

    t0 = time.time()
    logger.info("--- Stage 1: Training Classification Head on Combined Data (5 Epochs) ---")
    h1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=5,
        callbacks=callbacks,
        verbose=1,
    )

    logger.info("--- Stage 2: Fine-Tuning Top 30 Backbone Layers (3 Epochs) ---")
    model = unfreeze_for_fine_tuning(model, num_layers_to_unfreeze=30, fine_tune_learning_rate=1e-5)
    h2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=3,
        callbacks=callbacks,
        verbose=1,
    )
    duration = time.time() - t0

    model.save(str(model_path))
    logger.info("Saved Unified Candidate Model to: %s (Training time: %.1fs)", model_path, duration)

    # Save training history
    combined_history = {}
    for k in h1.history.keys():
        combined_history[k] = h1.history[k] + h2.history.get(k, [])

    hist_path = REPORTS_DIR / "unified_multidataset_history.csv"
    with open(hist_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch"] + list(combined_history.keys()))
        for ep in range(len(combined_history["loss"])):
            writer.writerow([ep + 1] + [combined_history[k][ep] for k in combined_history.keys()])

    return model_path


def evaluate_benchmark(
    cand_model_path: Path,
    pv_test_recs: List[Dict[str, Any]],
    pd_test_recs: List[Dict[str, Any]],
    class_names: List[str],
) -> Dict[str, Any]:
    """Evaluates candidate vs production model across Laboratory, Field, and Combined Test Sets."""
    import tensorflow as tf

    logger.info("Evaluating Unified Candidate Model vs. Production Model...")
    prod_model_path = BASE_DIR / "models" / "best_model.keras"

    cand_model = tf.keras.models.load_model(str(cand_model_path), compile=False)
    prod_model = tf.keras.models.load_model(str(prod_model_path), compile=False)

    def _eval_dataset(recs: List[Dict[str, Any]], name: str) -> Dict[str, Any]:
        imgs, labels = [], []
        for r in recs:
            try:
                img = load_and_preprocess_image(r["filepath"], target_size=(224, 224))
                imgs.append(img)
                labels.append(r["class_index"])
            except Exception as e:
                logger.warning("Error loading %s: %s", r["filepath"], e)

        x = np.array(imgs, dtype=np.float32)
        y = np.array(labels, dtype=np.int32)

        # Candidate
        c_preds = np.argmax(cand_model.predict(x, batch_size=32, verbose=0), axis=1)
        c_acc = float(np.mean(c_preds == y))
        c_rep = classification_report(y, c_preds, labels=list(range(len(class_names))), target_names=class_names, output_dict=True, zero_division=0)
        c_cm = confusion_matrix(y, c_preds, labels=list(range(len(class_names))))

        # Production
        p_preds = np.argmax(prod_model.predict(x, batch_size=32, verbose=0), axis=1)
        p_acc = float(np.mean(p_preds == y))
        p_rep = classification_report(y, p_preds, labels=list(range(len(class_names))), target_names=class_names, output_dict=True, zero_division=0)

        return {
            "name": name,
            "sample_count": len(y),
            "candidate": {
                "accuracy_percent": round(c_acc * 100, 2),
                "macro_f1": round(c_rep["macro avg"]["f1-score"], 4),
                "weighted_f1": round(c_rep["weighted avg"]["f1-score"], 4),
            },
            "production": {
                "accuracy_percent": round(p_acc * 100, 2),
                "macro_f1": round(p_rep["macro avg"]["f1-score"], 4),
                "weighted_f1": round(p_rep["weighted avg"]["f1-score"], 4),
            },
            "candidate_cm": c_cm,
        }

    lab_res = _eval_dataset(pv_test_recs, "PlantVillage (Laboratory Controlled)")
    field_res = _eval_dataset(pd_test_recs, "PlantDoc (Natural Field Conditions)")
    combined_res = _eval_dataset(pv_test_recs + pd_test_recs, "Combined Multi-Domain Test Set")

    # Save Confusion Matrix for Combined Test Set
    plt.figure(figsize=(18, 15))
    plt.imshow(combined_res["candidate_cm"], interpolation="nearest", cmap="GnBu")
    plt.title(f"Unified Multi-Dataset Model Confusion Matrix (Combined Test Acc: {combined_res['candidate']['accuracy_percent']}%)", fontsize=14, fontweight="bold")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.xticks(np.arange(len(class_names)), class_names, rotation=90, fontsize=8)
    plt.yticks(np.arange(len(class_names)), class_names, fontsize=8)
    plt.xlabel("Predicted Class", fontweight="bold")
    plt.ylabel("Ground Truth Class", fontweight="bold")
    plt.tight_layout()
    cm_path = REPORTS_DIR / "unified_multidataset_confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()

    eval_summary = {
        "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "candidate_model_path": str(cand_model_path.resolve()),
        "candidate_sha256": compute_sha256(cand_model_path),
        "benchmarks": [
            {
                "benchmark_name": lab_res["name"],
                "samples": lab_res["sample_count"],
                "candidate_accuracy": lab_res["candidate"]["accuracy_percent"],
                "candidate_macro_f1": lab_res["candidate"]["macro_f1"],
                "production_accuracy": lab_res["production"]["accuracy_percent"],
                "production_macro_f1": lab_res["production"]["macro_f1"],
            },
            {
                "benchmark_name": field_res["name"],
                "samples": field_res["sample_count"],
                "candidate_accuracy": field_res["candidate"]["accuracy_percent"],
                "candidate_macro_f1": field_res["candidate"]["macro_f1"],
                "production_accuracy": field_res["production"]["accuracy_percent"],
                "production_macro_f1": field_res["production"]["macro_f1"],
            },
            {
                "benchmark_name": combined_res["name"],
                "samples": combined_res["sample_count"],
                "candidate_accuracy": combined_res["candidate"]["accuracy_percent"],
                "candidate_macro_f1": combined_res["candidate"]["macro_f1"],
                "production_accuracy": combined_res["production"]["accuracy_percent"],
                "production_macro_f1": combined_res["production"]["macro_f1"],
            },
        ],
    }

    eval_json_path = REPORTS_DIR / "unified_multidataset_eval.json"
    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump(eval_summary, f, indent=2)

    logger.info("Saved Unified Evaluation JSON: %s", eval_json_path)
    logger.info("Saved Unified Confusion Matrix: %s", cm_path)
    return eval_summary


def main():
    print("=" * 75)
    print("Unified Multi-Dataset Pipeline: PlantVillage + PlantDoc + PlantDoc-OD")
    print("=" * 75)

    prod_path = BASE_DIR / "models" / "best_model.keras"
    initial_hash = compute_sha256(prod_path)
    print(f"Verified Production Model SHA-256 Before Run: {initial_hash}")

    train_recs, val_recs, pv_test_recs, pd_test_recs, class_names = load_combined_manifests()

    cand_model_path = train_unified_model(train_recs, val_recs, class_names)

    eval_summary = evaluate_benchmark(cand_model_path, pv_test_recs, pd_test_recs, class_names)

    final_hash = compute_sha256(prod_path)
    if initial_hash != final_hash:
        logger.critical("SAFETY VIOLATION: models/best_model.keras was modified!")
        sys.exit(1)
    else:
        print(f"\n[SAFETY VERIFIED] models/best_model.keras remained 100% UNTOUCHED.")

    print("\n" + "=" * 75)
    print("UNIFIED MODEL BENCHMARK RESULTS")
    print("=" * 75)
    for b in eval_summary["benchmarks"]:
        print(f"Dataset: {b['benchmark_name']} ({b['samples']} samples)")
        print(f"  * Unified Candidate:  Accuracy = {b['candidate_accuracy']}%, Macro F1 = {b['candidate_macro_f1']}")
        print(f"  * Production Model:   Accuracy = {b['production_accuracy']}%, Macro F1 = {b['production_macro_f1']}")
        print("-" * 75)


if __name__ == "__main__":
    main()
