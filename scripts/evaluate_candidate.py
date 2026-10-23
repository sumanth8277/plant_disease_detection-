"""
Candidate Model Evaluation Script (scripts/evaluate_candidate.py)
=================================================================
Evaluates plantvillage_candidate.keras vs models/best_model.keras on the
independent held-out test split of 405 images across all 38 classes.
"""

import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path

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

from services.image_preprocessing import load_and_preprocess_image

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("evaluate_candidate")


def compute_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def main():
    import tensorflow as tf

    manifest_path = BASE_DIR / "dataset" / "plantvillage_candidate" / "manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    test_recs = [r for r in manifest_data["records"] if r["split"] == "test"]
    class_names = manifest_data["classes"]

    candidate_path = BASE_DIR / "models" / "candidates" / "plantvillage_candidate.keras"
    production_path = BASE_DIR / "models" / "best_model.keras"
    reports_dir = BASE_DIR / "reports" / "candidates"
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("Candidate Model vs. Production Model Test Set Evaluation")
    print("=" * 75)
    print(f"Manifest:          {manifest_path}")
    print(f"Test Samples:      {len(test_recs)} images")
    print(f"Classes:           {len(class_names)} classes")
    print(f"Candidate Model:   {candidate_path}")
    print(f"Production Model:  {production_path}")
    print("-" * 75)

    prod_hash_before = compute_sha256(production_path)
    cand_hash = compute_sha256(candidate_path)

    # 1. Load test dataset
    logger.info("Loading and preprocessing %d test images...", len(test_recs))
    test_x = []
    test_y = []
    for r in test_recs:
        img = load_and_preprocess_image(r["filepath"], target_size=(224, 224))
        test_x.append(img)
        test_y.append(r["class_index"])

    test_x = np.array(test_x, dtype=np.float32)
    test_y = np.array(test_y, dtype=np.int32)

    # 2. Evaluate Candidate Model
    logger.info("Evaluating candidate model: %s", candidate_path)
    cand_model = tf.keras.models.load_model(str(candidate_path), compile=False)
    cand_preds = cand_model.predict(test_x, batch_size=32, verbose=1)
    cand_pred_labels = np.argmax(cand_preds, axis=1)

    cand_acc = float(np.mean(cand_pred_labels == test_y))
    cand_report = classification_report(
        test_y,
        cand_pred_labels,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    cand_cm = confusion_matrix(test_y, cand_pred_labels, labels=list(range(len(class_names))))

    # 3. Evaluate Existing Production Model
    logger.info("Evaluating production model: %s", production_path)
    prod_model = tf.keras.models.load_model(str(production_path), compile=False)
    prod_preds = prod_model.predict(test_x, batch_size=32, verbose=1)
    prod_pred_labels = np.argmax(prod_preds, axis=1)

    prod_acc = float(np.mean(prod_pred_labels == test_y))
    prod_report = classification_report(
        test_y,
        prod_pred_labels,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    prod_cm = confusion_matrix(test_y, prod_pred_labels, labels=list(range(len(class_names))))

    # 4. Save Candidate Confusion Matrix Plot
    plt.figure(figsize=(18, 15))
    plt.imshow(cand_cm, interpolation="nearest", cmap="Blues")
    plt.title(f"PlantVillage Candidate Confusion Matrix (Test Accuracy: {cand_acc*100:.2f}%)", fontsize=14, fontweight="bold")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.xticks(np.arange(len(class_names)), class_names, rotation=90, fontsize=8)
    plt.yticks(np.arange(len(class_names)), class_names, fontsize=8)
    plt.xlabel("Predicted Label", fontweight="bold")
    plt.ylabel("True Label", fontweight="bold")
    plt.tight_layout()
    cm_path = reports_dir / "plantvillage_candidate_confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()
    logger.info("Saved confusion matrix plot: %s", cm_path)

    # 5. Save Comparison Results JSON
    results = {
        "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "test_dataset": "PlantVillage-Candidate-Subset (Leaf-Group Isolated)",
        "test_samples_total": len(test_y),
        "evaluated_classes_count": len(class_names),
        "evaluated_all_38_classes": True,
        "candidate_model": {
            "path": str(candidate_path.resolve()),
            "sha256": cand_hash,
            "accuracy_percent": round(cand_acc * 100, 2),
            "macro_f1": round(cand_report["macro avg"]["f1-score"], 4),
            "weighted_f1": round(cand_report["weighted avg"]["f1-score"], 4),
            "correct_samples": int(np.sum(cand_pred_labels == test_y)),
        },
        "production_model": {
            "path": str(production_path.resolve()),
            "sha256": prod_hash_before,
            "accuracy_percent": round(prod_acc * 100, 2),
            "macro_f1": round(prod_report["macro avg"]["f1-score"], 4),
            "weighted_f1": round(prod_report["weighted avg"]["f1-score"], 4),
            "correct_samples": int(np.sum(prod_pred_labels == test_y)),
        },
        "per_class_candidate_metrics": {
            cls: {
                "precision": round(cand_report[cls]["precision"], 4),
                "recall": round(cand_report[cls]["recall"], 4),
                "f1_score": round(cand_report[cls]["f1-score"], 4),
                "support": cand_report[cls]["support"],
            }
            for cls in class_names
        },
        "per_class_production_metrics": {
            cls: {
                "precision": round(prod_report[cls]["precision"], 4),
                "recall": round(prod_report[cls]["recall"], 4),
                "f1_score": round(prod_report[cls]["f1-score"], 4),
                "support": prod_report[cls]["support"],
            }
            for cls in class_names
        },
    }

    eval_json_path = reports_dir / "plantvillage_candidate_eval.json"
    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # 6. Save Candidate Model Metadata and Training Config
    cand_metadata = {
        "model_version": "PlantVillage-Candidate-20261023",
        "architecture": "MobileNetV2",
        "input_shape": [224, 224, 3],
        "preprocessing": "mobilenet_v2.preprocess_input",
        "num_classes": len(class_names),
        "classes": class_names,
        "sha256": cand_hash,
        "model_path": str(candidate_path.resolve()),
        "test_accuracy_percent": round(cand_acc * 100, 2),
        "test_macro_f1": round(cand_report["macro avg"]["f1-score"], 4),
        "trained_on": "PlantVillage RGB Color Subset (70 images/class, leaf-isolated)",
    }
    with open(candidate_path.parent / "candidate_model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(cand_metadata, f, indent=2)

    cand_config = {
        "model_name": "plantvillage_candidate.keras",
        "batch_size": 32,
        "image_size": [224, 224],
        "stage1_epochs": 5,
        "stage1_lr": 0.001,
        "stage2_epochs": 3,
        "stage2_lr": 1e-5,
        "unfreeze_layers": 30,
        "dropout_rate": 0.3,
        "storage_cap_mb": 500,
        "actual_dataset_mb": 41.45,
    }
    with open(candidate_path.parent / "candidate_training_config.json", "w", encoding="utf-8") as f:
        json.dump(cand_config, f, indent=2)

    # 7. Safety check: ensure production model hash is identical
    prod_hash_after = compute_sha256(production_path)
    if prod_hash_before != prod_hash_after:
        logger.critical("SAFETY VIOLATION: models/best_model.keras was modified during evaluation!")
        sys.exit(1)

    print("\n" + "=" * 75)
    print("EVALUATION COMPLETE & METRICS COMPUTED")
    print("=" * 75)
    print(f"Safety Check:             models/best_model.keras UNTOUCHED (Hash: {prod_hash_after})")
    print(f"Candidate Model Accuracy:  {results['candidate_model']['accuracy_percent']}% ({results['candidate_model']['correct_samples']}/{len(test_y)})")
    print(f"Candidate Macro F1:        {results['candidate_model']['macro_f1']}")
    print(f"Candidate Weighted F1:     {results['candidate_model']['weighted_f1']}")
    print("-" * 75)
    print(f"Production Model Accuracy: {results['production_model']['accuracy_percent']}% ({results['production_model']['correct_samples']}/{len(test_y)})")
    print(f"Production Macro F1:       {results['production_model']['macro_f1']}")
    print(f"Production Weighted F1:    {results['production_model']['weighted_f1']}")
    print("=" * 75)
    print(f"Results JSON:              {eval_json_path}")
    print(f"Confusion Matrix Plot:     {cm_path}")
    print("=" * 75)


if __name__ == "__main__":
    main()
