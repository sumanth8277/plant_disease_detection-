"""
Model Evaluation and Error Diagnostics Tool (scripts/evaluate_model.py)
========================================================================
Evaluates trained MobileNetV2 model on held-out test split across multi-source datasets.
Features:
- Test loss and overall accuracy.
- Per-class precision, recall, F1-score, and support counts.
- Macro and weighted F1-scores.
- Per-dataset source breakdown (assessing laboratory vs. real-world domain shift).
- Confusion matrix plot saved to reports/confusion_matrix.png.
- Misclassified examples logged to reports/misclassified_examples.json.
- Insufficient test support audit (< 5 examples).
- Explicit documentation of domain-shift limitations and unsupported plant rejection.
"""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Configure Windows stdout encoding
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from services.image_preprocessing import load_and_preprocess_image

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("evaluate_model")


def render_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    output_path: Path,
) -> None:
    """Renders and saves a labeled confusion matrix heatmap using matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n = len(class_names)
        fig_size = max(8, min(24, n * 0.7))
        fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.8))

        im = ax.imshow(cm, interpolation="nearest", cmap="Greens")
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set(
            xticks=np.arange(n),
            yticks=np.arange(n),
            xticklabels=class_names,
            yticklabels=class_names,
            title="Multi-Dataset Foliar Disease Confusion Matrix",
            ylabel="True Class",
            xlabel="Predicted Class",
        )

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=8)
        plt.setp(ax.get_yticklabels(), fontsize=8)

        # Annotate text if matrix size is manageable
        if n <= 40:
            thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
            for i in range(n):
                for j in range(n):
                    val = cm[i, j]
                    if val > 0:
                        ax.text(
                            j, i, f"{val}",
                            ha="center", va="center",
                            color="white" if val > thresh else "black",
                            fontsize=7,
                        )

        plt.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=180)
        plt.close()
        logger.info("Saved confusion matrix plot: %s", output_path)
    except Exception as err:
        logger.warning("Could not render confusion matrix plot: %s", err)


def evaluate_test_set(
    manifest_path: Path,
    model_path: Path,
    output_dir: Path,
    batch_size: int = 16,
) -> Dict[str, Any]:
    """Loads model and runs test-set evaluation."""
    import tensorflow as tf
    from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

    output_dir.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        logger.warning("Manifest not found at %s.", manifest_path)
        return {
            "status": "manifest_not_found",
            "message": f"Manifest not found at {manifest_path}.",
        }

    if not model_path.exists():
        logger.warning("Model file not found at %s.", model_path)
        return {
            "status": "model_not_found",
            "message": f"Model file not found at {model_path}.",
        }

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    records = manifest_data.get("records", [])
    class_indices_map = manifest_data.get("class_indices", {})
    if isinstance(class_indices_map, dict):
        # Ensure integer keys
        idx_to_class = {int(k): v for k, v in class_indices_map.items()}
    else:
        idx_to_class = {i: c for i, c in enumerate(class_indices_map)}

    class_names = [idx_to_class[i] for i in sorted(idx_to_class.keys())]

    test_recs = [r for r in records if r.get("split") == "test"]
    if not test_recs:
        logger.warning("No test records in manifest.")
        return {
            "status": "test_set_empty",
            "message": "No test split records found in manifest.",
        }

    logger.info("Loading model from: %s", model_path)
    model = tf.keras.models.load_model(str(model_path))

    # Run inference
    y_true: List[int] = []
    y_pred: List[int] = []
    y_probs: List[float] = []
    misclassified: List[Dict[str, Any]] = []
    source_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0})

    logger.info("Running inference on %d test images...", len(test_recs))
    for rec in test_recs:
        fpath = rec["filepath"]
        true_idx = int(rec["class_index"])
        source_name = rec.get("source", "unknown")

        try:
            arr = load_and_preprocess_image(fpath, target_size=(224, 224))
            batch = np.expand_dims(arr, axis=0)
            preds = model.predict(batch, verbose=0)[0]
            pred_idx = int(np.argmax(preds))
            conf = float(preds[pred_idx])

            y_true.append(true_idx)
            y_pred.append(pred_idx)
            y_probs.append(conf)

            source_stats[source_name]["total"] += 1
            if pred_idx == true_idx:
                source_stats[source_name]["correct"] += 1
            else:
                if len(misclassified) < 100:
                    misclassified.append({
                        "file": rec.get("relative_path") or Path(fpath).name,
                        "source": source_name,
                        "true_class": idx_to_class.get(true_idx, str(true_idx)),
                        "predicted_class": idx_to_class.get(pred_idx, str(pred_idx)),
                        "confidence": round(conf, 4),
                    })
        except Exception as err:
            logger.warning("Failed processing test image %s: %s", fpath, err)

    if not y_true:
        return {"status": "inference_failed", "message": "Failed to process any test images."}

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    overall_accuracy = float(np.mean(y_true_arr == y_pred_arr))

    # Compute sklearn metrics
    report_dict = classification_report(
        y_true_arr,
        y_pred_arr,
        target_names=[idx_to_class[i] for i in sorted(list(set(y_true) | set(y_pred)))],
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=list(range(len(class_names))))
    cm_path = output_dir / "confusion_matrix.png"
    render_confusion_matrix(cm, class_names, cm_path)

    # Per source accuracy
    source_breakdown = {}
    for s_name, s_info in source_stats.items():
        tot = s_info["total"]
        cor = s_info["correct"]
        acc = round(cor / tot, 4) if tot > 0 else 0.0
        source_breakdown[s_name] = {
            "total_test_samples": tot,
            "correct_predictions": cor,
            "accuracy": acc,
            "accuracy_percent": round(acc * 100, 2),
        }

    # Insufficient test support check (< 5 samples)
    class_support_counts = defaultdict(int)
    for y in y_true:
        class_support_counts[idx_to_class.get(y, str(y))] += 1

    insufficient_classes = [
        {"class": c, "test_samples": cnt}
        for c, cnt in class_support_counts.items()
        if cnt < 5
    ]

    # Save reports
    eval_results = {
        "timestamp": datetime.now().isoformat(),
        "model_path": str(model_path.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "total_test_images": len(y_true),
        "overall_accuracy": round(overall_accuracy, 4),
        "overall_accuracy_percent": round(overall_accuracy * 100, 2),
        "macro_f1": round(float(report_dict.get("macro avg", {}).get("f1-score", 0.0)), 4),
        "weighted_f1": round(float(report_dict.get("weighted avg", {}).get("f1-score", 0.0)), 4),
        "per_dataset_source_performance": source_breakdown,
        "classes_with_insufficient_test_support": insufficient_classes,
        "domain_shift_notes": (
            "Models evaluated on real-world in-situ field imagery (e.g. PlantDoc) typically exhibit "
            "lower top-1 accuracy compared to clean laboratory imagery (e.g. PlantVillage) due to "
            "complex backgrounds, varying lighting, shadows, and concurrent multiple diseases."
        ),
        "unsupported_species_notes": (
            "Specimens from plant species not explicitly included in the trained class list (such as Neem, "
            "ornamental plants, or wild weeds) must be handled through out-of-distribution detection "
            "(entropy, confidence margin) and returned as UNSUPPORTED_SPECIMEN to prevent pesticide misapplication."
        ),
        "classification_report": report_dict,
    }

    report_json_path = output_dir / "classification_report.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2, ensure_ascii=False)
    logger.info("Saved classification report: %s", report_json_path)

    misclassified_path = output_dir / "misclassified_examples.json"
    with open(misclassified_path, "w", encoding="utf-8") as f:
        json.dump(misclassified, f, indent=2, ensure_ascii=False)
    logger.info("Saved misclassified examples log: %s", misclassified_path)

    return eval_results


def main():
    parser = argparse.ArgumentParser(description="Multi-Dataset Model Evaluation & Metrics")
    parser.add_argument(
        "--manifest",
        type=str,
        default=str(BASE_DIR / "reports" / "dataset_manifest.json"),
        help="Path to manifest JSON",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(BASE_DIR / "models" / "best_model.keras"),
        help="Path to trained model .keras file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(BASE_DIR / "reports"),
        help="Output directory for reports and figures",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")

    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    model_path = Path(args.model_path)
    output_dir = Path(args.output_dir)

    # Fallback to model/saved_model if models/best_model.keras not found yet
    if not model_path.exists():
        fallback_model = BASE_DIR / "model" / "saved_model" / "plant_disease_model.keras"
        if fallback_model.exists():
            model_path = fallback_model

    print("=" * 75)
    print("Multi-Dataset Foliar Disease Model Evaluation Tool")
    print("=" * 75)
    print(f"Manifest:   {manifest_path.resolve()}")
    print(f"Model Path: {model_path.resolve()}")
    print(f"Output Dir: {output_dir.resolve()}")
    print("-" * 75)

    res = evaluate_test_set(
        manifest_path=manifest_path,
        model_path=model_path,
        output_dir=output_dir,
        batch_size=args.batch_size,
    )

    print("\n[EVALUATION COMPLETE]")
    print(f"Status: {res.get('status', 'success')}")
    if "overall_accuracy" in res:
        print(f"Test Accuracy: {res['overall_accuracy_percent']}%")
        print(f"Macro F1:      {res['macro_f1']}")
        print(f"Weighted F1:   {res['weighted_f1']}")
        print("\nSource Performance:")
        for s_name, s_stat in res.get("per_dataset_source_performance", {}).items():
            print(f"  * {s_name}: {s_stat['accuracy_percent']}% ({s_stat['correct_predictions']}/{s_stat['total_test_samples']})")
    else:
        print(f"Message: {res.get('message')}")
    print("=" * 75)


if __name__ == "__main__":
    main()
