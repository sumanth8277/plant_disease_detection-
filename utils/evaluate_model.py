"""
Model Evaluation and Metrics Visualization (utils/evaluate_model.py)
Evaluates the best-saved MobileNetV2 model strictly on the held-out test set.
Computes test loss, accuracy, precision, recall, F1-score, and support counts.
Generates a confusion matrix plot, training curves, and reports under reports/training/.

Usage:
    python utils/evaluate_model.py
    python utils/evaluate_model.py --model-path model/plant_disease_model.keras
"""

import argparse
import csv
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Configure UTF-8 on Windows consoles if supported
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import Config
from services.image_preprocessing import load_and_preprocess_image

logger = logging.getLogger(__name__)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    output_path: Path,
) -> None:
    """Renders and saves a labeled confusion matrix heatmap using matplotlib."""
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(8, len(class_names) * 0.9), max(6, len(class_names) * 0.7)))
    im = ax.imshow(cm, interpolation="nearest", cmap="Greens")
    ax.figure.colorbar(im, ax=ax)

    # Show all ticks and label with class names
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title="Test Set Confusion Matrix — Plant Disease Classification",
        ylabel="True Foliar Class",
        xlabel="Predicted Foliar Class",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations
    thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            ax.text(
                j,
                i,
                f"{val}",
                ha="center",
                va="center",
                color="white" if val > thresh else "black",
                fontsize=9,
            )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=150)
    plt.close(fig)


def plot_training_history(history_csv: Path, output_path: Path) -> bool:
    """Renders training vs. validation accuracy and loss curves from CSV history."""
    if not history_csv.exists():
        return False

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = []
    loss = []
    val_loss = []
    acc = []
    val_acc = []

    with open(history_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row.get("epoch", len(epochs))))
            loss.append(float(row.get("loss", 0.0)))
            val_loss.append(float(row.get("val_loss", 0.0)))
            # Handle either accuracy or acc naming
            acc.append(float(row.get("accuracy", row.get("acc", 0.0))))
            val_acc.append(float(row.get("val_accuracy", row.get("val_acc", 0.0))))

    if not epochs:
        return False

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy Plot
    ax1.plot(epochs, acc, label="Training Accuracy", color="#15803d", lw=2)
    ax1.plot(epochs, val_acc, label="Validation Accuracy", color="#0284c7", lw=2, linestyle="--")
    ax1.set_title("Model Accuracy Across Epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="lower right")

    # Loss Plot
    ax2.plot(epochs, loss, label="Training Loss", color="#b91c1c", lw=2)
    ax2.plot(epochs, val_loss, label="Validation Loss", color="#d97706", lw=2, linestyle="--")
    ax2.set_title("Model Cross-Entropy Loss Across Epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=150)
    plt.close(fig)
    return True


def evaluate_trained_model(
    model_path: Optional[Path] = None,
    dataset_root: Optional[Path] = None,
    manifest_path: Optional[Path] = None,
    class_indices_path: Optional[Path] = None,
    reports_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Evaluates the saved model artifact strictly on the test set partition.
    """
    import tensorflow as tf
    from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

    model_path = model_path or (Config.BASE_DIR / "model" / "plant_disease_model.keras")
    dataset_root = dataset_root or (Config.BASE_DIR / "dataset" / "raw")
    manifest_path = manifest_path or (Config.BASE_DIR / "reports" / "training" / "dataset_manifest.json")
    class_indices_path = class_indices_path or (Config.BASE_DIR / "model" / "class_indices.json")
    reports_dir = reports_dir or (Config.BASE_DIR / "reports" / "training")

    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at '{model_path}'. "
            "Please run 'python utils/train_model.py' to produce a model artifact before evaluating."
        )

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Dataset manifest not found at '{manifest_path}'. "
            "Please run 'python utils/prepare_dataset.py' to generate test splits."
        )

    if not class_indices_path.exists():
        raise FileNotFoundError(f"Class indices not found at '{class_indices_path}'.")

    # Load resources
    print(f"[INFO] Loading model from: {model_path}")
    model = tf.keras.models.load_model(str(model_path))

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    with open(class_indices_path, "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)

    # Sort class names matching indices: index 0 -> class_names[0]
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    test_records = manifest_data.get("manifest", {}).get("test", [])
    if not test_records:
        raise ValueError("No test records found in manifest. Cannot evaluate without a held-out test set.")

    print(f"[INFO] Evaluating on {len(test_records)} held-out test images across {len(class_names)} classes...")

    # Load and preprocess all test images
    X_test_list = []
    y_true = []
    valid_test_records = []

    for rec in test_records:
        img_path = dataset_root / rec["relative_path"]
        if not img_path.exists():
            logger.warning("Test image missing on disk: %s", img_path)
            continue
        try:
            arr = load_and_preprocess_image(img_path)
            X_test_list.append(arr)
            y_true.append(int(rec["class_index"]))
            valid_test_records.append(rec)
        except Exception as err:
            logger.warning("Failed to load test image %s: %s", img_path, err)

    if not X_test_list:
        raise ValueError("None of the test images specified in manifest could be loaded from disk.")

    X_test = np.stack(X_test_list, axis=0)
    y_true = np.array(y_true, dtype=np.int32)

    # Predict
    y_pred_probs = model.predict(X_test, batch_size=16, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Evaluate test loss and accuracy via model.evaluate
    eval_results = model.evaluate(X_test, y_true, batch_size=16, verbose=0)
    test_loss = float(eval_results[0])
    test_accuracy = float(eval_results[1])

    # Scikit-learn metrics
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)

    # Per-class metrics
    class_report_dict = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    class_report_text = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0,
    )

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    # Save artifacts
    reports_dir.mkdir(parents=True, exist_ok=True)

    cm_plot_path = reports_dir / "confusion_matrix.png"
    plot_confusion_matrix(cm, class_names, cm_plot_path)

    history_plot_path = reports_dir / "training_history.png"
    history_csv_path = reports_dir / "training_history.csv"
    plot_training_history(history_csv_path, history_plot_path)

    text_report_path = reports_dir / "classification_report.txt"
    with open(text_report_path, "w", encoding="utf-8") as f:
        f.write(class_report_text)

    eval_summary: Dict[str, Any] = {
        "evaluation_timestamp": datetime.now().isoformat(),
        "model_path": str(model_path.resolve()),
        "test_images_evaluated": len(y_true),
        "total_classes": len(class_names),
        "test_loss": round(test_loss, 4),
        "test_accuracy": round(test_accuracy, 4),
        "macro_metrics": {
            "precision": round(float(p_macro), 4),
            "recall": round(float(r_macro), 4),
            "f1_score": round(float(f1_macro), 4),
        },
        "weighted_metrics": {
            "precision": round(float(p_weighted), 4),
            "recall": round(float(r_weighted), 4),
            "f1_score": round(float(f1_weighted), 4),
        },
        "per_class_metrics": {
            cname: {
                "precision": round(class_report_dict[cname]["precision"], 4),
                "recall": round(class_report_dict[cname]["recall"], 4),
                "f1_score": round(class_report_dict[cname]["f1-score"], 4),
                "support": class_report_dict[cname]["support"],
            }
            for cname in class_names if cname in class_report_dict
        },
        "confusion_matrix": cm.tolist(),
        "plots": {
            "confusion_matrix": str(cm_plot_path.relative_to(Config.BASE_DIR)),
            "training_history": str(history_plot_path.relative_to(Config.BASE_DIR)) if history_plot_path.exists() else None,
        },
    }

    eval_json_path = reports_dir / "evaluation_report.json"
    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump(eval_summary, f, indent=2)

    return eval_summary


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained MobileNetV2 disease classifier on held-out test images."
    )
    parser.add_argument("--model-path", type=str, default=str(Config.BASE_DIR / "model" / "plant_disease_model.keras"))
    parser.add_argument("--dataset-root", type=str, default=str(Config.BASE_DIR / "dataset" / "raw"))
    parser.add_argument("--manifest-path", type=str, default=str(Config.BASE_DIR / "reports" / "training" / "dataset_manifest.json"))
    parser.add_argument("--class-indices-path", type=str, default=str(Config.BASE_DIR / "model" / "class_indices.json"))
    parser.add_argument("--reports-dir", type=str, default=str(Config.BASE_DIR / "reports" / "training"))

    args = parser.parse_args()

    print("=" * 70)
    print("Independent Model Evaluation — Held-Out Test Set")
    print("=" * 70)
    print(f"Model Path:     {args.model_path}")
    print(f"Manifest Path:  {args.manifest_path}")
    print(f"Reports Output: {args.reports_dir}")
    print("-" * 70)

    try:
        report = evaluate_trained_model(
            model_path=Path(args.model_path),
            dataset_root=Path(args.dataset_root),
            manifest_path=Path(args.manifest_path),
            class_indices_path=Path(args.class_indices_path),
            reports_dir=Path(args.reports_dir),
        )

        print("\n" + "=" * 70)
        print("[SUCCESS] Evaluation Complete!")
        print(f"Test Images Evaluated: {report['test_images_evaluated']}")
        print(f"Test Loss:             {report['test_loss']}")
        print(f"Test Accuracy:         {report['test_accuracy'] * 100:.2f}%")
        print(f"Macro F1-Score:        {report['macro_metrics']['f1_score']:.4f}")
        print(f"Weighted F1-Score:     {report['weighted_metrics']['f1_score']:.4f}")
        print("-" * 70)
        print(f"Report JSON:           {report.get('plots', {}).get('confusion_matrix')}")
        print("=" * 70)

    except Exception as err:
        print(f"\n[ERROR] Evaluation failed: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
