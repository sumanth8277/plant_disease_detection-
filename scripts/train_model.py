"""
MobileNetV2 Plant Disease Model Training Pipeline (scripts/train_model.py)
==========================================================================
Trains MobileNetV2 transfer learning model on standardized foliar disease dataset splits.
Features:
- 224x224x3 RGB image inputs with MobileNetV2 preprocessing.
- Stage 1: Frozen MobileNetV2 feature extractor + custom classification head.
- Stage 2: Fine-tuning of upper layers with reduced learning rate (optional).
- Early stopping, model checkpointing, LR reduction on plateau, CSV logging.
- Memory protection with configurable batch size and CPU/GPU detection.
- Generates:
  - models/best_model.keras
  - models/model_metadata.json
  - models/training_config.json
  - reports/training_history.csv
  - reports/training/loss_accuracy_curve.png
"""

import argparse
import csv
import hashlib
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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

from config import Config
from models.disease_classifier import build_disease_classifier, unfreeze_for_fine_tuning
from services.image_preprocessing import get_augmentation_pipeline, load_and_preprocess_image

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_model")


def compute_file_sha256(filepath: Path) -> str:
    """Computes SHA-256 digest of saved model artifact for provenance auditing."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def plot_training_curves(history: Dict[str, List[float]], output_path: Path) -> None:
    """Renders and saves loss and accuracy curves across training epochs."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs = range(1, len(history.get("loss", [])) + 1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Accuracy
        if "accuracy" in history:
            ax1.plot(epochs, history["accuracy"], label="Train Accuracy", color="#2e7d32", lw=2)
        if "val_accuracy" in history:
            ax1.plot(epochs, history["val_accuracy"], label="Validation Accuracy", color="#1565c0", lw=2)
        ax1.set_title("Model Accuracy Across Epochs", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Accuracy")
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.legend(loc="lower right")

        # Loss
        if "loss" in history:
            ax2.plot(epochs, history["loss"], label="Train Loss", color="#c62828", lw=2)
        if "val_loss" in history:
            ax2.plot(epochs, history["val_loss"], label="Validation Loss", color="#e65100", lw=2)
        ax2.set_title("Cross-Entropy Loss Across Epochs", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend(loc="upper right")

        plt.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        plt.close()
        logger.info("Saved training curves to: %s", output_path)
    except Exception as err:
        logger.warning("Could not generate training plot: %s", err)


def build_tf_dataset(
    records: List[Dict[str, Any]],
    batch_size: int = 16,
    is_training: bool = False,
    image_size: Tuple[int, int] = (224, 224),
    seed: int = 42,
) -> Any:
    """Creates a tf.data.Dataset from manifest records with on-the-fly augmentation for training."""
    import tensorflow as tf

    paths = [r["filepath"] for r in records]
    labels = [int(r["class_index"]) for r in records]

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if is_training:
        ds = ds.shuffle(buffer_size=min(len(records), 1000), seed=seed, reshuffle_each_iteration=True)

    def _loader(path_tensor, label_tensor):
        def _py_read(p):
            p_str = p.decode("utf-8")
            return load_and_preprocess_image(p_str, target_size=image_size)

        img_tensor = tf.numpy_function(_py_read, [path_tensor], tf.float32)
        img_tensor.set_shape([image_size[0], image_size[1], 3])
        return img_tensor, label_tensor

    ds = ds.map(_loader, num_parallel_calls=tf.data.AUTOTUNE)

    if is_training:
        aug_layer = get_augmentation_pipeline()
        ds = ds.map(lambda x, y: (aug_layer(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def train_pipeline(
    manifest_path: Path,
    output_model_path: Path,
    config_output_path: Path,
    history_csv_path: Path,
    plot_output_path: Path,
    metadata_output_path: Path,
    epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 0.001,
    fine_tune: bool = True,
    fine_tune_epochs: int = 5,
    fine_tune_lr: float = 1e-5,
    random_seed: int = 42,
    resume_from: Optional[Union[str, Path]] = None,
    fine_tune_only: bool = False,
) -> Dict[str, Any]:
    """Executes the full MobileNetV2 training workflow with checkpoint resumption support."""
    import tensorflow as tf

    # Set seeds
    tf.keras.utils.set_random_seed(random_seed)

    # Detect GPU / CPU
    gpus = tf.config.list_physical_devices("GPU")
    device_info = f"GPU ({len(gpus)} device(s))" if gpus else "CPU (oneDNN enabled)"
    logger.info("Hardware execution environment: %s", device_info)

    # Check manifest
    if not manifest_path.exists():
        logger.warning("Manifest not found at %s.", manifest_path)
        return {
            "status": "manifest_not_found",
            "message": f"Manifest not found at '{manifest_path}'. Run scripts/prepare_dataset.py first.",
        }

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    records = manifest_data.get("records", [])
    class_indices = manifest_data.get("class_indices", {})
    num_classes = len(class_indices)

    if not records or num_classes < 2:
        logger.warning(
            "Manifest has insufficient data: %d records, %d classes.",
            len(records),
            num_classes,
        )
        return {
            "status": "insufficient_data",
            "records_found": len(records),
            "classes_found": num_classes,
            "message": (
                "Insufficient records or classes to train a new model. "
                "Ensure at least 2 classes with valid images exist in dataset/."
            ),
        }

    train_recs = [r for r in records if r.get("split") == "train"]
    val_recs = [r for r in records if r.get("split") == "val"]

    logger.info("Training Split: %d images | Validation Split: %d images", len(train_recs), len(val_recs))
    if not train_recs or not val_recs:
        return {
            "status": "split_empty",
            "message": "Train or validation split is empty. Cannot train.",
        }

    # Data pipelines
    train_ds = build_tf_dataset(train_recs, batch_size=batch_size, is_training=True, seed=random_seed)
    val_ds = build_tf_dataset(val_recs, batch_size=batch_size, is_training=False, seed=random_seed)

    # Build or Load model
    resumed = False
    if resume_from and Path(resume_from).exists():
        resume_p = Path(resume_from)
        logger.info("Resuming training from existing model: %s", resume_p)
        model = tf.keras.models.load_model(str(resume_p), compile=False)
        actual_outputs = model.output_shape[-1]
        if actual_outputs != num_classes:
            raise ValueError(
                f"Model at '{resume_p}' output dimension ({actual_outputs}) does not match manifest ({num_classes})."
            )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        resumed = True
    else:
        logger.info("Building fresh MobileNetV2 classifier with %d classes...", num_classes)
        model = build_disease_classifier(
            num_classes=num_classes,
            input_shape=(224, 224, 3),
            weights="imagenet",
            dropout_rate=0.3,
            initial_learning_rate=learning_rate,
        )

    # Checkpoint setup
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    history_csv_path.parent.mkdir(parents=True, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(output_model_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(history_csv_path), append=False),
    ]

    best_val_loss = 999.0
    combined_history: Dict[str, List[float]] = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}

    # Stage 1: Head training
    if not fine_tune_only and epochs > 0:
        logger.info("--- Stage 1: Training Classification Head (%d epochs) ---", epochs)
        t0 = time.time()
        h1 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1,
        )
        stage1_time = time.time() - t0
        logger.info("Stage 1 completed in %.2f seconds.", stage1_time)
        best_val_loss = min(h1.history.get("val_loss", [999.0]))
        for k in combined_history:
            if k in h1.history:
                combined_history[k].extend(h1.history[k])
    else:
        logger.info("Skipping Stage 1 (Classification head already trained or fine-tune only specified).")

    # Stage 2: Fine-Tuning (Optional)
    if fine_tune and fine_tune_epochs > 0:
        logger.info("--- Stage 2: Fine-Tuning Upper Layers (%d epochs) ---", fine_tune_epochs)
        unfreeze_for_fine_tuning(model, num_layers_to_unfreeze=30, fine_tune_learning_rate=fine_tune_lr)

        append_history = not fine_tune_only and epochs > 0
        ft_callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=3,
                restore_best_weights=True,
                verbose=1,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(output_model_path),
                monitor="val_loss",
                save_best_only=True,
                verbose=1,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=2,
                min_lr=1e-7,
                verbose=1,
            ),
            tf.keras.callbacks.CSVLogger(str(history_csv_path), append=append_history),
        ]

        t1 = time.time()
        h2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=fine_tune_epochs,
            callbacks=ft_callbacks,
            verbose=1,
        )
        stage2_time = time.time() - t1
        logger.info("Stage 2 completed in %.2f seconds.", stage2_time)
        h2_val_loss = min(h2.history.get("val_loss", [999.0]))
        best_val_loss = min(best_val_loss, h2_val_loss)
        for k in combined_history:
            if k in h2.history:
                combined_history[k].extend(h2.history[k])

    # Plot curves
    plot_training_curves(combined_history, plot_output_path)

    # Save model artifact if not already saved by checkpoint
    if not output_model_path.exists():
        model.save(str(output_model_path))

    # Compute model SHA-256
    model_sha = compute_file_sha256(output_model_path)

    # Save training configuration
    train_config = {
        "architecture": "MobileNetV2",
        "input_shape": [224, 224, 3],
        "preprocessing": "mobilenet_v2.preprocess_input [-1.0, 1.0]",
        "num_classes": num_classes,
        "epochs": 0 if fine_tune_only else epochs,
        "batch_size": batch_size,
        "initial_learning_rate": learning_rate,
        "fine_tune": fine_tune,
        "fine_tune_epochs": fine_tune_epochs,
        "fine_tune_lr": fine_tune_lr,
        "random_seed": random_seed,
        "resumed_from": str(resume_from) if resumed else None,
        "device": device_info,
        "manifest_path": str(manifest_path.resolve()),
        "train_samples": len(train_recs),
        "val_samples": len(val_recs),
        "trained_at": datetime.now().isoformat(),
        "model_sha256": model_sha,
    }

    config_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_output_path, "w", encoding="utf-8") as f:
        json.dump(train_config, f, indent=2)
    logger.info("Saved training configuration: %s", config_output_path)

    # Save model metadata
    metadata = {
        "model_version": f"MobileNetV2-{datetime.now().strftime('%Y%m%d')}",
        "architecture": "MobileNetV2",
        "input_shape": [224, 224, 3],
        "preprocessing": "mobilenet_v2.preprocess_input",
        "num_classes": num_classes,
        "classes": sorted(list(class_indices.values())),
        "class_indices": class_indices,
        "model_sha256": model_sha,
        "best_val_loss": round(float(best_val_loss), 4),
        "trained_at": datetime.now().isoformat(),
    }
    metadata_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved model metadata: %s", metadata_output_path)

    return {
        "status": "trained",
        "model_path": str(output_model_path),
        "model_sha256": model_sha,
        "num_classes": num_classes,
        "best_val_loss": best_val_loss,
    }


def main():
    parser = argparse.ArgumentParser(description="Train MobileNetV2 Foliar Disease Classifier")
    parser.add_argument(
        "--manifest",
        type=str,
        default=str(BASE_DIR / "reports" / "dataset_manifest.json"),
        help="Path to dataset manifest JSON",
    )
    parser.add_argument(
        "--output-model",
        type=str,
        default=str(BASE_DIR / "models" / "best_model.keras"),
        help="Output path for best model weights",
    )
    parser.add_argument(
        "--config-output",
        type=str,
        default=str(BASE_DIR / "models" / "training_config.json"),
        help="Output path for training config JSON",
    )
    parser.add_argument(
        "--metadata-output",
        type=str,
        default=str(BASE_DIR / "models" / "model_metadata.json"),
        help="Output path for model metadata JSON",
    )
    parser.add_argument(
        "--history-csv",
        type=str,
        default=str(BASE_DIR / "reports" / "training_history.csv"),
        help="Output path for training history CSV",
    )
    parser.add_argument(
        "--plot-output",
        type=str,
        default=str(BASE_DIR / "reports" / "training" / "loss_accuracy_curve.png"),
        help="Output path for training curves plot",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Stage 1 training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (configurable for RAM/VRAM limits)")
    parser.add_argument("--lr", type=float, default=0.001, help="Initial Adam learning rate")
    parser.add_argument("--fine-tune", action="store_true", help="Enable stage 2 fine-tuning")
    parser.add_argument("--fine-tune-epochs", type=int, default=5, help="Stage 2 fine-tuning epochs")
    parser.add_argument("--fine-tune-lr", type=float, default=1e-5, help="Stage 2 learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to existing model weights/checkpoint to continue training from",
    )
    parser.add_argument(
        "--fine-tune-only",
        action="store_true",
        help="Skip Stage 1 head training and directly fine-tune top backbone layers",
    )

    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    output_model = Path(args.output_model)

    fine_tune_enabled = args.fine_tune or args.fine_tune_only

    print("=" * 75)
    print("MobileNetV2 Multi-Dataset Foliar Disease Training Pipeline")
    print("=" * 75)
    print(f"Manifest Path: {manifest_path.resolve()}")
    print(f"Model Output:  {output_model.resolve()}")
    print(f"Batch Size:    {args.batch_size}")
    if args.resume_from:
        print(f"Resume Model:  {args.resume_from}")
    print(f"Stage 1 Epochs:{0 if args.fine_tune_only else args.epochs} (LR: {args.lr})")
    print(f"Fine-Tuning:   {fine_tune_enabled} (Epochs: {args.fine_tune_epochs}, LR: {args.fine_tune_lr})")
    print("-" * 75)

    res = train_pipeline(
        manifest_path=manifest_path,
        output_model_path=output_model,
        config_output_path=Path(args.config_output),
        history_csv_path=Path(args.history_csv),
        plot_output_path=Path(args.plot_output),
        metadata_output_path=Path(args.metadata_output),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        fine_tune=fine_tune_enabled,
        fine_tune_epochs=args.fine_tune_epochs,
        fine_tune_lr=args.fine_tune_lr,
        random_seed=args.seed,
        resume_from=args.resume_from,
        fine_tune_only=args.fine_tune_only,
    )

    print("\n[TRAINING RESULT]")
    print(f"Status: {res.get('status')}")
    print(f"Message/Path: {res.get('message') or res.get('model_path')}")
    print("=" * 75)


if __name__ == "__main__":
    main()
