"""
MobileNetV2 Training Pipeline (utils/train_model.py)
Loads the dataset manifest and class indices, builds a MobileNetV2 transfer learning model,
applies foliar data augmentation to the training split, executes stage-1 and optional
stage-2 fine-tuning, and saves best validation weights alongside training metadata.

Usage:
    python utils/train_model.py --epochs 10 --batch-size 16
    python utils/train_model.py --epochs 10 --fine-tune --fine-tune-epochs 5
"""

import argparse
import hashlib
import json
import logging
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
from models.disease_classifier import build_disease_classifier, unfreeze_for_fine_tuning
from services.image_preprocessing import get_augmentation_pipeline, load_and_preprocess_image

logger = logging.getLogger(__name__)


def compute_file_sha256(filepath: Path) -> str:
    """Computes SHA-256 digest of saved model artifact for provenance auditing."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def create_tf_dataset(
    records: List[Dict[str, Any]],
    dataset_root: Path,
    image_size: Tuple[int, int] = (224, 224),
    batch_size: int = 16,
    is_training: bool = False,
    seed: int = 42,
) -> "tf.data.Dataset":
    """
    Builds a high-performance tf.data.Dataset from manifest records.
    Augmentation is applied ONLY when is_training is True.
    """
    import tensorflow as tf

    paths = [str(dataset_root / r["relative_path"]) for r in records]
    labels = [int(r["class_index"]) for r in records]

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if is_training:
        ds = ds.shuffle(buffer_size=min(len(records), 1000), seed=seed, reshuffle_each_iteration=True)

    def _load_image_tensor(path_tensor, label_tensor):
        # py_function wrapper to invoke load_and_preprocess_image using Pillow
        def _py_loader(p):
            p_str = p.decode("utf-8")
            arr = load_and_preprocess_image(p_str, target_size=image_size)
            return arr.astype(np.float32)

        img = tf.numpy_function(_py_loader, [path_tensor], tf.float32)
        img.set_shape([image_size[0], image_size[1], 3])
        return img, label_tensor

    ds = ds.map(_load_image_tensor, num_parallel_calls=tf.data.AUTOTUNE)

    if is_training:
        aug_layer = get_augmentation_pipeline(seed=seed)
        ds = ds.map(lambda x, y: (aug_layer(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def train_disease_classifier(
    dataset_root: Optional[Path] = None,
    manifest_path: Optional[Path] = None,
    class_indices_path: Optional[Path] = None,
    model_output_dir: Optional[Path] = None,
    image_size: int = 224,
    batch_size: int = 16,
    epochs: int = 10,
    fine_tune: bool = False,
    fine_tune_epochs: int = 5,
    initial_learning_rate: float = 0.001,
    fine_tune_learning_rate: float = 1e-5,
    patience: int = 3,
    random_seed: int = 42,
    weights: Optional[str] = "imagenet",
) -> Dict[str, Any]:
    """
    Executes the MobileNetV2 transfer learning training pipeline.
    """
    import tensorflow as tf

    # Set deterministic random seeds
    random.seed(random_seed)
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)

    dataset_root = dataset_root or (Config.BASE_DIR / "dataset" / "raw")
    manifest_path = manifest_path or (Config.BASE_DIR / "reports" / "training" / "dataset_manifest.json")
    class_indices_path = class_indices_path or (Config.BASE_DIR / "model" / "class_indices.json")
    model_output_dir = model_output_dir or (Config.BASE_DIR / "model")

    # 1. Check prerequisites
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Dataset manifest not found at '{manifest_path}'. "
            "Please run 'python utils/prepare_dataset.py' to generate the manifest before training."
        )

    if not class_indices_path.exists():
        raise FileNotFoundError(
            f"Class indices dictionary not found at '{class_indices_path}'. "
            "Please run 'python utils/prepare_dataset.py' first."
        )

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    with open(class_indices_path, "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)

    train_records = manifest_data.get("manifest", {}).get("train", [])
    val_records = manifest_data.get("manifest", {}).get("val", [])

    if not train_records or not val_records:
        raise ValueError(
            f"Manifest at '{manifest_path}' does not contain sufficient train/val records. "
            f"Found {len(train_records)} train, {len(val_records)} val."
        )

    num_classes = len(class_to_idx)
    logger.info("Starting training for %d classes with %d train and %d val samples.", num_classes, len(train_records), len(val_records))

    # Detect hardware
    gpus = tf.config.list_physical_devices("GPU")
    device_name = f"GPU ({len(gpus)} device(s))" if gpus else "CPU"
    print(f"[INFO] Compute Environment: {device_name}")

    # Build datasets
    train_ds = create_tf_dataset(
        records=train_records,
        dataset_root=dataset_root,
        image_size=(image_size, image_size),
        batch_size=batch_size,
        is_training=True,
        seed=random_seed,
    )

    val_ds = create_tf_dataset(
        records=val_records,
        dataset_root=dataset_root,
        image_size=(image_size, image_size),
        batch_size=batch_size,
        is_training=False,
        seed=random_seed,
    )

    # 2. Build model
    model = build_disease_classifier(
        num_classes=num_classes,
        input_shape=(image_size, image_size, 3),
        weights=weights,
        initial_learning_rate=initial_learning_rate,
    )

    # Setup directories
    model_output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = model_output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    reports_training_dir = Config.BASE_DIR / "reports" / "training"
    reports_training_dir.mkdir(parents=True, exist_ok=True)

    best_model_path = model_output_dir / "plant_disease_model.keras"
    checkpoint_filepath = checkpoints_dir / "best_checkpoint.keras"
    csv_log_path = reports_training_dir / "training_history.csv"

    # Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_filepath),
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=max(1, patience - 1),
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(csv_log_path), append=False),
    ]

    start_time = time.time()

    # 3. Stage 1: Train classification head with frozen backbone
    print("\n--- Starting Stage 1 Training: Classification Head ---")
    history_stage1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
    )

    total_history = {k: [float(v) for v in vals] for k, vals in history_stage1.history.items()}

    # 4. Optional Stage 2: Fine-Tuning
    if fine_tune:
        print("\n--- Starting Stage 2 Training: Fine-Tuning Top Backbone Layers ---")
        model = unfreeze_for_fine_tuning(
            model,
            num_layers_to_unfreeze=30,
            fine_tune_learning_rate=fine_tune_learning_rate,
        )
        history_stage2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs + fine_tune_epochs,
            initial_epoch=len(history_stage1.epoch),
            callbacks=callbacks,
        )
        for k, vals in history_stage2.history.items():
            total_history.setdefault(k, []).extend([float(v) for v in vals])

    duration_sec = round(time.time() - start_time, 2)

    # 5. Save final best model
    if checkpoint_filepath.exists():
        # Load best weights into model
        model = tf.keras.models.load_model(str(checkpoint_filepath))
    model.save(str(best_model_path))

    # Also export to saved_model directory
    saved_model_dir = model_output_dir / "saved_model"
    try:
        model.export(str(saved_model_dir))
    except Exception as err:
        logger.warning("SavedModel format export skipped or unsupported: %s", err)

    model_checksum = compute_file_sha256(best_model_path)

    # 6. Save metadata
    metadata = {
        "model_name": "PlantCare_MobileNetV2",
        "training_timestamp": datetime.now().isoformat(),
        "tensorflow_version": tf.__version__,
        "hardware_device": device_name,
        "input_shape": [image_size, image_size, 3],
        "num_classes": num_classes,
        "class_indices": class_to_idx,
        "training_duration_seconds": duration_sec,
        "hyperparameters": {
            "batch_size": batch_size,
            "epochs_configured": epochs,
            "fine_tuning_enabled": fine_tune,
            "fine_tune_epochs": fine_tune_epochs if fine_tune else 0,
            "initial_learning_rate": initial_learning_rate,
            "fine_tune_learning_rate": fine_tune_learning_rate if fine_tune else None,
            "random_seed": random_seed,
            "patience": patience,
        },
        "model_artifact": {
            "path": str(best_model_path.relative_to(Config.BASE_DIR)),
            "sha256": model_checksum,
            "format": "Keras (.keras)",
        },
        "final_epoch_metrics": {
            k: round(v[-1], 4) for k, v in total_history.items() if len(v) > 0
        },
    }

    metadata_path = model_output_dir / "model_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Train MobileNetV2 transfer learning model on plant disease dataset."
    )
    parser.add_argument("--dataset-root", type=str, default=str(Config.BASE_DIR / "dataset" / "raw"))
    parser.add_argument("--manifest-path", type=str, default=str(Config.BASE_DIR / "reports" / "training" / "dataset_manifest.json"))
    parser.add_argument("--class-indices-path", type=str, default=str(Config.BASE_DIR / "model" / "class_indices.json"))
    parser.add_argument("--model-output-dir", type=str, default=str(Config.BASE_DIR / "model"))
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--fine-tune", action="store_true", help="Enable Stage 2 fine-tuning")
    parser.add_argument("--fine-tune-epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--fine-tune-lr", type=float, default=0.00001)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    print("=" * 70)
    print("MobileNetV2 Plant Disease Classifier — Training Pipeline")
    print("=" * 70)
    print(f"Dataset Root:    {args.dataset_root}")
    print(f"Manifest:        {args.manifest_path}")
    print(f"Image Size:      {args.image_size}x{args.image_size}")
    print(f"Batch Size:      {args.batch_size}")
    print(f"Epochs:          {args.epochs} (Fine-tuning: {args.fine_tune})")
    print(f"Output Dir:      {args.model_output_dir}")
    print("-" * 70)

    try:
        metadata = train_disease_classifier(
            dataset_root=Path(args.dataset_root),
            manifest_path=Path(args.manifest_path),
            class_indices_path=Path(args.class_indices_path),
            model_output_dir=Path(args.model_output_dir),
            image_size=args.image_size,
            batch_size=args.batch_size,
            epochs=args.epochs,
            fine_tune=args.fine_tune,
            fine_tune_epochs=args.fine_tune_epochs,
            initial_learning_rate=args.learning_rate,
            fine_tune_learning_rate=args.fine_tune_lr,
            patience=args.patience,
            random_seed=args.seed,
        )

        print("\n" + "=" * 70)
        print("[SUCCESS] Model Training Completed!")
        print(f"Artifact Saved: {metadata['model_artifact']['path']}")
        print(f"SHA-256 Digest: {metadata['model_artifact']['sha256']}")
        print("Final Metrics:")
        for metric, val in metadata["final_epoch_metrics"].items():
            print(f"  * {metric}: {val}")
        print("=" * 70)

    except Exception as err:
        print(f"\n[ERROR] Training could not proceed: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
