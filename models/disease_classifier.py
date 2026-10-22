"""
MobileNetV2 Plant Disease Classifier Architecture (models/disease_classifier.py)
Implements transfer learning model construction, head fine-tuning,
layer freezing/unfreezing, and compilation for multi-class foliar disease diagnosis.
"""

import logging
from typing import List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


def build_disease_classifier(
    num_classes: int,
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    weights: Optional[str] = "imagenet",
    dropout_rate: float = 0.3,
    initial_learning_rate: float = 0.001,
) -> "tf.keras.Model":
    """
    Constructs a transfer learning model with a MobileNetV2 feature extractor
    and a custom dense classification head.

    Args:
        num_classes: Dynamic number of plant disease categories.
        input_shape: Input image shape (default: 224, 224, 3).
        weights: "imagenet" for pre-trained weights, or None for random initialization (offline/tests).
        dropout_rate: Dropout rate for regularization before the output layer.
        initial_learning_rate: Learning rate for Adam optimizer.

    Returns:
        Compiled tf.keras.Model with frozen MobileNetV2 backbone.
    """
    import tensorflow as tf

    if num_classes < 2:
        raise ValueError(f"num_classes must be at least 2 for multi-class classification (got {num_classes}).")

    logger.info(
        "Building MobileNetV2 classifier: num_classes=%d, input_shape=%s, weights=%s",
        num_classes,
        input_shape,
        weights,
    )

    # 1. Base MobileNetV2 feature extractor
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights=weights,
    )

    # 2. Freeze the base model for Initial Stage (head training only)
    base_model.trainable = False

    # 3. Build classification head
    inputs = tf.keras.Input(shape=input_shape, name="input_leaf_image")
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pool")(x)
    if dropout_rate > 0.0:
        x = tf.keras.layers.Dropout(dropout_rate, name="dropout_regularization")(x)

    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        name="disease_predictions",
    )(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="PlantCare_MobileNetV2")

    # 4. Compile model with Adam and Sparse Categorical Crossentropy
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=initial_learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def unfreeze_for_fine_tuning(
    model: "tf.keras.Model",
    num_layers_to_unfreeze: int = 30,
    fine_tune_learning_rate: float = 1e-5,
    **kwargs,
) -> "tf.keras.Model":
    """
    Unfreezes top layers of the MobileNetV2 backbone for Stage 2 fine-tuning.
    Keeps BatchNormalization layers frozen to protect moving mean and variance estimates.

    Args:
        model: Trained Stage 1 Keras model.
        num_layers_to_unfreeze: Number of top backbone layers to unfreeze (from the end).
        fine_tune_learning_rate: Low learning rate for gentle weight adjustments (e.g., 1e-5).

    Returns:
        Recompiled tf.keras.Model ready for fine-tuning.
    """
    if "fine_tune_layers" in kwargs:
        num_layers_to_unfreeze = kwargs["fine_tune_layers"]
    if "learning_rate" in kwargs:
        fine_tune_learning_rate = kwargs["learning_rate"]
    import tensorflow as tf

    # Locate base model layer
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) or "mobilenetv2" in layer.name.lower():
            base_model = layer
            break

    if base_model is None:
        logger.warning("Could not find nested base model; unfreezing top layers on main model.")
        base_model = model

    base_model.trainable = True

    # Freeze all layers except the last `num_layers_to_unfreeze`
    layers_count = len(base_model.layers)
    freeze_until = max(0, layers_count - num_layers_to_unfreeze)

    for i, layer in enumerate(base_model.layers):
        if i < freeze_until or isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True

    # Recompile with smaller learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=fine_tune_learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    logger.info(
        "Model prepared for fine-tuning: %d layers unfrozen, lr=%e",
        num_layers_to_unfreeze,
        fine_tune_learning_rate,
    )
    return model
