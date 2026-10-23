"""Model definition: pretrained backbone + small classification head.

Why transfer learning: we have thousands of images, not millions.
An ImageNet-pretrained backbone already knows edges, fur, eyes and wheels;
we only teach it our three piles. Training from scratch on this data would
overfit quickly and generalise badly to phone photos.
"""

from __future__ import annotations

import keras
from keras import layers

# All of these include their own input preprocessing, so the saved model
# accepts raw 0-255 pixels and inference code cannot forget to rescale.
BACKBONES = {
    "efficientnetv2b0": keras.applications.EfficientNetV2B0,
    "efficientnetv2b1": keras.applications.EfficientNetV2B1,
    "mobilenetv3large": keras.applications.MobileNetV3Large,
}


def augmentation(seed: int = 42) -> keras.Sequential:
    """Make Oxford-style studio photos look more like real phone photos.

    Active only during training; at inference these layers pass images
    through unchanged, so they are safe to keep inside the saved model.
    """
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal", seed=seed),
            layers.RandomRotation(0.08, fill_mode="constant", seed=seed),
            layers.RandomZoom((-0.25, 0.15), fill_mode="constant", seed=seed),  # off-centre / smaller animals
            layers.RandomTranslation(0.1, 0.1, fill_mode="constant", seed=seed),
            layers.RandomBrightness(0.35, value_range=(0, 255), seed=seed),  # dim indoor light
            layers.RandomContrast(0.3, seed=seed),
        ],
        name="augmentation",
    )


def build_model(
    num_classes: int = 3,
    image_size: int = 224,
    backbone: str = "efficientnetv2b0",
    dropout: float = 0.3,
    pretrained: bool = True,
    seed: int = 42,
) -> tuple[keras.Model, keras.Model]:
    """Returns (model, backbone). The model takes raw RGB pixels in [0, 255]."""
    base = BACKBONES[backbone](
        include_top=False,
        weights="imagenet" if pretrained else None,
        input_shape=(image_size, image_size, 3),
        pooling="avg",
    )
    base.trainable = False

    inputs = keras.Input(shape=(image_size, image_size, 3), name="image")
    x = augmentation(seed)(inputs)
    # training=False keeps BatchNorm statistics frozen, even while fine-tuning.
    x = base(x, training=False)
    x = layers.Dropout(dropout, seed=seed)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="probabilities")(x)
    return keras.Model(inputs, outputs, name=f"catdog_{backbone}_{num_classes}way"), base


def unfreeze_top(base: keras.Model, n_layers: int) -> None:
    """Fine-tuning: unfreeze the last n layers, but never BatchNorm."""
    base.trainable = True
    for layer in base.layers[:-n_layers]:
        layer.trainable = False
    for layer in base.layers[-n_layers:]:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
