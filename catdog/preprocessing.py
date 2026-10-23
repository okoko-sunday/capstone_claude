"""The one resize function shared by training and inference.

If training squashes images and inference pads them (or vice versa), the
model sees a different distribution on assessment day than it was trained
on. Keeping one implementation makes that mistake impossible.

We letterbox (resize with padding) instead of squashing or centre-cropping:
phone photos come in portrait and landscape, and a crop can cut the animal
out of the frame.
"""

from __future__ import annotations

import tensorflow as tf


def resize_for_model(image: tf.Tensor, image_size: int) -> tf.Tensor:
    """uint8/float HxWx3 image -> float32 image_size x image_size x 3 in [0, 255].

    Pixel scaling is *not* done here: the backbone includes its own
    preprocessing layer, so the saved model accepts raw 0-255 pixels.
    """
    image = tf.cast(image, tf.float32)
    image = tf.image.resize_with_pad(image, image_size, image_size, antialias=True)
    return tf.clip_by_value(image, 0.0, 255.0)
