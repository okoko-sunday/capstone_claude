import io
import json
import os

import numpy as np
import pytest
from PIL import Image

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


def jpeg_bytes(w=64, h=48, color=(200, 120, 40), fmt="JPEG", mode="RGB", **save_kwargs) -> bytes:
    buf = io.BytesIO()
    Image.new(mode, (w, h), color).save(buf, fmt, **save_kwargs)
    return buf.getvalue()


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory):
    """A real (untrained) Keras model saved the same way training saves one."""
    import keras

    from catdog.config import LABELS, METADATA_FILENAME, MODEL_FILENAME

    size = 32
    inputs = keras.Input((size, size, 3))
    x = keras.layers.Rescaling(1 / 255)(inputs)
    x = keras.layers.GlobalAveragePooling2D()(x)
    outputs = keras.layers.Dense(3, activation="softmax")(x)
    model = keras.Model(inputs, outputs)

    d = tmp_path_factory.mktemp("model")
    model.save(d / MODEL_FILENAME)
    (d / METADATA_FILENAME).write_text(
        json.dumps({"labels": list(LABELS), "image_size": size, "backbone": "tiny-test", "policy": {"animal_threshold": 0.5, "tta": True}})
    )
    return d


@pytest.fixture
def image_folder(tmp_path):
    (tmp_path / "a.jpg").write_bytes(jpeg_bytes(4000, 3000))
    (tmp_path / "b.png").write_bytes(jpeg_bytes(30, 90, (0, 0, 0, 0), fmt="PNG", mode="RGBA"))
    (tmp_path / "c_gray.jpg").write_bytes(jpeg_bytes(50, 50, 128, mode="L"))
    (tmp_path / "broken.jpg").write_bytes(b"definitely not a jpeg")
    (tmp_path / "notes.txt").write_text("ignore me")
    (tmp_path / ".hidden.jpg").write_bytes(jpeg_bytes())
    return tmp_path


@pytest.fixture
def rng():
    return np.random.default_rng(0)
