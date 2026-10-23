"""Turn images into (label, confidence) rows.

The Predictor is the only code path from pixels to answers. The CLI, the
notebook's assessment cell and the web API all call it, so a fix here is a
fix everywhere.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from catdog.config import (
    ARTIFACTS_DIR,
    CAT,
    DOG,
    LABELS,
    METADATA_FILENAME,
    MODEL_FILENAME,
    NEITHER,
    DecisionPolicy,
)
from catdog.inference.image_io import ImageLoadError, list_images, load_image


@dataclass
class Prediction:
    filename: str
    label: str  # "cat" | "dog" | "neither" | "error"
    confidence: float  # probability of `label` under the model, 0-1
    probabilities: dict[str, float] = field(default_factory=dict)
    # "model": the arg-max of the 3-way softmax.
    # "threshold": the model leaned towards an animal, but too weakly, so the
    #              decision policy overrode it to "neither".
    decided_by: str = "model"
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def decide(probs: np.ndarray, policy: DecisionPolicy) -> tuple[int, str]:
    """Apply the decision policy to one probability vector (cat, dog, neither)."""
    top = int(np.argmax(probs))
    if top != NEITHER and max(probs[CAT], probs[DOG]) < policy.animal_threshold:
        return NEITHER, "threshold"
    return top, "model"


class Predictor:
    def __init__(self, model, image_size: int, policy: DecisionPolicy | None = None, metadata: dict | None = None):
        self.model = model
        self.image_size = image_size
        self.policy = policy or DecisionPolicy()
        self.metadata = metadata or {}

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, model_dir: str | Path = ARTIFACTS_DIR) -> "Predictor":
        """Load `model.keras` + `model_meta.json` from a directory. No retraining."""
        import keras  # imported lazily: the server starts faster without it

        model_dir = Path(model_dir)
        model_path = model_dir / MODEL_FILENAME
        if not model_path.exists():
            raise FileNotFoundError(
                f"No model at {model_path}. Train one (python -m catdog train) "
                f"or copy model.keras and {METADATA_FILENAME} into {model_dir}."
            )
        meta_path = model_dir / METADATA_FILENAME
        metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        if tuple(metadata.get("labels", LABELS)) != LABELS:
            raise ValueError(f"Model labels {metadata['labels']} do not match code labels {LABELS}")

        model = keras.models.load_model(model_path, compile=False)
        policy = DecisionPolicy(**metadata.get("policy", {}))
        image_size = int(metadata.get("image_size", model.input_shape[1]))
        return cls(model, image_size, policy, metadata)

    # ---------------------------------------------------------------- inference
    def predict_probabilities(self, images: Sequence[np.ndarray], batch_size: int = 16) -> np.ndarray:
        """RGB uint8 arrays of any size -> (N, 3) probabilities."""
        import tensorflow as tf

        from catdog.preprocessing import resize_for_model

        out = []
        for start in range(0, len(images), batch_size):
            batch = tf.stack([resize_for_model(img, self.image_size) for img in images[start : start + batch_size]])
            probs = self.model(batch, training=False).numpy()
            if self.policy.tta:
                probs = (probs + self.model(tf.image.flip_left_right(batch), training=False).numpy()) / 2
            out.append(probs)
        return np.concatenate(out) if out else np.zeros((0, len(LABELS)), dtype=np.float32)

    def predict_sources(self, items: Iterable[tuple[str, str | Path | bytes]], batch_size: int = 16) -> list[Prediction]:
        """Predict (name, path-or-bytes) pairs. A file that fails to decode
        becomes an "error" row instead of crashing the whole run."""
        items = list(items)
        results: list[Prediction | None] = [None] * len(items)
        positions, images = [], []
        for i, (name, source) in enumerate(items):
            try:
                images.append(load_image(source))
                positions.append(i)
            except ImageLoadError as exc:
                results[i] = Prediction(name, "error", 0.0, error=str(exc))

        probs = self.predict_probabilities(images, batch_size=batch_size)
        for i, p in zip(positions, probs):
            idx, decided_by = decide(p, self.policy)
            results[i] = Prediction(
                filename=items[i][0],
                label=LABELS[idx],
                confidence=float(p[idx]),
                probabilities={label: float(v) for label, v in zip(LABELS, p)},
                decided_by=decided_by,
            )
        return results  # type: ignore[return-value]

    def predict_folder(self, folder: str | Path, recursive: bool = False) -> list[Prediction]:
        folder = Path(folder)
        paths = list_images(folder, recursive=recursive)
        return self.predict_sources([(str(p.relative_to(folder)), p) for p in paths])

    def info(self) -> dict:
        return {**self.metadata, "policy": asdict(self.policy), "image_size": self.image_size}


def format_table(predictions: Sequence[Prediction]) -> str:
    """The deliverable table from the brief: filename, prediction, confidence."""
    width = max([len("filename"), *(len(p.filename) for p in predictions)]) + 2
    lines = [f"{'filename':<{width}}{'prediction':<12}{'confidence':>10}"]
    for p in predictions:
        conf = "—" if p.label == "error" else f"{p.confidence * 100:.1f}%"
        flag = "  (below animal threshold)" if p.decided_by == "threshold" else ""
        flag = f"  ({p.error})" if p.error else flag
        lines.append(f"{p.filename:<{width}}{p.label:<12}{conf:>10}{flag}")
    counts = {label: sum(p.label == label for p in predictions) for label in (*LABELS, "error")}
    summary = "  ".join(f"{k}: {v}" for k, v in counts.items() if v or k != "error")
    lines.append(f"\n{len(predictions)} images  ·  {summary}")
    return "\n".join(lines)


def predict_folder_table(folder: str | Path, model_dir: str | Path = ARTIFACTS_DIR) -> list[Prediction]:
    """One call for assessment day: load the saved model, predict, print the table."""
    started = time.perf_counter()
    predictor = Predictor.load(model_dir)
    predictions = predictor.predict_folder(folder)
    print(format_table(predictions))
    print(f"Done in {time.perf_counter() - started:.1f}s")
    return predictions
