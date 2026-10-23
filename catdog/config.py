"""Central configuration.

Everything that must agree between training and inference (labels, image
size, preprocessing) lives here, so the two can never drift apart.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Order matters: it is the order of the softmax outputs.
LABELS: tuple[str, ...] = ("cat", "dog", "neither")
CAT, DOG, NEITHER = range(3)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = Path(os.environ.get("CATDOG_ARTIFACTS", PROJECT_ROOT / "artifacts"))
DATA_DIR = Path(os.environ.get("CATDOG_DATA", PROJECT_ROOT / "data"))

# Your own photos, one sub-folder per label: data/own/{cat,dog,neither}/*.jpg
OWN_PHOTOS_DIR = DATA_DIR / "own"
# A held-out set that is never trained on: data/holdout/{cat,dog,neither}/*.jpg
HOLDOUT_DIR = DATA_DIR / "holdout"

MODEL_FILENAME = "model.keras"
METADATA_FILENAME = "model_meta.json"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff", ".heic", ".heif"}


@dataclass
class TrainConfig:
    """Hyper-parameters for one training run. Saved next to the model."""

    image_size: int = 224
    batch_size: int = 32
    backbone: str = "efficientnetv2b0"

    # Stage 1: frozen backbone, train the head only.
    head_epochs: int = 6
    head_lr: float = 1e-3
    # Stage 2: unfreeze the top of the backbone and fine-tune gently.
    finetune_epochs: int = 4
    finetune_lr: float = 1e-5
    finetune_layers: int = 40
    dropout: float = 0.3
    label_smoothing: float = 0.05

    val_fraction: float = 0.15
    seed: int = 42
    # Cap per class so the larger dog set does not dominate. None = no cap.
    max_per_class: int | None = 4000
    # Smoke-test mode: use only this many images per source ("start with fifty").
    limit: int | None = None

    # Every source mixed into training (see catdog.data.sources.SOURCES).
    sources: list[str] = field(default_factory=lambda: ["oxford_iiit_pet", "caltech101", "imagenette", "own"])

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DecisionPolicy:
    """How raw softmax probabilities become a final label.

    The 3-class model already has a "neither" output. As a second safety net,
    if the strongest animal probability is below `animal_threshold`, the
    image is called "neither" anyway. The threshold is tuned on validation
    data by `catdog.evaluation.threshold` and stored in the model metadata.
    """

    animal_threshold: float = 0.55
    tta: bool = True  # average the prediction with a horizontally flipped copy
