"""Train, measure, tune and save — in that order, every time.

    three-class (Approach A, the shipped model):
        artifacts/model.keras + artifacts/model_meta.json + artifacts/reports/*
    two-class   (Approach B, the evidence):
        artifacts/approach_b/model.keras + artifacts/reports/approach_b.json
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf

from catdog.config import ARTIFACTS_DIR, LABELS, METADATA_FILENAME, MODEL_FILENAME, NEITHER, DecisionPolicy, TrainConfig
from catdog.data.exclusions import CALTECH101_EXCLUDE, CALTECH101_KEEP_NOTES, IMAGENETTE_EXCLUDE
from catdog.data.pipeline import build_datasets
from catdog.evaluation.metrics import evaluate, format_report, plot_confusion_matrix
from catdog.evaluation.threshold import animal_max_prob, threshold_study, tune_animal_threshold
from catdog.inference.predictor import decide
from catdog.modeling.model import build_model, unfreeze_top


def _one_hot(num_classes: int, smoothing: float):
    def fn(x, y):
        return x, tf.one_hot(y, num_classes) * (1 - smoothing) + smoothing / num_classes

    return fn


def _fit(model, base, train_ds, val_ds, cfg: TrainConfig, class_weight: dict[int, float]) -> dict:
    callbacks = lambda: [  # noqa: E731
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=1),
    ]
    history: dict[str, list[float]] = {}

    print(f"[train] stage 1/2: head only, {cfg.head_epochs} epochs")
    model.compile(keras.optimizers.Adam(cfg.head_lr), loss="categorical_crossentropy", metrics=["accuracy"])
    h = model.fit(train_ds, validation_data=val_ds, epochs=cfg.head_epochs, class_weight=class_weight, callbacks=callbacks())
    for k, v in h.history.items():
        history.setdefault(k, []).extend(float(x) for x in v)

    if cfg.finetune_epochs:
        print(f"[train] stage 2/2: fine-tune top {cfg.finetune_layers} backbone layers, {cfg.finetune_epochs} epochs")
        unfreeze_top(base, cfg.finetune_layers)
        model.compile(keras.optimizers.Adam(cfg.finetune_lr), loss="categorical_crossentropy", metrics=["accuracy"])
        h = model.fit(train_ds, validation_data=val_ds, epochs=cfg.finetune_epochs, class_weight=class_weight, callbacks=callbacks())
        for k, v in h.history.items():
            history.setdefault(k, []).extend(float(x) for x in v)
    return history


def _collect(model, ds: tf.data.Dataset, tta: bool = DecisionPolicy.tta) -> tuple[np.ndarray, np.ndarray]:
    """Validation probabilities, computed exactly as the Predictor will (incl. flip TTA)."""
    probs, ys = [], []
    for x, y in ds:
        p = model(x, training=False).numpy()
        if tta:
            p = (p + model(tf.image.flip_left_right(x), training=False).numpy()) / 2
        probs.append(p)
        ys.append(y.numpy())
    return np.concatenate(probs), np.concatenate(ys)


def train(
    cfg: TrainConfig | None = None,
    out_dir: str | Path = ARTIFACTS_DIR,
    two_class: bool = False,
    pretrained: bool = True,
    tfds_dir: str | None = None,
) -> dict:
    cfg = cfg or TrainConfig()
    keras.utils.set_random_seed(cfg.seed)
    out_dir = Path(out_dir)
    reports = out_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    started = time.time()

    bundle = build_datasets(cfg, two_class=two_class, tfds_dir=tfds_dir)
    labels = LABELS[:2] if two_class else LABELS
    n = len(labels)
    print(f"[data] train totals {bundle.totals('train')}  val totals {bundle.totals('val')}")

    fit_val = bundle.val
    if two_class:
        n_pets_val = sum(bundle.totals("val")[label] for label in labels)
        fit_val = (
            bundle.val.unbatch()
            .filter(lambda _, y: y != NEITHER)
            .apply(tf.data.experimental.assert_cardinality(n_pets_val))
            .batch(cfg.batch_size)
        )
    model, base = build_model(n, cfg.image_size, cfg.backbone, cfg.dropout, pretrained, cfg.seed)
    history = _fit(
        model,
        base,
        bundle.train.map(_one_hot(n, cfg.label_smoothing)),
        fit_val.map(_one_hot(n, 0.0)),
        cfg,
        bundle.class_weights(labels),
    )

    probs, y_true = _collect(model, bundle.val)
    common = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "train_seconds": round(time.time() - started, 1),
        "config": cfg.to_dict(),
        "train_counts": bundle.train_counts,
        "val_counts": bundle.val_counts,
        "history": history,
        "versions": {"tensorflow": tf.__version__, "keras": keras.__version__, "python": platform.python_version()},
    }

    if two_class:
        return _finish_two_class(model, probs, y_true, out_dir, reports, common)
    return _finish_three_class(model, probs, y_true, cfg, out_dir, reports, common)


def _finish_two_class(model, probs, y_true, out_dir: Path, reports: Path, common: dict) -> dict:
    """Approach B: would a threshold on a cats-vs-dogs model find 'neither'?"""
    top = probs.max(axis=1)
    study = threshold_study(top[y_true != NEITHER], top[y_true == NEITHER])
    # How the plain 2-class model (no threshold) does on cats vs dogs only:
    pets = y_true != NEITHER
    pet_accuracy = float((probs[pets].argmax(1) == y_true[pets]).mean()) if pets.any() else None
    result = {**common, "approach": "B-two-class-threshold", "pet_accuracy": pet_accuracy, "threshold_study": study}

    (out_dir / "approach_b").mkdir(exist_ok=True)
    model.save(out_dir / "approach_b" / MODEL_FILENAME)
    (reports / "approach_b.json").write_text(json.dumps(result, indent=2))
    best = study["best"]
    print(
        f"\n[approach B] cat-vs-dog accuracy {pet_accuracy:.1%}. Best threshold {best['threshold']:.2f}: "
        f"{best['pet_recall']:.1%} of pets stay cat/dog, {best['neither_recall']:.1%} of negatives become neither. "
        f"{study['negatives_confident_over_90']:.1%} of negatives get >90% confidence as cat/dog."
    )
    return result


def _finish_three_class(model, probs, y_true, cfg: TrainConfig, out_dir: Path, reports: Path, common: dict) -> dict:
    """Approach A: tune the safety-net threshold, evaluate, save model + metadata."""
    raw = evaluate([LABELS[i] for i in y_true], [LABELS[i] for i in probs.argmax(1)])
    threshold, sweep = tune_animal_threshold(probs, y_true)
    policy = DecisionPolicy(animal_threshold=threshold)
    tuned = evaluate([LABELS[i] for i in y_true], [LABELS[decide(p, policy)[0]] for p in probs])
    print("\n[val] raw arg-max:\n" + format_report(raw))
    print(f"\n[val] with animal threshold {threshold:.2f}:\n" + format_report(tuned))

    metadata = {
        **common,
        "labels": list(LABELS),
        "approach": "A-three-class",
        "backbone": cfg.backbone,
        "image_size": cfg.image_size,
        "policy": asdict(policy),
        "validation": {"raw": raw.to_dict(), "tuned": tuned.to_dict(), "threshold_sweep": sweep},
        # Animal-probability histogram on validation, per true class (UI / report).
        "val_animal_prob_hist": {
            LABELS[c]: np.histogram(animal_max_prob(probs[y_true == c]), bins=20, range=(0, 1))[0].tolist() for c in range(3)
        },
        "exclusions": {"caltech101": CALTECH101_EXCLUDE, "imagenette": IMAGENETTE_EXCLUDE, "caltech101_kept_notes": CALTECH101_KEEP_NOTES},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(out_dir / MODEL_FILENAME)
    (out_dir / METADATA_FILENAME).write_text(json.dumps(metadata, indent=2))
    (reports / "validation.txt").write_text(format_report(tuned))
    try:
        plot_confusion_matrix(tuned, reports / "validation_confusion.png", "Validation (held-out slice)")
    except ImportError:
        pass
    print(f"\n[saved] {out_dir / MODEL_FILENAME} and {METADATA_FILENAME}")
    return metadata
