"""Mix the sources into balanced, cached train/validation datasets."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import tensorflow as tf

from catdog.config import DATA_DIR, LABELS, NEITHER, TrainConfig
from catdog.data.exclusions import CALTECH101_EXCLUDE, IMAGENETTE_EXCLUDE
from catdog.data.sources import load_source
from catdog.preprocessing import resize_for_model


@dataclass
class DataBundle:
    train: tf.data.Dataset  # batched, shuffled, float32 images
    val: tf.data.Dataset  # batched, deterministic order, includes all 3 classes
    train_counts: dict[str, dict[str, int]] = field(default_factory=dict)  # source -> label -> n
    val_counts: dict[str, dict[str, int]] = field(default_factory=dict)

    def totals(self, which: str = "train") -> dict[str, int]:
        counts = self.train_counts if which == "train" else self.val_counts
        total = Counter()
        for per_label in counts.values():
            total.update(per_label)
        return {label: total.get(label, 0) for label in LABELS}

    def class_weights(self, labels: tuple[str, ...] = LABELS) -> dict[int, float]:
        """Inverse-frequency weights so 'cat' is not drowned out by 'dog'."""
        totals = self.totals("train")
        n = sum(totals[label] for label in labels)
        return {i: n / (len(labels) * max(totals[label], 1)) for i, label in enumerate(labels)}


def _split(ds: tf.data.Dataset, val_fraction: float, want_val: bool) -> tf.data.Dataset:
    # Buckets of 20 keep the split proportional even for a few dozen own photos.
    bucket = max(1, int(round(val_fraction * 20)))
    ds = ds.enumerate().filter(lambda i, _: tf.equal(i % 20 < bucket, want_val))
    return ds.map(lambda _, pair: pair)


def _cache_path(cfg: TrainConfig, source: str, split: str) -> Path:
    # The cache key changes whenever anything that affects the images does.
    key = json.dumps(
        [source, split, cfg.image_size, cfg.seed, cfg.limit, cfg.val_fraction, sorted(CALTECH101_EXCLUDE), sorted(IMAGENETTE_EXCLUDE)]
    )
    digest = hashlib.sha1(key.encode()).hexdigest()[:10]
    path = DATA_DIR / "cache" / f"{source}-{split}-{digest}"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _prepared(cfg: TrainConfig, source: str, split: str, tfds_dir: str | None) -> tf.data.Dataset:
    ds = load_source(source, seed=cfg.seed, limit=cfg.limit, data_dir=tfds_dir)
    ds = _split(ds, cfg.val_fraction, want_val=(split == "val"))
    ds = ds.map(
        # Store uint8 in the cache: 4x smaller than float32.
        lambda img, y: (tf.cast(resize_for_model(img, cfg.image_size), tf.uint8), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return ds.cache(str(_cache_path(cfg, source, split)))


def _count(ds: tf.data.Dataset) -> dict[str, int]:
    """One pass over the data: counts labels and fills the cache."""
    counts = Counter(int(y) for y in ds.map(lambda _, y: y).as_numpy_iterator())
    return {LABELS[k]: v for k, v in sorted(counts.items())}


def build_datasets(cfg: TrainConfig, two_class: bool = False, tfds_dir: str | None = None) -> DataBundle:
    """two_class=True builds the Approach-B training set: cats and dogs only.
    Validation always keeps all three classes so 'neither' can be measured."""
    sources = cfg.sources
    train_parts, val_parts = [], []
    bundle = DataBundle(train=None, val=None)  # type: ignore[arg-type]

    for source in sources:
        train_ds, val_ds = (_prepared(cfg, source, s, tfds_dir) for s in ("train", "val"))
        print(f"[data] preparing {source} (first run downloads and caches; later runs are fast)")
        bundle.train_counts[source] = _count(train_ds)
        bundle.val_counts[source] = _count(val_ds)
        print(f"[data]   train {bundle.train_counts[source]}  val {bundle.val_counts[source]}")
        if two_class:
            train_ds = train_ds.filter(lambda _, y: y != NEITHER)
        n = sum(bundle.train_counts[source].get(label, 0) for label in (LABELS[:2] if two_class else LABELS))
        if n:
            train_parts.append((train_ds.shuffle(1024, seed=cfg.seed).repeat(), n))
        val_parts.append(val_ds)

    if not train_parts:
        raise RuntimeError("No training images found. Check your sources and data/own folders.")

    total = sum(n for _, n in train_parts)
    # Interleave sources in proportion to their size, so every batch is mixed.
    train = tf.data.Dataset.sample_from_datasets(
        [ds for ds, _ in train_parts], weights=[n / total for _, n in train_parts], seed=cfg.seed
    ).take(total)
    # Tell Keras the epoch length so it shows a real progress bar / ETA.
    train = train.apply(tf.data.experimental.assert_cardinality(total))
    val = val_parts[0]
    for part in val_parts[1:]:
        val = val.concatenate(part)
    val = val.apply(tf.data.experimental.assert_cardinality(sum(bundle.totals("val").values())))

    to_float = lambda x, y: (tf.cast(x, tf.float32), y)  # noqa: E731
    bundle.train = train.batch(cfg.batch_size).map(to_float).prefetch(tf.data.AUTOTUNE)
    bundle.val = val.batch(cfg.batch_size).map(to_float).prefetch(tf.data.AUTOTUNE)
    return bundle
