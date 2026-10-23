"""Every place an image can come from, normalised to (image, label) pairs.

Each source returns a tf.data.Dataset of (uint8 HxWx3 image, int32 label)
with labels from catdog.config.LABELS. Images are shuffled *before*
decoding (cheap: compressed bytes) so that a `limit` takes a varied sample
rather than the first classes on disk.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import numpy as np
import tensorflow as tf

from catdog.config import CAT, DOG, NEITHER, OWN_PHOTOS_DIR
from catdog.data.exclusions import CALTECH101_EXCLUDE, IMAGENETTE_EXCLUDE
from catdog.inference.image_io import ImageLoadError, load_image, load_labelled_folder

SHUFFLE_BUFFER = 20_000  # of *encoded* images, a few hundred MB at most


@dataclass(frozen=True)
class SourceInfo:
    name: str
    role: str  # "positive" (cat/dog) or "negative" (neither) or "mixed"
    description: str
    default_limit: int | None = None


SOURCES: dict[str, SourceInfo] = {
    "oxford_iiit_pet": SourceInfo("oxford_iiit_pet", "positive", "Oxford-IIIT Pet: 7,349 photos, 37 breeds, species label used."),
    "caltech101": SourceInfo("caltech101", "negative", "Caltech-101 everyday objects, minus animal classes that are or look like pets.", 3000),
    "imagenette": SourceInfo("imagenette", "negative", "Imagenette (ImageNet subset): real-world scenes, cluttered backgrounds; dog class removed.", 3000),
    "own": SourceInfo("own", "mixed", "Your own phone photos in data/own/{cat,dog,neither}/."),
}


def load_source(name: str, seed: int, limit: int | None = None, data_dir: str | None = None) -> tf.data.Dataset:
    if name not in _LOADERS:
        raise KeyError(f"Unknown source {name!r}. Choose from {sorted(_LOADERS)}")
    limit = limit if limit is not None else SOURCES[name].default_limit
    return _LOADERS[name](seed=seed, limit=limit, data_dir=data_dir)


# --------------------------------------------------------------------- tfds
def _tfds(dataset: str, split: str, seed: int, data_dir: str | None):
    import tensorflow_datasets as tfds

    return tfds.load(
        dataset,
        split=split,
        with_info=True,
        shuffle_files=False,  # determinism: we shuffle ourselves, with a seed
        decoders={"image": tfds.decode.SkipDecoding()},
        data_dir=data_dir,
    )


def _finish(ds: tf.data.Dataset, seed: int, limit: int | None, to_pair: Callable) -> tf.data.Dataset:
    ds = ds.shuffle(SHUFFLE_BUFFER, seed=seed, reshuffle_each_iteration=False)
    if limit:
        ds = ds.take(limit)
    return ds.map(to_pair, num_parallel_calls=tf.data.AUTOTUNE)


def _decode(encoded: tf.Tensor) -> tf.Tensor:
    return tf.io.decode_image(encoded, channels=3, expand_animations=False)


def _oxford_pets(seed: int, limit: int | None, data_dir: str | None) -> tf.data.Dataset:
    ds, info = _tfds("oxford_iiit_pet", "train+test", seed, data_dir)
    species = [n.lower() for n in info.features["species"].names]
    assert species == ["cat", "dog"] and (CAT, DOG) == (0, 1), f"Unexpected species order {species}"
    return _finish(ds, seed, limit, lambda ex: (_decode(ex["image"]), tf.cast(ex["species"], tf.int32)))


def _excluded_ids(names: list[str], exclude: dict[str, str], dataset: str) -> list[int]:
    lowered = [n.lower() for n in names]
    missing = [e for e in exclude if e.lower() not in lowered]
    if missing:
        raise ValueError(f"{dataset}: exclusion(s) {missing} not in class list — typo? Classes: {names}")
    return [lowered.index(e.lower()) for e in exclude]


def _negatives(dataset: str, split: str, exclude: dict[str, str], seed: int, limit: int | None, data_dir: str | None):
    ds, info = _tfds(dataset, split, seed, data_dir)
    banned = tf.constant(_excluded_ids(info.features["label"].names, exclude, dataset), dtype=tf.int64)
    ds = ds.filter(lambda ex: tf.reduce_all(tf.not_equal(tf.cast(ex["label"], tf.int64), banned)))
    return _finish(ds, seed, limit, lambda ex: (_decode(ex["image"]), tf.constant(NEITHER, tf.int32)))


def _caltech(seed: int, limit: int | None, data_dir: str | None) -> tf.data.Dataset:
    return _negatives("caltech101", "train+test", CALTECH101_EXCLUDE, seed, limit, data_dir)


def _imagenette(seed: int, limit: int | None, data_dir: str | None) -> tf.data.Dataset:
    return _negatives("imagenette/320px-v2", "train+validation", IMAGENETTE_EXCLUDE, seed, limit, data_dir)


# ---------------------------------------------------------------- own photos
def _own(seed: int, limit: int | None, data_dir: str | None) -> tf.data.Dataset:
    items = load_labelled_folder(OWN_PHOTOS_DIR)
    rng = np.random.default_rng(seed)
    rng.shuffle(items)
    items = items[:limit] if limit else items

    def gen():
        for path, label in items:
            try:
                # Same loader as inference: EXIF rotation, HEIC, huge photos...
                yield load_image(path), label
            except ImageLoadError as exc:
                print(f"[own] skipping {path}: {exc}")

    return tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(None, None, 3), dtype=tf.uint8),
            tf.TensorSpec(shape=(), dtype=tf.int32),
        ),
    )


_LOADERS: dict[str, Callable[..., tf.data.Dataset]] = {
    "oxford_iiit_pet": _oxford_pets,
    "caltech101": _caltech,
    "imagenette": _imagenette,
    "own": _own,
}
