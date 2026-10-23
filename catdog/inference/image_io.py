"""Load any image a phone, a browser or a dataset can throw at us.

Handles: 4000x3000 photos, portrait/landscape (EXIF orientation), PNG with
transparency, palette and greyscale images, 16-bit TIFFs, CMYK JPEGs,
animated GIFs (first frame) and, if `pillow-heif` is installed, iPhone HEIC.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from catdog.config import IMAGE_EXTENSIONS

try:  # Optional: iPhone photos are often HEIC.
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:  # pragma: no cover - optional dependency
    pass

# Decoding a 12-megapixel JPEG at full size is slow and pointless when the
# model looks at 224x224. Downscale early, keeping plenty of headroom.
MAX_DECODE_SIDE = 1024

# Guard against decompression bombs while still allowing large phone photos.
Image.MAX_IMAGE_PIXELS = 120_000_000


class ImageLoadError(ValueError):
    """Raised when a file cannot be decoded as an image."""


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS and not path.name.startswith(".")


def list_images(folder: str | Path, recursive: bool = False) -> list[Path]:
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Not a folder: {folder}")
    pattern = "**/*" if recursive else "*"
    return sorted((p for p in folder.glob(pattern) if is_image_file(p)), key=lambda p: str(p).lower())


def load_labelled_folder(root: str | Path) -> list[tuple[Path, int]]:
    """root/{cat,dog,neither}/**/*.jpg -> [(path, label_index)] in LABELS order."""
    from catdog.config import LABELS

    root = Path(root)
    items = []
    for idx, label in enumerate(LABELS):
        if (root / label).is_dir():
            items += [(p, idx) for p in list_images(root / label, recursive=True)]
    return items


def load_image(source: str | Path | bytes, max_side: int = MAX_DECODE_SIDE) -> np.ndarray:
    """Decode `source` (a path or raw bytes) into an RGB uint8 array (H, W, 3)."""
    try:
        fp = io.BytesIO(source) if isinstance(source, (bytes, bytearray)) else open(source, "rb")
        with fp, Image.open(fp) as img:
            # JPEG only: let libjpeg decode at reduced scale (much faster).
            img.draft("RGB", (max_side, max_side))
            img.seek(0)  # first frame of animated GIF/WebP
            img = ImageOps.exif_transpose(img)  # phone photos store rotation in EXIF
            img = _to_rgb(img)
            img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
            return np.asarray(img, dtype=np.uint8).copy()
    except (OSError, SyntaxError, ValueError, Image.DecompressionBombError) as exc:
        name = source if not isinstance(source, (bytes, bytearray)) else "<upload>"
        raise ImageLoadError(f"Could not read image {name}: {exc}") from exc


def _to_rgb(img: Image.Image) -> Image.Image:
    if img.mode == "RGB":
        return img
    if img.mode == "P" and "transparency" in img.info:
        img = img.convert("RGBA")
    if img.mode in ("RGBA", "LA", "PA"):
        # Composite transparency onto white rather than letting it turn black.
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img.convert("RGBA"), mask=img.convert("RGBA").split()[-1])
        return background
    if img.mode in ("I;16", "I;16B", "I;16L", "I"):
        arr = np.asarray(img, dtype=np.float32)
        span = float(arr.max() - arr.min()) or 1.0
        arr = ((arr - arr.min()) / span * 255.0).astype(np.uint8)
        return Image.fromarray(arr).convert("RGB")
    return img.convert("RGB")
