import io

import numpy as np
import pytest
from PIL import Image

from catdog.inference.image_io import MAX_DECODE_SIDE, ImageLoadError, list_images, load_image
from tests.conftest import jpeg_bytes


def test_huge_photo_is_downscaled():
    arr = load_image(jpeg_bytes(4000, 3000))
    assert arr.dtype == np.uint8 and arr.shape[2] == 3
    assert max(arr.shape[:2]) <= MAX_DECODE_SIDE


def test_transparent_png_composited_on_white():
    arr = load_image(jpeg_bytes(20, 20, (0, 0, 0, 0), fmt="PNG", mode="RGBA"))
    assert arr.shape == (20, 20, 3)
    assert arr.min() == 255  # fully transparent → white, not black


@pytest.mark.parametrize("mode,color", [("L", 90), ("P", 3), ("CMYK", (0, 0, 0, 0)), ("I;16", 4000)])
def test_odd_modes_become_rgb(mode, color):
    fmt = "TIFF" if mode in ("CMYK", "I;16") else "PNG"
    arr = load_image(jpeg_bytes(16, 12, color, fmt=fmt, mode=mode))
    assert arr.shape == (12, 16, 3)


def test_exif_orientation_is_applied():
    img = Image.new("RGB", (60, 30), (255, 0, 0))
    exif = img.getexif()
    exif[0x0112] = 6  # "rotate 90° CW" — how phones store portrait shots
    buf = io.BytesIO()
    img.save(buf, "JPEG", exif=exif)
    assert load_image(buf.getvalue()).shape[:2] == (60, 30)


def test_corrupt_file_raises_clean_error(tmp_path):
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"\x00\x01garbage")
    with pytest.raises(ImageLoadError):
        load_image(bad)


def test_list_images_filters_and_sorts(image_folder):
    names = [p.name for p in list_images(image_folder)]
    assert names == ["a.jpg", "b.png", "broken.jpg", "c_gray.jpg"]  # no .txt, no dotfiles
