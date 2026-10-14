"""Image preprocessing shared by training, evaluation and inference.

Using ONE implementation everywhere guarantees that the web app feeds the model
exactly what it saw during training. Pure PIL/numpy (no torch needed).
"""
import io
import random

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ImageValidationError(ValueError):
    """Raised for uploads that are not usable images. Message is safe to show users."""


def load_image_safely(data, max_pixels=25_000_000, min_side=64):
    """Decode image bytes -> RGB PIL image, verifying real content.

    - rejects non-image / truncated / corrupt data
    - rejects decompression bombs (too many pixels) and tiny images
    - applies EXIF orientation, then re-encodes pixels (drops metadata)
    """
    if not data:
        raise ImageValidationError("The uploaded file is empty.")
    try:
        probe = Image.open(io.BytesIO(data))
        if probe.format not in {"JPEG", "PNG"}:
            raise ImageValidationError("Only JPG, JPEG and PNG images are supported.")
        w, h = probe.size
        if w * h > max_pixels:
            raise ImageValidationError("The image has too many pixels. Please upload a smaller image.")
        probe.verify()                       # structural integrity check
        img = Image.open(io.BytesIO(data))   # verify() invalidates the object; reopen
        img.load()                           # full decode (catches truncation)
    except ImageValidationError:
        raise
    except Exception:
        raise ImageValidationError("The file is not a valid or readable image.")
    img = ImageOps.exif_transpose(img).convert("RGB")
    if min(img.size) < min_side:
        raise ImageValidationError(f"The image is too small (minimum {min_side}px per side).")
    return img


def resize_for_model(img, size=IMAGE_SIZE):
    return img.convert("RGB").resize((size, size), Image.BICUBIC)


def preprocess_pil(img, size=IMAGE_SIZE, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """PIL image -> float32 array [3, size, size], ImageNet-normalised."""
    arr = np.asarray(resize_for_model(img, size), dtype=np.float32) / 255.0
    arr = (arr - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    return np.ascontiguousarray(arr.transpose(2, 0, 1))


def augment_pil(img, rng=None):
    """Mild training-only augmentation: flip, small rotation/crop, brightness/contrast.

    Intentionally gentle - strong colour changes could alter the appearance of
    redness, which is clinically meaningful for acne.
    """
    rng = rng or random
    img = img.convert("RGB")
    if rng.random() < 0.5:
        img = ImageOps.mirror(img)
    if rng.random() < 0.7:
        img = img.rotate(rng.uniform(-15, 15), resample=Image.BILINEAR, fillcolor=(0, 0, 0))
    if rng.random() < 0.5:
        w, h = img.size
        s = rng.uniform(0.85, 1.0)
        cw, ch = int(w * s), int(h * s)
        x0, y0 = rng.randint(0, w - cw), rng.randint(0, h - ch)
        img = img.crop((x0, y0, x0 + cw, y0 + ch))
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.85, 1.15))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.85, 1.15))
    return img
