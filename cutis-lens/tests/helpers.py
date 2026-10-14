"""Test helpers. IMPORTANT: MockPredictor returns made-up scores. It exists ONLY to test the
web app's plumbing (upload, rendering, PDF). It is never used by the real application."""
import io
import random

import numpy as np
from PIL import Image


def image_bytes(fmt="JPEG", size=(160, 140), color=(200, 150, 140)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format=fmt)
    return buf.getvalue()


def noisy_image(path, size=(96, 96), seed=0):
    rng = np.random.RandomState(seed)
    Image.fromarray(rng.randint(0, 255, (size[1], size[0], 3), dtype=np.uint8)).save(path)


class MockPredictor:
    is_ready = True
    status = "MOCK model (tests only)"
    model_info = {"architecture": "mock", "image_size": 224, "num_classes": 4, "device": "cpu"}

    def predict(self, pil_img):
        cam = np.zeros((7, 7), np.float32)
        cam[1, 1] = 1.0
        cam[1, 2] = cam[2, 1] = 0.6
        scores = [0.10, 0.62, 0.20, 0.08]
        names = ["Level 0 (mild)", "Level 1 (moderate)", "Level 2 (severe)", "Level 3 (very severe)"]
        items = sorted(({"index": i, "name": f"level_{i}", "display": names[i], "score": s}
                        for i, s in enumerate(scores)), key=lambda d: -d["score"])
        return {"probabilities": items, "top": items[0], "cam": cam, "explained_class": items[0]["index"]}


def build_synthetic_dataset(root, per_class=(12, 10, 8, 6), corrupt=1):
    """Tiny fake 'ACNE04-like' tree for pipeline tests (names mimic levleN_k.jpg). NOT real data."""
    jp = root / "JPEGImages"
    jp.mkdir(parents=True, exist_ok=True)
    k = 0
    for level, n in enumerate(per_class):
        for i in range(n):
            noisy_image(jp / f"levle{level}_{i}.jpg", seed=k)
            k += 1
    for j in range(corrupt):
        (jp / f"levle0_bad{j}.jpg").write_bytes(b"not an image")
    ann = root / "Annotations"
    ann.mkdir(exist_ok=True)
    (ann / "levle0_0.xml").write_text(
        "<annotation><object><name>acne</name><bndbox><xmin>1</xmin><ymin>2</ymin><xmax>10</xmax><ymax>12</ymax></bndbox></object></annotation>")
    return root
