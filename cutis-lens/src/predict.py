"""Inference: loads the trained checkpoint ONCE and returns scores + Grad-CAM for a PIL image."""
import logging
import threading
from pathlib import Path

import numpy as np

from .preprocessing import preprocess_pil

log = logging.getLogger(__name__)


class Predictor:
    """Wraps model loading. `is_ready` is False (with a human-readable `status`) when the
    checkpoint or PyTorch is missing - the web app then refuses to analyse instead of faking."""

    def __init__(self, model_path, device="cpu"):
        self.model_path = Path(model_path)
        self.device = device
        self.model = None
        self.arch = None
        self.classes = []
        self.image_size = 224
        self.meta = {}
        self.is_ready = False
        self.status = "Model not loaded."
        self._lock = threading.Lock()   # Grad-CAM hooks are per-call; serialise requests

    def load(self):
        if not self.model_path.exists():
            self.status = ("No trained model found. Train one first: "
                           "python -m src.train  (see README).")
            return self
        try:
            import torch  # noqa: F401
            from .model import load_checkpoint
        except ImportError:
            self.status = "PyTorch is not installed, so the model cannot run (see README)."
            return self
        try:
            self.model, ckpt = load_checkpoint(self.model_path, self.device)
        except Exception:  # noqa: BLE001
            log.exception("Failed to load checkpoint")
            self.status = "The model file could not be loaded (it may be corrupt or incompatible)."
            return self
        self.arch = ckpt["arch"]
        self.classes = ckpt["classes"]
        self.image_size = ckpt.get("image_size", 224)
        self.meta = {k: ckpt[k] for k in ("best_epoch", "val_macro_f1", "seed", "trained_at") if k in ckpt}
        self.is_ready, self.status = True, "Model loaded."
        log.info("Loaded %s (%d classes) from %s", self.arch, len(self.classes), self.model_path.name)
        return self

    @property
    def model_info(self):
        return {"architecture": self.arch, "image_size": self.image_size,
                "num_classes": len(self.classes), "device": self.device, **self.meta}

    def predict(self, pil_img):
        """Returns {'probabilities': [...sorted desc], 'top': {...}, 'cam': ndarray}."""
        if not self.is_ready:
            raise RuntimeError(self.status)
        import torch
        from .explainability import GradCAM
        from .model import get_target_layer

        x = torch.from_numpy(preprocess_pil(pil_img, self.image_size)).unsqueeze(0).to(self.device)
        with self._lock:
            self.model.eval()
            with GradCAM(self.model, get_target_layer(self.model, self.arch)) as cam_fn:
                cam, probs, idx = cam_fn(x)
        probs = np.asarray(probs, dtype=float)
        items = [{"index": c["index"], "name": c["name"], "display": c["display"], "score": float(probs[c["index"]])}
                 for c in self.classes]
        items.sort(key=lambda d: d["score"], reverse=True)
        return {"probabilities": items, "top": items[0], "cam": cam, "explained_class": int(idx)}
