"""Explainable AI: Grad-CAM (Selvaraju et al., 2017) + heatmap rendering.

A Grad-CAM heatmap shows which image regions most increased the model's score for the
explained class. It indicates model attention/sensitivity - it is NOT proof that a lesion
exists at that location.
"""
import io

import cv2
import numpy as np
from PIL import Image

try:
    import torch
except ImportError:
    torch = None

MAX_DISPLAY_SIDE = 1024


class ExplainabilityError(ValueError):
    pass


# ----------------------------------------------------------------------------- Grad-CAM (torch)
class GradCAM:
    """Grad-CAM on a chosen conv layer. Use as a context manager so hooks are always removed."""

    def __init__(self, model, target_layer):
        if torch is None:
            raise ImportError("PyTorch is required for Grad-CAM.")
        if target_layer is None:
            raise ExplainabilityError("No target layer available for this architecture.")
        self.model, self.layer = model, target_layer
        self._acts = None
        self._grads = None
        self._handle = target_layer.register_forward_hook(self._on_forward)

    def _on_forward(self, module, inputs, output):
        self._acts = output
        if output.requires_grad:
            output.register_hook(lambda g: setattr(self, "_grads", g))

    def close(self):
        self._handle.remove()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __call__(self, x, class_idx=None):
        """x: tensor [1,3,H,W]. Returns (cam float32 [h,w] unnormalised, probs ndarray, class_idx)."""
        if x.ndim != 4 or x.shape[0] != 1:
            raise ExplainabilityError("Grad-CAM expects a single image tensor of shape [1,3,H,W].")
        self.model.zero_grad(set_to_none=True)
        x = x.clone().requires_grad_(True)  # guarantees a graph even if all weights are frozen
        with torch.enable_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
            idx = int(logits.argmax(1)) if class_idx is None else int(class_idx)
            logits[0, idx].backward()
        if self._acts is None or self._grads is None:
            raise ExplainabilityError("Could not capture activations/gradients from the target layer.")
        acts, grads = self._acts[0].detach(), self._grads[0].detach()   # [C,h,w]
        weights = grads.mean(dim=(1, 2), keepdim=True)
        cam = torch.relu((weights * acts).sum(0)).cpu().numpy().astype(np.float32)
        return cam, probs, idx


# ----------------------------------------------------------------------------- rendering (numpy/cv2)
def normalize_cam(cam):
    """Scale to [0,1]. A constant/empty map becomes all zeros (no false attention)."""
    cam = np.asarray(cam, dtype=np.float32)
    if cam.ndim != 2 or cam.size == 0:
        raise ExplainabilityError("The activation map must be a non-empty 2-D array.")
    cam = np.nan_to_num(cam, nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = float(cam.min()), float(cam.max())
    if hi - lo < 1e-8:
        return np.zeros_like(cam)
    return (cam - lo) / (hi - lo)


def _check_rgb(image_rgb):
    a = np.asarray(image_rgb)
    if a.ndim != 3 or a.shape[2] != 3 or a.dtype != np.uint8:
        raise ExplainabilityError("image must be an RGB uint8 array of shape [H,W,3].")
    return a


def fit_display_size(img, max_side=MAX_DISPLAY_SIDE):
    """Downscale a PIL image so the longest side <= max_side (keeps aspect ratio)."""
    w, h = img.size
    s = max_side / max(w, h)
    return img if s >= 1 else img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)


def heatmap_rgb(cam, size_hw):
    """cam -> colour heatmap (JET) at size (H,W), RGB uint8."""
    c = normalize_cam(cam)
    c = cv2.resize(c, (size_hw[1], size_hw[0]), interpolation=cv2.INTER_CUBIC)
    c = np.clip(c, 0, 1)
    bgr = cv2.applyColorMap((c * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), c


def overlay_heatmap(image_rgb, cam, alpha=0.45):
    """Blend heatmap over the image. Low-attention areas stay close to the original."""
    img = _check_rgb(image_rgb)
    heat, c = heatmap_rgb(cam, img.shape[:2])
    w = (alpha * c[..., None]).astype(np.float32)            # weight grows with attention
    out = img.astype(np.float32) * (1 - w) + heat.astype(np.float32) * w
    return np.clip(out, 0, 255).astype(np.uint8)


def side_by_side(image_rgb, overlay_rgb, gap=12):
    a, b = _check_rgb(image_rgb), _check_rgb(overlay_rgb)
    if a.shape != b.shape:
        raise ExplainabilityError("Images must have identical shapes.")
    spacer = np.full((a.shape[0], gap, 3), 255, np.uint8)
    return np.concatenate([a, spacer, b], axis=1)


def to_png_bytes(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def attention_summary(cam, threshold=0.6):
    """Plain facts about the heatmap: where the peak is and how widely attention is spread."""
    c = normalize_cam(cam)
    h, w = c.shape
    if not c.any():
        return {"has_attention": False, "peak_region": None, "high_fraction": 0.0, "spread": "none"}
    y, x = np.unravel_index(int(c.argmax()), c.shape)
    row = ("upper", "middle", "lower")[min(2, int(3 * y / h))]
    col = ("left", "centre", "right")[min(2, int(3 * x / w))]
    region = "centre" if (row, col) == ("middle", "centre") else f"{row}-{col}".replace("middle-", "")
    frac = float((c >= threshold).mean())
    spread = "concentrated" if frac < 0.12 else ("moderately spread" if frac < 0.35 else "widely spread")
    return {"has_attention": True, "peak_region": region, "high_fraction": round(frac, 3), "spread": spread}


def confidence_band(score):
    if score >= 0.75:
        return "relatively high"
    if score >= 0.5:
        return "moderate"
    return "low (the model is uncertain)"


def explain_text(top_display, top_score, runner_up, summary):
    """Short, honest, plain-language explanation of one result."""
    parts = [f"The model's highest-scoring category is {top_display} with a softmax score of "
             f"{top_score:.0%} ({confidence_band(top_score)} confidence)."]
    if runner_up:
        parts.append(f"The next closest category is {runner_up['display']} ({runner_up['score']:.0%}).")
    if summary["has_attention"]:
        parts.append(f"The heatmap's strongest area is in the {summary['peak_region']} part of the image, "
                     f"and the model's attention is {summary['spread']}.")
    else:
        parts.append("The heatmap shows no distinct area of attention for this image.")
    parts.append("Warm colours mark regions that influenced the prediction most. This shows model attention, "
                 "not proof that a lesion is present, and the score is not the probability of a medical condition.")
    return " ".join(parts)


def render_all(original_img, cam, alpha=0.45):
    """Produce the PNG visualisations used by the dashboard and the PDF report."""
    img = fit_display_size(original_img.convert("RGB"))
    rgb = np.asarray(img, dtype=np.uint8)
    overlay = overlay_heatmap(rgb, cam, alpha)
    heat, _ = heatmap_rgb(cam, rgb.shape[:2])
    return {"original": to_png_bytes(rgb), "overlay": to_png_bytes(overlay),
            "heatmap": to_png_bytes(heat), "side_by_side": to_png_bytes(side_by_side(rgb, overlay))}
