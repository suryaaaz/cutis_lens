"""Model factory + checkpoint helpers. torch/torchvision are imported lazily."""
SUPPORTED_ARCHS = ("efficientnet_b0", "mobilenet_v3_large")


def build_model(arch, num_classes, pretrained=True):
    """ImageNet-pretrained backbone with a fresh classification head (transfer learning)."""
    import torch.nn as nn
    from torchvision import models

    if arch == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.efficientnet_b0(weights=weights)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    elif arch == "mobilenet_v3_large":
        weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.mobilenet_v3_large(weights=weights)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    else:
        raise ValueError(f"Unsupported architecture '{arch}'. Choose from {SUPPORTED_ARCHS}.")
    return m


def get_target_layer(model, arch):
    """Last convolutional block - the standard Grad-CAM target for these backbones."""
    if arch in SUPPORTED_ARCHS:
        return model.features[-1]
    return None


def save_checkpoint(path, model, arch, classes, image_size, extra=None):
    """state_dict + primitives only, so it can be loaded with torch.load(weights_only=True)."""
    import torch
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {"state_dict": model.state_dict(), "arch": arch, "num_classes": len(classes),
               "classes": classes, "image_size": int(image_size)}
    payload.update(extra or {})
    torch.save(payload, path)


def load_checkpoint(path, device="cpu"):
    """Rebuild the model from a checkpoint. Returns (model, checkpoint_dict)."""
    import torch
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = build_model(ckpt["arch"], ckpt["num_classes"], pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt
