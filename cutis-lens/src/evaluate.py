"""Evaluate the saved model on the held-out TEST split only.

    python -m src.evaluate
Outputs (in outputs/): evaluation_summary.json, classification_report.txt, confusion_matrix.png
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import (accuracy_score, classification_report, cohen_kappa_score, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score)

from .utils import load_json, save_json, setup_logging


def compute_metrics(y_true, y_pred, y_prob, class_names):
    """Pure function (no torch). y_prob: [n, n_classes] softmax scores."""
    y_true, y_pred, y_prob = np.asarray(y_true), np.asarray(y_pred), np.asarray(y_prob)
    labels = list(range(len(class_names)))
    p, r, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    out = {
        "n_test": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(p.mean()), "macro_recall": float(r.mean()), "macro_f1": float(f1.mean()),
        "quadratic_weighted_kappa": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
        "per_class": {class_names[i]: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f1[i]),
                                       "support": int(support[i])} for i in labels},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "roc_auc_ovr_macro": None,
    }
    if len(np.unique(y_true)) == len(class_names):   # AUC only defined if every class is present
        try:
            out["roc_auc_ovr_macro"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr",
                                                           labels=labels, average="macro"))
        except ValueError:
            pass
    out["report_text"] = classification_report(y_true, y_pred, labels=labels, target_names=class_names,
                                               zero_division=0)
    return out


def plot_confusion(cm, class_names, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(cm, cmap="Purples")
    ax.set(xticks=range(len(class_names)), yticks=range(len(class_names)), xticklabels=class_names,
           yticklabels=class_names, xlabel="Predicted", ylabel="True", title="Confusion matrix (test set)")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="models/best_model.pth")
    ap.add_argument("--splits", default="outputs/splits.json")
    ap.add_argument("--data-root", default="data/ACNE04")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)
    setup_logging()

    import torch
    from torch.utils.data import DataLoader

    from .dataset import AcneDataset
    from .model import load_checkpoint

    model, ckpt = load_checkpoint(a.checkpoint, a.device)
    splits = load_json(a.splits)
    names = [c["name"] for c in ckpt["classes"]]
    dl = DataLoader(AcneDataset(a.data_root, splits["test"], ckpt["image_size"], train=False),
                    batch_size=a.batch_size, num_workers=0)
    ys, ps = [], []
    with torch.no_grad():
        for x, y in dl:
            ps.append(torch.softmax(model(x.to(a.device)), 1).cpu().numpy())
            ys += y.tolist()
    prob = np.concatenate(ps)
    m = compute_metrics(ys, prob.argmax(1), prob, names)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "classification_report.txt").write_text(m.pop("report_text"), encoding="utf-8")
    m["checkpoint"], m["architecture"] = Path(a.checkpoint).name, ckpt["arch"]
    save_json(m, out / "evaluation_summary.json")
    plot_confusion(m["confusion_matrix"], names, out / "confusion_matrix.png")
    print(f"Test images: {m['n_test']}  accuracy: {m['accuracy']:.3f}  macro-F1: {m['macro_f1']:.3f}  "
          f"QWK: {m['quadratic_weighted_kappa']:.3f}")
    print("Saved results to", out.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
