"""Train the severity classifier (transfer learning).

    python -m src.train --config configs/train.yaml
    python -m src.train --epochs 2 --max-batches 5     # quick smoke test
"""
import argparse
import csv
import logging
import sys
import time
from collections import Counter
from pathlib import Path

import yaml

from .utils import load_json, save_json, set_seed, setup_logging

log = logging.getLogger("train")


def class_weights(items, n_classes):
    """Inverse-frequency weights to counter class imbalance."""
    counts = Counter(i["label"] for i in items)
    total = sum(counts.values())
    return [total / (n_classes * max(1, counts.get(c, 0))) for c in range(n_classes)]


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train.yaml")
    for k in ("arch", "splits", "data_root", "checkpoint", "log_dir", "device"):
        ap.add_argument(f"--{k.replace('_', '-')}")
    for k in ("epochs", "batch_size", "patience", "seed", "image_size", "num_workers"):
        ap.add_argument(f"--{k.replace('_', '-')}", type=int)
    for k in ("lr", "weight_decay"):
        ap.add_argument(f"--{k.replace('_', '-')}", type=float)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--max-batches", type=int, default=0, help="debug: limit batches per epoch")
    a = ap.parse_args(argv)
    with open(a.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for k, v in vars(a).items():
        if v is not None and k not in ("config", "no_pretrained", "max_batches"):
            cfg[k] = v
    if a.no_pretrained:
        cfg["pretrained"] = False
    cfg["max_batches"] = a.max_batches
    return cfg


def run_epoch(model, loader, criterion, device, optimizer=None, max_batches=0):
    import torch
    train = optimizer is not None
    model.train(train)
    loss_sum, n, preds, labels = 0.0, 0, [], []
    with torch.set_grad_enabled(train):
        for b, (x, y) in enumerate(loader):
            if max_batches and b >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * len(y)
            n += len(y)
            preds += out.argmax(1).cpu().tolist()
            labels += y.cpu().tolist()
    return loss_sum / max(n, 1), preds, labels


def plot_curves(history, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(ep, [h["train_loss"] for h in history], label="train")
    ax[0].plot(ep, [h["val_loss"] for h in history], label="validation")
    ax[0].set(title="Loss", xlabel="epoch")
    ax[1].plot(ep, [h["train_acc"] for h in history], label="train")
    ax[1].plot(ep, [h["val_acc"] for h in history], label="validation")
    ax[1].set(title="Accuracy", xlabel="epoch")
    for a in ax:
        a.legend()
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None):
    cfg = parse_args(argv)
    setup_logging()
    import torch
    from sklearn.metrics import accuracy_score, f1_score
    from torch.utils.data import DataLoader

    from .dataset import AcneDataset
    from .model import build_model, save_checkpoint

    set_seed(cfg["seed"])
    device = ("cuda" if torch.cuda.is_available() else "cpu") if cfg["device"] == "auto" else cfg["device"]
    log.info("Device: %s", device)

    splits_path = Path(cfg["splits"])
    if not splits_path.exists():
        log.error("splits file not found (%s). Run scripts/prepare_dataset.py first.", splits_path)
        return 1
    splits = load_json(splits_path)
    classes = splits["classes"]
    mk = lambda items, train: AcneDataset(cfg["data_root"], items, cfg["image_size"], train)  # noqa: E731
    g = torch.Generator().manual_seed(cfg["seed"])
    train_dl = DataLoader(mk(splits["train"], True), batch_size=cfg["batch_size"], shuffle=True,
                          num_workers=cfg["num_workers"], generator=g)
    val_dl = DataLoader(mk(splits["val"], False), batch_size=cfg["batch_size"], num_workers=cfg["num_workers"])

    model = build_model(cfg["arch"], len(classes), cfg["pretrained"]).to(device)
    w = torch.tensor(class_weights(splits["train"], len(classes)), dtype=torch.float32, device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=w)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])

    history, best_f1, best_epoch, bad = [], -1.0, 0, 0
    log_dir = Path(cfg["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        tr_loss, tr_p, tr_y = run_epoch(model, train_dl, criterion, device, optimizer, cfg["max_batches"])
        va_loss, va_p, va_y = run_epoch(model, val_dl, criterion, device, None, cfg["max_batches"])
        scheduler.step()
        row = {"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss,
               "train_acc": accuracy_score(tr_y, tr_p), "val_acc": accuracy_score(va_y, va_p),
               "val_macro_f1": f1_score(va_y, va_p, average="macro", zero_division=0)}
        history.append(row)
        log.info("epoch %d/%d  train_loss %.4f  val_loss %.4f  val_acc %.3f  val_macroF1 %.3f  (%.0fs)",
                 epoch, cfg["epochs"], tr_loss, va_loss, row["val_acc"], row["val_macro_f1"], time.time() - t0)
        if row["val_macro_f1"] > best_f1:
            best_f1, best_epoch, bad = row["val_macro_f1"], epoch, 0
            save_checkpoint(cfg["checkpoint"], model, cfg["arch"], classes, cfg["image_size"],
                            {"best_epoch": epoch, "val_macro_f1": float(best_f1), "seed": cfg["seed"],
                             "trained_at": time.strftime("%Y-%m-%d %H:%M:%S")})
        else:
            bad += 1
            if bad >= cfg["patience"]:
                log.info("Early stopping (no improvement for %d epochs).", cfg["patience"])
                break

    with open(log_dir / "training_history.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(history[0]))
        wr.writeheader()
        wr.writerows(history)
    plot_curves(history, log_dir / "training_curves.png")
    ckpt_dir = Path(cfg["checkpoint"]).parent
    save_json({k: v for k, v in cfg.items()}, ckpt_dir / "model_config.json")
    save_json({"classes": classes}, ckpt_dir / "class_mapping.json")
    log.info("Best epoch %d (val macro-F1 %.3f). Checkpoint: %s", best_epoch, best_f1, cfg["checkpoint"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
