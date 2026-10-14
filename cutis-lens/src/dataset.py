"""ACNE04 dataset inspection, label building, leakage-safe splitting, and the PyTorch Dataset.

Design rule: NOTHING here invents labels. Labels come only from (a) a CSV you provide or
(b) a filename pattern you explicitly choose after inspecting the files. Both are logged
in the output so you can verify them.
"""
import csv
import logging
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image
from sklearn.model_selection import train_test_split

from .preprocessing import augment_pil, preprocess_pil
from .utils import file_md5

try:
    import torch
    from torch.utils.data import Dataset as _BaseDataset
except ImportError:  # the web app / tests can import this module without torch
    torch = None
    _BaseDataset = object

log = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
ANNOTATION_EXTS = {".xml", ".json", ".txt", ".csv"}
DEFAULT_FILENAME_REGEX = r"lev\w*?(\d)_"   # candidate only: verify with inspect_dataset.py!
DEFAULT_DISPLAY = {0: "Level 0 (mild)", 1: "Level 1 (moderate)", 2: "Level 2 (severe)", 3: "Level 3 (very severe)"}


# ----------------------------------------------------------------------------- inspection
def scan_dataset(root):
    """Walk the dataset folder. Returns images, annotation files and a folder summary."""
    root = Path(root)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {root}")
    images, annotations, other = [], defaultdict(list), []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        rel = p.relative_to(root)
        if ext in IMAGE_EXTS:
            images.append(rel)
        elif ext in ANNOTATION_EXTS:
            annotations[ext].append(rel)
        else:
            other.append(rel)
    folders = Counter(str(r.parent) for r in images)
    return {"root": str(root), "images": images, "annotations": dict(annotations),
            "other": other, "image_folders": dict(folders)}


def parse_voc_xml(path):
    """Parse a Pascal-VOC style xml -> list of {name, xmin, ymin, xmax, ymax}. [] if not VOC."""
    try:
        tree = ET.parse(path)
    except ET.ParseError:
        return []
    boxes = []
    for obj in tree.getroot().findall("object"):
        bb = obj.find("bndbox")
        if bb is None:
            continue
        try:
            boxes.append({"name": (obj.findtext("name") or "").strip(),
                          **{k: float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")}})
        except (TypeError, ValueError):
            continue
    return boxes


def validate_images(root, rel_paths):
    """Fully decode each image. Returns (ok_paths, corrupt[(path, reason)])."""
    ok, corrupt = [], []
    for rel in rel_paths:
        try:
            with Image.open(Path(root) / rel) as im:
                im.verify()
            with Image.open(Path(root) / rel) as im:
                im.load()
            ok.append(rel)
        except Exception as e:  # noqa: BLE001 - any decode failure means "unusable"
            corrupt.append((str(rel), type(e).__name__))
    return ok, corrupt


# ----------------------------------------------------------------------------- labels
def read_labels_csv(csv_path):
    """CSV with columns: filename,label. `filename` may be a bare name or relative path."""
    out = {}
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not {"filename", "label"} <= set(reader.fieldnames):
            raise ValueError("labels CSV must have the columns: filename,label")
        for row in reader:
            out[row["filename"].strip()] = row["label"].strip()
    return out


def build_label_table(root, rel_images, source, labels_csv=None, filename_regex=DEFAULT_FILENAME_REGEX):
    """Create rows {path, label, label_source, md5, n_boxes}. Images without a label are
    returned separately - they are never guessed."""
    root = Path(root)
    csv_labels = read_labels_csv(labels_csv) if source == "csv" else {}
    if source not in {"csv", "filename"}:
        raise ValueError("label source must be 'csv' or 'filename'")
    pattern = re.compile(filename_regex)

    voc = {}
    for rel in scan_dataset(root)["annotations"].get(".xml", []):
        voc.setdefault(Path(rel).stem, rel)

    rows, unlabeled = [], []
    for rel in rel_images:
        rel = Path(rel)
        label = None
        if source == "csv":
            label = csv_labels.get(rel.name) or csv_labels.get(rel.as_posix())
        else:
            m = pattern.search(rel.name)
            label = m.group(1) if m else None
        if label is None:
            unlabeled.append(rel.as_posix())
            continue
        n_boxes = len(parse_voc_xml(root / voc[rel.stem])) if rel.stem in voc else None
        rows.append({"path": rel.as_posix(), "label": str(label), "label_source": source,
                     "md5": file_md5(root / rel), "n_boxes": n_boxes})
    return rows, unlabeled


# ----------------------------------------------------------------------------- splitting
def make_splits(rows, ratios=(0.7, 0.15, 0.15), seed=42):
    """Stratified train/val/test split that cannot leak exact duplicates.

    Identical files (same md5) form a group that is kept in ONE split. Groups whose
    members carry conflicting labels are excluded and reported, never silently resolved.
    """
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("ratios must sum to 1")
    groups = defaultdict(list)
    for r in rows:
        groups[r["md5"]].append(r)
    keep, conflicts = [], []
    for md5, members in groups.items():
        if len({m["label"] for m in members}) > 1:
            conflicts.append([m["path"] for m in members])
        else:
            keep.append(members)
    labels = [g[0]["label"] for g in keep]
    counts = Counter(labels)
    if min(counts.values()) < 3:
        raise ValueError(f"Each class needs at least 3 unique images to split; counts: {dict(counts)}")

    idx = list(range(len(keep)))
    train_val, test = train_test_split(idx, test_size=ratios[2], stratify=labels, random_state=seed)
    val_rel = ratios[1] / (ratios[0] + ratios[1])
    train, val = train_test_split(train_val, test_size=val_rel, stratify=[labels[i] for i in train_val],
                                  random_state=seed)

    def expand(ids):
        return sorted(({"path": m["path"], "label": m["label"]} for i in ids for m in keep[i]),
                     key=lambda d: d["path"])

    class_values = sorted(counts, key=lambda v: (len(v), v))
    classes = [{"index": i, "name": f"level_{v}" if v.isdigit() else v,
                "display": DEFAULT_DISPLAY.get(int(v), f"Level {v}") if v.isdigit() else v,
                "raw_label": v} for i, v in enumerate(class_values)]
    to_index = {c["raw_label"]: c["index"] for c in classes}
    splits = {"seed": seed, "ratios": list(ratios), "classes": classes,
              "label_conflicts_excluded": conflicts}
    for name, ids in (("train", train), ("val", val), ("test", test)):
        items = expand(ids)
        for it in items:
            it["label"] = to_index[it["label"]]
        splits[name] = items
    check_no_leakage(splits)
    return splits


def check_no_leakage(splits):
    sets = {k: {i["path"] for i in splits[k]} for k in ("train", "val", "test")}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = sets[a] & sets[b]
        if overlap:
            raise AssertionError(f"Data leakage: {len(overlap)} files appear in both {a} and {b}")
    return True


def split_summary(splits):
    names = [c["name"] for c in splits["classes"]]
    return {k: {names[i]: n for i, n in sorted(Counter(it["label"] for it in splits[k]).items())}
            for k in ("train", "val", "test")}


# ----------------------------------------------------------------------------- torch dataset
class AcneDataset(_BaseDataset):
    """Loads images listed in a splits.json section. Training gets mild augmentation;
    val/test images are only resized + normalised (originals untouched on disk)."""

    def __init__(self, root, items, image_size=224, train=False):
        if torch is None:
            raise ImportError("PyTorch is required for AcneDataset (see README installation step).")
        self.root, self.items, self.size, self.train = Path(root), items, image_size, train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        with Image.open(self.root / it["path"]) as im:
            img = im.convert("RGB")
        if self.train:
            img = augment_pil(img)
        return torch.from_numpy(preprocess_pil(img, self.size)), int(it["label"])
