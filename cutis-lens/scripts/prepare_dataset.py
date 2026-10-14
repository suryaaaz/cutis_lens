"""Build labels.csv + a reproducible, leakage-safe splits.json from a local ACNE04 folder.

You MUST choose how labels are obtained (we never guess):
  --label-source filename   (use a regex on file names; verify with inspect_dataset.py first)
  --label-source csv --labels-csv my_labels.csv   (columns: filename,label)
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset import (DEFAULT_FILENAME_REGEX, build_label_table, make_splits, scan_dataset,  # noqa: E402
                         split_summary, validate_images)
from src.utils import save_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/ACNE04")
    ap.add_argument("--label-source", required=True, choices=["filename", "csv"])
    ap.add_argument("--labels-csv")
    ap.add_argument("--filename-regex", default=DEFAULT_FILENAME_REGEX)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ratios", type=float, nargs=3, default=[0.7, 0.15, 0.15])
    ap.add_argument("--out", default="outputs")
    a = ap.parse_args()

    if a.label_source == "csv" and not a.labels_csv:
        ap.error("--labels-csv is required with --label-source csv")
    info = scan_dataset(a.data_root)
    print(f"Found {len(info['images'])} images")
    ok, corrupt = validate_images(a.data_root, info["images"])
    print(f"Readable: {len(ok)}  corrupt: {len(corrupt)}")
    rows, unlabeled = build_label_table(a.data_root, ok, a.label_source, a.labels_csv, a.filename_regex)
    print(f"Labelled: {len(rows)}  without label (excluded, not guessed): {len(unlabeled)}")
    if not rows:
        print("[!] No labelled images - check --label-source / regex. Aborting.")
        return 1

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "labels.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label", "label_source", "md5", "n_boxes"])
        w.writeheader()
        w.writerows(rows)
    splits = make_splits(rows, tuple(a.ratios), a.seed)
    splits["label_source"] = a.label_source
    splits["excluded"] = {"corrupt": [c[0] for c in corrupt], "unlabeled": unlabeled}
    save_json(splits, out / "splits.json")
    save_json({"classes": splits["classes"]}, out / "class_mapping.json")
    print("Split sizes:", {k: len(splits[k]) for k in ("train", "val", "test")})
    print("Per-class counts:", split_summary(splits))
    if splits["label_conflicts_excluded"]:
        print(f"[!] {len(splits['label_conflicts_excluded'])} duplicate group(s) had conflicting labels and were excluded.")
    print(f"Saved: {out/'labels.csv'}, {out/'splits.json'}, {out/'class_mapping.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
