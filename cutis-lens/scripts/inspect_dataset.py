"""Inspect a locally available ACNE04 folder BEFORE preparing it.

Usage:  python scripts/inspect_dataset.py --data-root data/ACNE04
It prints the real folder structure, annotation files, sample text-file contents, VOC
box statistics and a preview of what the filename-label pattern would produce.
"""
import argparse
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset import DEFAULT_FILENAME_REGEX, parse_voc_xml, scan_dataset, validate_images  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/ACNE04")
    ap.add_argument("--filename-regex", default=DEFAULT_FILENAME_REGEX)
    ap.add_argument("--skip-validate", action="store_true", help="skip decoding every image")
    a = ap.parse_args()

    root = Path(a.data_root)
    if not root.exists():
        print(f"[!] Dataset folder not found: {root}\n    See data/README.md for download/setup steps.")
        return 1
    info = scan_dataset(root)
    imgs = info["images"]
    print(f"Dataset root : {root.resolve()}")
    print(f"Images found : {len(imgs)}")
    for folder, n in sorted(info["image_folders"].items()):
        print(f"   {n:6d} images in  {folder}")
    print("Annotation-like files:")
    for ext, files in info["annotations"].items():
        print(f"   {ext}: {len(files)} file(s), e.g. {[f.as_posix() for f in files[:3]]}")
    if not info["annotations"]:
        print("   (none found)")

    for ext in (".txt", ".csv", ".json"):
        for f in info["annotations"].get(ext, [])[:2]:
            print(f"\n--- first lines of {f.as_posix()} ---")
            try:
                with open(root / f, encoding="utf-8", errors="replace") as fh:
                    for _ in range(5):
                        print("   " + fh.readline().rstrip()[:150])
            except OSError as e:
                print("   (unreadable:", e, ")")

    xmls = info["annotations"].get(".xml", [])
    if xmls:
        names, boxes = Counter(), []
        for x in xmls[:300]:
            b = parse_voc_xml(root / x)
            boxes.append(len(b))
            names.update(o["name"] for o in b)
        print(f"\nVOC-style xml: parsed {len(boxes)} files; object names: {dict(names)}; "
              f"boxes/image min={min(boxes)} max={max(boxes)} mean={sum(boxes)/len(boxes):.1f}")

    pat = re.compile(a.filename_regex)
    matched = Counter()
    unmatched = []
    for r in imgs:
        m = pat.search(r.name)
        (matched.update([m.group(1)]) if m else unmatched.append(r.name))
    print(f"\nFilename pattern /{a.filename_regex}/ -> label counts: {dict(sorted(matched.items()))}")
    print(f"   unmatched files: {len(unmatched)} {unmatched[:3]}")
    print("   Sample names:", [r.name for r in imgs[:4]])
    print("   >>> VERIFY these labels against the dataset documentation before training. <<<")

    if not a.skip_validate and imgs:
        ok, bad = validate_images(root, imgs)
        print(f"\nReadable images: {len(ok)}   corrupt/unreadable: {len(bad)}")
        for p, why in bad[:10]:
            print(f"   corrupt: {p} ({why})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
