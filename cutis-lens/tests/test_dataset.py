import csv
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src import dataset as ds
from src.preprocessing import augment_pil, preprocess_pil, load_image_safely, ImageValidationError
from tests.helpers import build_synthetic_dataset, image_bytes, noisy_image


class DatasetPipelineTests(unittest.TestCase):
    """Runs on a SYNTHETIC tiny dataset (random noise images) - checks pipeline logic only."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = build_synthetic_dataset(Path(cls.tmp.name) / "ACNE04")
        cls.info = ds.scan_dataset(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_scan_finds_images_and_annotations(self):
        self.assertEqual(len(self.info["images"]), 37)          # 12+10+8+6 + 1 corrupt
        self.assertEqual(len(self.info["annotations"][".xml"]), 1)

    def test_missing_folder(self):
        with self.assertRaises(FileNotFoundError):
            ds.scan_dataset(self.root / "nope")

    def test_voc_parse(self):
        boxes = ds.parse_voc_xml(self.root / "Annotations" / "levle0_0.xml")
        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0]["xmax"], 10.0)
        (self.root / "Annotations" / "broken.xml").write_text("<a><b>")
        self.assertEqual(ds.parse_voc_xml(self.root / "Annotations" / "broken.xml"), [])

    def test_validate_reports_corrupt(self):
        ok, bad = ds.validate_images(self.root, self.info["images"])
        self.assertEqual(len(ok), 36)
        self.assertEqual([Path(p).name for p, _ in bad], ["levle0_bad0.jpg"])

    def _rows(self):
        ok, _ = ds.validate_images(self.root, self.info["images"])
        return ds.build_label_table(self.root, ok, "filename")

    def test_filename_labels_and_boxes(self):
        rows, unlabeled = self._rows()
        self.assertEqual(unlabeled, [])
        counts = {l: sum(r["label"] == l for r in rows) for l in "0123"}
        self.assertEqual(counts, {"0": 12, "1": 10, "2": 8, "3": 6})
        r0 = next(r for r in rows if r["path"].endswith("levle0_0.jpg"))
        self.assertEqual(r0["n_boxes"], 1)

    def test_unmatched_files_are_not_guessed(self):
        extra = self.root / "JPEGImages" / "mystery.jpg"
        noisy_image(extra, seed=999)
        try:
            ok, _ = ds.validate_images(self.root, ds.scan_dataset(self.root)["images"])
            rows, unlabeled = ds.build_label_table(self.root, ok, "filename")
            self.assertIn("JPEGImages/mystery.jpg", unlabeled)
            self.assertFalse(any("mystery" in r["path"] for r in rows))
        finally:
            extra.unlink()

    def test_csv_label_source(self):
        p = Path(self.tmp.name) / "labels.csv"
        with open(p, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["filename", "label"])
            w.writerow(["levle0_1.jpg", "3"])
        ok, _ = ds.validate_images(self.root, self.info["images"])
        rows, unlabeled = ds.build_label_table(self.root, ok, "csv", p)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["label"], "3")
        self.assertEqual(len(unlabeled), len(ok) - 1)
        bad = Path(self.tmp.name) / "bad.csv"
        bad.write_text("a,b\n1,2\n")
        with self.assertRaises(ValueError):
            ds.read_labels_csv(bad)

    def test_split_properties(self):
        rows, _ = self._rows()
        s = ds.make_splits(rows, seed=1)
        self.assertEqual(sum(len(s[k]) for k in ("train", "val", "test")), len(rows))
        self.assertTrue(ds.check_no_leakage(s))
        for k in ("train", "val", "test"):                       # every class in every split
            self.assertEqual({i["label"] for i in s[k]}, {0, 1, 2, 3})
        self.assertEqual([c["name"] for c in s["classes"]], ["level_0", "level_1", "level_2", "level_3"])

    def test_split_is_reproducible_and_seed_dependent(self):
        rows, _ = self._rows()
        a, b, c = ds.make_splits(rows, seed=7), ds.make_splits(rows, seed=7), ds.make_splits(rows, seed=8)
        self.assertEqual(a["test"], b["test"])
        self.assertNotEqual(a["test"], c["test"])

    def test_duplicates_stay_together_and_conflicts_excluded(self):
        rows, _ = self._rows()
        rows = [dict(r) for r in rows]
        dup = dict(rows[0], path="JPEGImages/copy_of_first.jpg")             # same md5, same label
        conflict_a = dict(rows[5], path="JPEGImages/conf_a.jpg")             # same md5 as rows[5]
        conflict_a["label"] = "3" if rows[5]["label"] != "3" else "0"        # different label
        s = ds.make_splits(rows + [dup, conflict_a], seed=3)
        where = {k: {i["path"] for i in s[k]} for k in ("train", "val", "test")}
        home = [k for k in where if rows[0]["path"] in where[k]][0]
        self.assertIn("JPEGImages/copy_of_first.jpg", where[home])
        self.assertEqual(len(s["label_conflicts_excluded"]), 1)
        self.assertTrue(all(rows[5]["path"] not in w for w in where.values()))

    def test_too_few_per_class_raises(self):
        rows = [{"path": f"{i}.jpg", "label": "0", "md5": str(i)} for i in range(10)] + \
               [{"path": "x.jpg", "label": "1", "md5": "x"}]
        with self.assertRaises(ValueError):
            ds.make_splits(rows)

    def test_bad_ratios(self):
        with self.assertRaises(ValueError):
            ds.make_splits([], ratios=(0.5, 0.5, 0.5))

    def test_leakage_detector_fires(self):
        with self.assertRaises(AssertionError):
            ds.check_no_leakage({"train": [{"path": "a"}], "val": [{"path": "a"}], "test": []})


class PreprocessingTests(unittest.TestCase):
    def test_preprocess_shape_dtype_range(self):
        from PIL import Image
        arr = preprocess_pil(Image.new("RGB", (300, 200), (255, 255, 255)))
        self.assertEqual(arr.shape, (3, 224, 224))
        self.assertEqual(arr.dtype, np.float32)
        self.assertAlmostEqual(float(arr[0, 0, 0]), (1 - 0.485) / 0.229, places=3)

    def test_augment_is_valid_and_deterministic_with_rng(self):
        import random
        from PIL import Image
        img = Image.fromarray(np.random.RandomState(0).randint(0, 255, (64, 64, 3), dtype=np.uint8))
        a = augment_pil(img, random.Random(5))
        b = augment_pil(img, random.Random(5))
        self.assertEqual(a.mode, "RGB")
        self.assertEqual(np.asarray(a).tolist(), np.asarray(b).tolist())

    def test_safe_loader(self):
        self.assertEqual(load_image_safely(image_bytes("PNG")).mode, "RGB")
        with self.assertRaises(ImageValidationError):
            load_image_safely(b"")
        with self.assertRaises(ImageValidationError):
            load_image_safely(image_bytes("PNG", (400, 400)), max_pixels=1000)


if __name__ == "__main__":
    unittest.main()
