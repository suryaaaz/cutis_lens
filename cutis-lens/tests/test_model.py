"""Metrics + training helpers always run. PyTorch tests auto-skip if torch is not installed.
The 'trained model' integration test is skipped until models/best_model.pth exists.
NOTE: tests that build a model use RANDOM weights; they verify mechanics, not accuracy."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.evaluate import compute_metrics
from src.train import class_weights

ROOT = Path(__file__).resolve().parents[1]
try:
    import torch
    import torchvision  # noqa: F401
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class MetricsTests(unittest.TestCase):
    def test_perfect_predictions(self):
        y = [0, 1, 2, 3, 0, 1, 2, 3]
        prob = np.eye(4)[y]
        m = compute_metrics(y, y, prob, ["a", "b", "c", "d"])
        self.assertEqual(m["accuracy"], 1.0)
        self.assertEqual(m["macro_f1"], 1.0)
        self.assertEqual(m["quadratic_weighted_kappa"], 1.0)
        self.assertEqual(m["roc_auc_ovr_macro"], 1.0)
        self.assertEqual(np.trace(np.array(m["confusion_matrix"])), 8)

    def test_known_values(self):
        y_true, y_pred = [0, 0, 1, 1], [0, 1, 1, 1]
        prob = np.array([[.9, .1], [.4, .6], [.2, .8], [.1, .9]])
        m = compute_metrics(y_true, y_pred, prob, ["x", "y"])
        self.assertAlmostEqual(m["accuracy"], 0.75)
        self.assertAlmostEqual(m["per_class"]["x"]["precision"], 1.0)
        self.assertAlmostEqual(m["per_class"]["x"]["recall"], 0.5)
        self.assertAlmostEqual(m["per_class"]["y"]["precision"], 2 / 3)
        self.assertEqual(m["confusion_matrix"], [[1, 1], [0, 2]])

    def test_missing_class_gives_no_auc_not_a_crash(self):
        m = compute_metrics([0, 0, 1], [0, 0, 1], np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0]]), ["a", "b", "c"])
        self.assertIsNone(m["roc_auc_ovr_macro"])
        self.assertEqual(m["per_class"]["c"]["support"], 0)

    def test_confusion_figure(self):
        from src.evaluate import plot_confusion
        with tempfile.TemporaryDirectory() as d:
            plot_confusion([[3, 1], [0, 4]], ["a", "b"], Path(d) / "cm.png")
            self.assertGreater((Path(d) / "cm.png").stat().st_size, 1000)

    def test_class_weights_favour_rare_classes(self):
        w = class_weights([{"label": 0}] * 8 + [{"label": 1}] * 2, 2)
        self.assertGreater(w[1], w[0])
        self.assertAlmostEqual(w[0], 10 / (2 * 8))


@unittest.skipUnless(HAS_TORCH, "PyTorch not installed")
class TorchMechanicsTests(unittest.TestCase):
    def test_forward_shapes_and_target_layer(self):
        from src.model import build_model, get_target_layer
        for arch in ("efficientnet_b0", "mobilenet_v3_large"):
            m = build_model(arch, 4, pretrained=False).eval()
            self.assertEqual(tuple(m(torch.zeros(2, 3, 224, 224)).shape), (2, 4))
            self.assertIsNotNone(get_target_layer(m, arch))
        with self.assertRaises(ValueError):
            build_model("resnet999", 4, False)

    def test_gradcam_output(self):
        from src.explainability import GradCAM
        from src.model import build_model, get_target_layer
        m = build_model("efficientnet_b0", 4, pretrained=False).eval()
        with GradCAM(m, get_target_layer(m, "efficientnet_b0")) as g:
            cam, probs, idx = g(torch.randn(1, 3, 224, 224))
        self.assertEqual(cam.ndim, 2)
        self.assertGreaterEqual(float(cam.min()), 0.0)
        self.assertAlmostEqual(float(probs.sum()), 1.0, places=4)
        self.assertIn(idx, range(4))
        with self.assertRaises(Exception):
            with GradCAM(m, get_target_layer(m, "efficientnet_b0")) as g:
                g(torch.randn(2, 3, 224, 224))

    def test_checkpoint_roundtrip_and_predictor(self):
        from PIL import Image
        from src.model import build_model, load_checkpoint, save_checkpoint
        from src.predict import Predictor
        classes = [{"index": i, "name": f"level_{i}", "display": f"Level {i}", "raw_label": str(i)} for i in range(4)]
        m = build_model("efficientnet_b0", 4, pretrained=False)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "m.pth"
            save_checkpoint(p, m, "efficientnet_b0", classes, 224, {"best_epoch": 1})
            m2, ck = load_checkpoint(p)
            self.assertEqual(ck["classes"], classes)
            x = torch.randn(1, 3, 224, 224)
            torch.testing.assert_close(m.eval()(x), m2(x))
            pred = Predictor(p).load()
            self.assertTrue(pred.is_ready)
            out = pred.predict(Image.new("RGB", (200, 150), (180, 120, 110)))
            self.assertEqual(len(out["probabilities"]), 4)
            self.assertAlmostEqual(sum(i["score"] for i in out["probabilities"]), 1.0, places=4)
            self.assertEqual(out["cam"].ndim, 2)


@unittest.skipUnless(HAS_TORCH and (ROOT / "models" / "best_model.pth").exists(), "no trained model yet")
class TrainedModelIntegration(unittest.TestCase):
    """Runs only after you have trained a model. Checks the real checkpoint works end-to-end."""

    def test_real_checkpoint_predicts(self):
        from PIL import Image
        from src.predict import Predictor
        pred = Predictor(ROOT / "models" / "best_model.pth").load()
        self.assertTrue(pred.is_ready, pred.status)
        out = pred.predict(Image.new("RGB", (256, 256), (190, 140, 130)))
        self.assertAlmostEqual(sum(i["score"] for i in out["probabilities"]), 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
