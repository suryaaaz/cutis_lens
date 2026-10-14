import io
import unittest

import numpy as np
from PIL import Image

from src import explainability as xai


class RenderingTests(unittest.TestCase):
    def setUp(self):
        self.rgb = np.random.RandomState(0).randint(0, 255, (80, 120, 3), dtype=np.uint8)

    def test_normalize_range_and_constant(self):
        n = xai.normalize_cam(np.array([[1.0, 2.0], [3.0, 5.0]]))
        self.assertEqual((n.min(), n.max()), (0.0, 1.0))
        self.assertFalse(xai.normalize_cam(np.ones((4, 4))).any())      # constant -> no fake attention
        self.assertFalse(xai.normalize_cam(np.full((3, 3), np.nan)).any())

    def test_invalid_inputs_raise(self):
        with self.assertRaises(xai.ExplainabilityError):
            xai.normalize_cam(np.zeros(5))
        with self.assertRaises(xai.ExplainabilityError):
            xai.normalize_cam(np.zeros((0, 0)))
        with self.assertRaises(xai.ExplainabilityError):
            xai.overlay_heatmap(self.rgb.astype(np.float32), np.ones((7, 7)))
        with self.assertRaises(xai.ExplainabilityError):
            xai.overlay_heatmap(np.zeros((10, 10)), np.ones((7, 7)))

    def test_overlay_shape_dtype_and_zero_cam_keeps_original(self):
        cam = np.zeros((7, 7), np.float32)
        cam[2, 3] = 1
        out = xai.overlay_heatmap(self.rgb, cam)
        self.assertEqual(out.shape, self.rgb.shape)
        self.assertEqual(out.dtype, np.uint8)
        self.assertFalse(np.array_equal(out, self.rgb))
        same = xai.overlay_heatmap(self.rgb, np.zeros((7, 7)))
        np.testing.assert_array_equal(same, self.rgb)

    def test_heatmap_and_side_by_side(self):
        h, _ = xai.heatmap_rgb(np.random.rand(7, 7), (80, 120))
        self.assertEqual(h.shape, (80, 120, 3))
        sbs = xai.side_by_side(self.rgb, self.rgb, gap=10)
        self.assertEqual(sbs.shape, (80, 250, 3))
        with self.assertRaises(xai.ExplainabilityError):
            xai.side_by_side(self.rgb, self.rgb[:10])

    def test_attention_summary_location(self):
        cam = np.zeros((9, 9), np.float32)
        cam[0, 0] = 1
        s = xai.attention_summary(cam)
        self.assertTrue(s["has_attention"])
        self.assertEqual(s["peak_region"], "upper-left")
        self.assertEqual(s["spread"], "concentrated")
        self.assertFalse(xai.attention_summary(np.zeros((5, 5)))["has_attention"])

    def test_confidence_band(self):
        self.assertIn("high", xai.confidence_band(0.9))
        self.assertEqual(xai.confidence_band(0.6), "moderate")
        self.assertIn("uncertain", xai.confidence_band(0.3))

    def test_explain_text_is_honest(self):
        s = xai.attention_summary(np.eye(6))
        t = xai.explain_text("Level 1 (moderate)", 0.62, {"display": "Level 2 (severe)", "score": 0.2}, s)
        self.assertIn("not proof", t)
        self.assertIn("not the probability of a medical condition", t)
        self.assertIn("62%", t)

    def test_render_all_returns_valid_pngs_and_downsizes(self):
        big = Image.fromarray(np.zeros((2000, 3000, 3), np.uint8))
        out = xai.render_all(big, np.random.rand(7, 7))
        self.assertEqual(set(out), {"original", "overlay", "heatmap", "side_by_side"})
        w, h = Image.open(io.BytesIO(out["original"])).size
        self.assertLessEqual(max(w, h), xai.MAX_DISPLAY_SIDE)
        for v in out.values():
            self.assertTrue(v.startswith(b"\x89PNG"))


if __name__ == "__main__":
    unittest.main()
