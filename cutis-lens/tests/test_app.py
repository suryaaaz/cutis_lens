import json
import tempfile
import unittest
from pathlib import Path

from app import create_app
from tests.helpers import MockPredictor


class PageTests(unittest.TestCase):
    """Uses MockPredictor: tests the web plumbing only, NOT a trained model."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.app = create_app({"TESTING": True, "SECRET_KEY": "t", "EVAL_SUMMARY_PATH": Path(self.tmp.name) / "none.json"},
                              predictor=MockPredictor())
        self.c = self.app.test_client()

    def tearDown(self):
        self.tmp.cleanup()

    def test_home_ok_and_has_disclaimer(self):
        r = self.c.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn(b"not a medical device", r.data)
        self.assertIn(b"Start analysis", r.data)

    def test_health(self):
        r = self.c.get("/health")
        self.assertEqual(r.status_code, 200)
        j = r.get_json()
        self.assertEqual(j["status"], "ok")
        self.assertTrue(j["model_loaded"])

    def test_analyze_page_has_upload_controls(self):
        r = self.c.get("/analyze")
        self.assertEqual(r.status_code, 200)
        for token in (b'id="dropzone"', b'id="file-input"', b'id="analyze-btn"'):
            self.assertIn(token, r.data)

    def test_about_without_results_shows_no_fake_numbers(self):
        r = self.c.get("/about")
        self.assertEqual(r.status_code, 200)
        self.assertIn(b"No evaluation results exist yet", r.data)

    def test_about_shows_real_summary_when_present(self):
        p = Path(self.tmp.name) / "s.json"
        p.write_text(json.dumps({"n_test": 10, "architecture": "x", "accuracy": 0.5, "macro_precision": 0.4,
                                 "macro_recall": 0.3, "macro_f1": 0.2, "quadratic_weighted_kappa": 0.1,
                                 "roc_auc_ovr_macro": None}))
        self.app.config["EVAL_SUMMARY_PATH"] = p
        r = self.c.get("/about")
        self.assertIn(b"0.500", r.data)
        self.assertNotIn(b"No evaluation results exist yet", r.data)

    def test_404_page_and_headers(self):
        r = self.c.get("/nope")
        self.assertEqual(r.status_code, 404)
        self.assertEqual(r.headers["X-Content-Type-Options"], "nosniff")
        self.assertIn("Content-Security-Policy", r.headers)

    def test_unknown_result_is_404(self):
        self.assertEqual(self.c.get("/results/doesnotexist").status_code, 404)
        self.assertEqual(self.c.get("/results/abc/image/overlay.png").status_code, 404)

    def test_debug_off_by_default(self):
        self.assertFalse(create_app({"TESTING": True}, predictor=MockPredictor()).config["DEBUG"])


if __name__ == "__main__":
    unittest.main()
