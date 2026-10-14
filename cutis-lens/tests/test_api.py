import io
import tempfile
import unittest
from pathlib import Path

from app import create_app
from src.predict import Predictor
from tests.helpers import MockPredictor, image_bytes


def upload(client, data, name="face.jpg", url="/api/analyze"):
    return client.post(url, data={"image": (io.BytesIO(data), name)}, content_type="multipart/form-data")


class ApiWithMockModel(unittest.TestCase):
    """MOCK-model tests: verify upload handling, response format, PDF - not model accuracy."""

    def setUp(self):
        self.app = create_app({"TESTING": True, "SECRET_KEY": "t", "MAX_CONTENT_LENGTH": 300 * 1024},
                              predictor=MockPredictor())
        self.c = self.app.test_client()

    def test_valid_upload_response_format(self):
        r = upload(self.c, image_bytes("JPEG"))
        self.assertEqual(r.status_code, 200)
        j = r.get_json()
        for key in ("analysis_id", "results_url", "predictions", "top", "attention", "explanation"):
            self.assertIn(key, j)
        self.assertEqual(len(j["predictions"]), 4)
        self.assertAlmostEqual(sum(p["score"] for p in j["predictions"]), 1.0, places=5)
        self.assertEqual(j["top"], j["predictions"][0])

    def test_png_upload_ok_and_results_page_renders(self):
        j = upload(self.c, image_bytes("PNG"), "a.png").get_json()
        page = self.c.get(j["results_url"])
        self.assertEqual(page.status_code, 200)
        self.assertIn(b"Grad-CAM", page.data)
        self.assertIn(b"Download PDF report", page.data)
        for kind in ("original", "overlay", "heatmap", "side_by_side"):
            img = self.c.get(f"/results/{j['analysis_id']}/image/{kind}.png")
            self.assertEqual(img.status_code, 200)
            self.assertTrue(img.data.startswith(b"\x89PNG"))

    def test_unsupported_extension(self):
        r = upload(self.c, image_bytes("JPEG"), "x.gif")
        self.assertEqual(r.status_code, 415)
        self.assertEqual(r.get_json()["code"], "bad_extension")

    def test_no_extension(self):
        self.assertEqual(upload(self.c, image_bytes("JPEG"), "noext").status_code, 415)

    def test_corrupted_image(self):
        r = upload(self.c, b"definitely not an image", "x.jpg")
        self.assertEqual(r.status_code, 422)
        self.assertEqual(r.get_json()["code"], "invalid_image")

    def test_truncated_image(self):
        r = upload(self.c, image_bytes("JPEG")[:200], "x.jpg")
        self.assertEqual(r.status_code, 422)

    def test_wrong_format_with_allowed_extension_rejected(self):
        r = upload(self.c, image_bytes("GIF"), "x.png")
        self.assertEqual(r.status_code, 422)

    def test_too_small_image(self):
        self.assertEqual(upload(self.c, image_bytes("PNG", (20, 20)), "t.png").status_code, 422)

    def test_empty_upload(self):
        self.assertEqual(self.c.post("/api/analyze", data={}).status_code, 400)
        self.assertEqual(upload(self.c, b"", "x.jpg").status_code, 422)

    def test_oversized_upload(self):
        r = upload(self.c, b"\xff\xd8" + b"0" * (400 * 1024), "big.jpg")
        self.assertEqual(r.status_code, 413)
        self.assertEqual(r.get_json()["code"], "too_large")

    def test_path_traversal_filename_is_sanitised(self):
        j = upload(self.c, image_bytes("JPEG"), "../../etc/passwd.jpg").get_json()
        page = self.c.get(j["results_url"]).data.decode()
        self.assertNotIn("..", page.split("<main")[1].split("</main>")[0].replace("...", ""))

    def test_error_messages_do_not_leak_paths(self):
        r = upload(self.c, b"junk", "x.jpg")
        self.assertNotIn("/home", r.get_data(as_text=True))
        self.assertNotIn("Traceback", r.get_data(as_text=True))

    def test_report_generation_is_real_pdf(self):
        j = upload(self.c, image_bytes("JPEG"), "me.jpg").get_json()
        r = self.c.post("/api/report", json={"analysis_id": j["analysis_id"]})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.mimetype, "application/pdf")
        self.assertTrue(r.data.startswith(b"%PDF"))
        self.assertGreater(len(r.data), 5000)
        r2 = self.c.post("/api/report", data={"analysis_id": j["analysis_id"]})   # HTML form path
        self.assertEqual(r2.status_code, 200)

    def test_invalid_report_requests(self):
        self.assertEqual(self.c.post("/api/report", json={}).status_code, 400)
        self.assertEqual(self.c.post("/api/report", data="garbage").status_code, 400)
        self.assertEqual(self.c.post("/api/report", json={"analysis_id": "nope"}).status_code, 404)
        self.assertEqual(self.c.get("/api/report").status_code, 405)

    def test_other_browser_cannot_read_or_export_my_result(self):
        j = upload(self.c, image_bytes("JPEG")).get_json()
        other = self.app.test_client()
        self.assertEqual(other.get(j["results_url"]).status_code, 404)
        self.assertEqual(other.get(f"/results/{j['analysis_id']}/image/original.png").status_code, 404)
        self.assertEqual(other.post("/api/report", json={"analysis_id": j["analysis_id"]}).status_code, 404)

    def test_delete_removes_result(self):
        j = upload(self.c, image_bytes("JPEG")).get_json()
        self.c.post(f"/results/{j['analysis_id']}/delete")
        self.assertEqual(self.c.get(j["results_url"]).status_code, 404)

    def test_result_expires(self):
        self.app.store.ttl = -1
        j = upload(self.c, image_bytes("JPEG")).get_json()
        self.assertEqual(self.c.get(j["results_url"]).status_code, 404)


class ApiWithoutModel(unittest.TestCase):
    """Real Predictor class with a checkpoint path that does not exist."""

    def test_missing_checkpoint_is_graceful(self):
        with tempfile.TemporaryDirectory() as d:
            pred = Predictor(Path(d) / "missing.pth").load()
            self.assertFalse(pred.is_ready)
            self.assertIn("No trained model", pred.status)
            app = create_app({"TESTING": True, "SECRET_KEY": "t"}, predictor=pred)
            c = app.test_client()
            r = upload(c, image_bytes("JPEG"))
            self.assertEqual(r.status_code, 503)
            self.assertEqual(r.get_json()["code"], "model_unavailable")
            self.assertNotIn(d, r.get_data(as_text=True))          # no filesystem path leaked
            self.assertFalse(c.get("/health").get_json()["model_loaded"])
            self.assertIn(b"No trained model is loaded", c.get("/analyze").data)

    def test_corrupt_checkpoint_is_graceful(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("PyTorch not installed")
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "bad.pth"
            p.write_bytes(b"garbage")
            pred = Predictor(p).load()
            self.assertFalse(pred.is_ready)


if __name__ == "__main__":
    unittest.main()
