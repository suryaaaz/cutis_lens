"""Cutis Lens - Flask application.

Run:  python app.py        (development)      |  gunicorn "app:create_app()"  (Linux hosting)
Uploaded images are processed in memory only and expire automatically; nothing is written to disk.
"""
import logging
import secrets
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

from flask import (Flask, Response, abort, jsonify, redirect, render_template, request, session, url_for)
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge
from werkzeug.utils import secure_filename
from PIL import Image

from config import load_config
from src import explainability as xai
from src.predict import Predictor
from src.preprocessing import ImageValidationError, load_image_safely
from src.report import DISCLAIMER, LIMITATIONS, build_report
from src.utils import load_json

log = logging.getLogger("cutis_lens")
IMAGE_KINDS = {"original", "overlay", "heatmap", "side_by_side"}


class ResultStore:
    """Thread-safe in-memory store with TTL + size cap. Entries are bound to a browser session."""

    def __init__(self, ttl, max_items):
        self.ttl, self.max_items = ttl, max_items
        self._d, self._lock = {}, threading.Lock()

    def _purge(self):
        now = time.time()
        for k in [k for k, v in self._d.items() if now - v["_t"] > self.ttl]:
            del self._d[k]
        while len(self._d) > self.max_items:
            del self._d[min(self._d, key=lambda k: self._d[k]["_t"])]

    def put(self, owner, data):
        aid = uuid.uuid4().hex
        with self._lock:
            self._purge()
            self._d[aid] = {**data, "_owner": owner, "_t": time.time(), "analysis_id": aid}
        return aid

    def get(self, aid, owner):
        with self._lock:
            self._purge()
            item = self._d.get(aid)
            return item if item and secrets.compare_digest(item["_owner"], owner or "") else None

    def delete_owner(self, owner):
        with self._lock:
            for k in [k for k, v in self._d.items() if v["_owner"] == owner]:
                del self._d[k]


def create_app(overrides=None, predictor=None):
    app = Flask(__name__)
    app.config.update(load_config())
    if overrides:
        app.config.update(overrides)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if app.config.get("SECRET_KEY_WAS_GENERATED"):
        log.warning("SECRET_KEY not set - using a random key (sessions reset on restart). Set it in .env.")
    Image.MAX_IMAGE_PIXELS = app.config["MAX_IMAGE_PIXELS"]

    app.predictor = predictor if predictor is not None else Predictor(app.config["MODEL_PATH"], app.config["DEVICE"]).load()
    app.store = ResultStore(app.config["RESULT_TTL_SECONDS"], app.config["MAX_STORED_RESULTS"])

    # ------------------------------------------------------------------ helpers
    def owner():
        if "cid" not in session:
            session["cid"] = secrets.token_hex(16)
        return session["cid"]

    def wants_json():
        return request.path.startswith("/api/")

    def fail(message, status, code):
        if wants_json():
            return jsonify({"error": message, "code": code}), status
        return render_template("error.html", status=status, message=message), status

    @app.after_request
    def security_headers(resp):
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["X-Frame-Options"] = "DENY"
        resp.headers["Referrer-Policy"] = "no-referrer"
        resp.headers["Cache-Control"] = "no-store" if request.path.startswith(("/api/", "/results")) else resp.headers.get("Cache-Control", "no-cache")
        resp.headers["Content-Security-Policy"] = ("default-src 'self'; img-src 'self' data: blob:; "
                                                   "style-src 'self' https://cdn.jsdelivr.net; "
                                                   "script-src 'self' https://cdn.jsdelivr.net; "
                                                   "font-src 'self' https://cdn.jsdelivr.net")
        return resp

    @app.context_processor
    def inject():
        return {"model_ready": app.predictor.is_ready, "model_status": app.predictor.status,
                "max_upload_mb": app.config["MAX_UPLOAD_MB"], "disclaimer": DISCLAIMER,
                "allowed_ext": sorted(app.config["ALLOWED_EXTENSIONS"])}

    # ------------------------------------------------------------------ pages
    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/analyze")
    def analyze_page():
        return render_template("analyze.html")

    @app.get("/about")
    def about():
        summary = None
        p = Path(app.config["EVAL_SUMMARY_PATH"])
        if p.exists():
            try:
                summary = load_json(p)
            except Exception:  # noqa: BLE001
                log.exception("Could not read evaluation summary")
        return render_template("about.html", summary=summary, model_info=app.predictor.model_info if app.predictor.is_ready else None,
                               limitations=LIMITATIONS)

    @app.get("/results/<aid>")
    def results(aid):
        item = app.store.get(aid, session.get("cid"))
        if not item:
            return fail("This result has expired or does not exist. Please analyse the image again.", 404, "not_found")
        return render_template("results.html", r=item)

    @app.get("/results/<aid>/image/<kind>.png")
    def result_image(aid, kind):
        item = app.store.get(aid, session.get("cid"))
        if not item or kind not in IMAGE_KINDS:
            abort(404)
        return Response(item["images"][kind], mimetype="image/png")

    @app.post("/results/<aid>/delete")
    def delete_result(aid):
        if app.store.get(aid, session.get("cid")):
            app.store.delete_owner(session["cid"])
        return redirect(url_for("analyze_page"))

    # ------------------------------------------------------------------ API
    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "model_loaded": app.predictor.is_ready,
                        "model_message": app.predictor.status, "time": datetime.now().isoformat(timespec="seconds")})

    @app.post("/api/analyze")
    def api_analyze():
        if not app.predictor.is_ready:
            return fail(app.predictor.status, 503, "model_unavailable")
        f = request.files.get("image")
        if f is None or not f.filename:
            return fail("No image was uploaded. Choose a JPG or PNG file.", 400, "no_file")
        name = secure_filename(f.filename) or "upload"
        if name.rsplit(".", 1)[-1].lower() not in app.config["ALLOWED_EXTENSIONS"] or "." not in name:
            return fail("Unsupported file type. Please upload a JPG, JPEG or PNG image.", 415, "bad_extension")
        try:
            img = load_image_safely(f.read(), app.config["MAX_IMAGE_PIXELS"], app.config["MIN_IMAGE_SIDE"])
        except ImageValidationError as e:
            return fail(str(e), 422, "invalid_image")
        try:
            pred = app.predictor.predict(img)
            images = xai.render_all(img, pred["cam"])
            summary = xai.attention_summary(pred["cam"])
        except xai.ExplainabilityError:
            log.exception("Explainability failure")
            return fail("The explanation could not be generated for this image.", 500, "xai_error")
        except Exception:  # noqa: BLE001
            log.exception("Inference failure")
            return fail("The analysis failed unexpectedly. Please try another image.", 500, "inference_error")

        probs = pred["probabilities"]
        explanation = xai.explain_text(pred["top"]["display"], pred["top"]["score"],
                                       probs[1] if len(probs) > 1 else None, summary)
        aid = app.store.put(owner(), {
            "filename": name, "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "predictions": probs, "top": pred["top"], "explanation": explanation, "attention": summary,
            "confidence_band": xai.confidence_band(pred["top"]["score"]),
            "model_info": app.predictor.model_info, "images": images})
        return jsonify({"analysis_id": aid, "results_url": url_for("results", aid=aid), "predictions": probs,
                        "top": pred["top"], "attention": summary, "explanation": explanation}), 200

    @app.post("/api/report")
    def api_report():
        payload = request.get_json(silent=True) or request.form
        aid = (payload.get("analysis_id") or "").strip() if payload else ""
        if not aid:
            return fail("analysis_id is required.", 400, "missing_id")
        item = app.store.get(aid, session.get("cid"))
        if not item:
            return fail("Analysis not found or expired.", 404, "not_found")
        try:
            pdf = build_report(item)
        except Exception:  # noqa: BLE001
            log.exception("Report failure")
            return fail("The report could not be generated.", 500, "report_error")
        return Response(pdf, mimetype="application/pdf",
                        headers={"Content-Disposition": f"attachment; filename=cutis-lens-report-{aid[:8]}.pdf"})

    # ------------------------------------------------------------------ errors
    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_):
        return fail(f"The file is too large. Maximum size is {app.config['MAX_UPLOAD_MB']} MB.", 413, "too_large")

    @app.errorhandler(HTTPException)
    def http_error(e):
        return fail(e.description if e.code != 404 else "The page you requested was not found.", e.code, "http_error")

    @app.errorhandler(Exception)
    def unexpected(e):
        log.exception("Unhandled error")
        return fail("Something went wrong on the server.", 500, "server_error")

    return app


if __name__ == "__main__":
    application = create_app()
    application.run(host="127.0.0.1", port=5000, debug=application.config["DEBUG"])
