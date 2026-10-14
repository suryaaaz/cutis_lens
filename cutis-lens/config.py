"""Central configuration. Everything is read from environment variables at call time."""
import os
import secrets
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def _env_bool(name, default=False):
    return os.environ.get(name, str(int(default))).strip().lower() in {"1", "true", "yes", "on"}


def _path(name, default):
    p = Path(os.environ.get(name, str(default)))
    return p if p.is_absolute() else BASE_DIR / p


def load_config():
    """Return a dict of Flask config values."""
    max_mb = int(os.environ.get("MAX_UPLOAD_MB", "8"))
    return {
        "SECRET_KEY": os.environ.get("SECRET_KEY") or secrets.token_hex(32),
        "SECRET_KEY_WAS_GENERATED": not os.environ.get("SECRET_KEY"),
        "MAX_UPLOAD_MB": max_mb,
        "MAX_CONTENT_LENGTH": max_mb * 1024 * 1024,
        "ALLOWED_EXTENSIONS": {"jpg", "jpeg", "png"},
        "MODEL_PATH": _path("MODEL_PATH", BASE_DIR / "models" / "best_model.pth"),
        "EVAL_SUMMARY_PATH": _path("EVAL_SUMMARY_PATH", BASE_DIR / "outputs" / "evaluation_summary.json"),
        "DEVICE": os.environ.get("DEVICE", "cpu"),
        "RESULT_TTL_SECONDS": int(os.environ.get("RESULT_TTL_SECONDS", "1800")),
        "MAX_STORED_RESULTS": int(os.environ.get("MAX_STORED_RESULTS", "200")),
        "MAX_IMAGE_PIXELS": int(os.environ.get("MAX_IMAGE_PIXELS", str(25_000_000))),
        "MIN_IMAGE_SIDE": 64,
        "DEBUG": _env_bool("FLASK_DEBUG", False),
        "SESSION_COOKIE_HTTPONLY": True,
        "SESSION_COOKIE_SAMESITE": "Lax",
        "SESSION_COOKIE_SECURE": _env_bool("SESSION_COOKIE_SECURE", False),
    }
