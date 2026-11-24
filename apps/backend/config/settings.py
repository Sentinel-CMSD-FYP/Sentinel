"""Django settings for the Sentinel backend.

Configuration comes from environment variables (see ``infra/compose.yaml``):
``DJANGO_SECRET_KEY``, ``DJANGO_DEBUG``, ``DATABASE_URL``, ``REDIS_URL``. Without
``DATABASE_URL`` the database falls back to SQLite so the API runs with nothing installed.
"""

import os
from pathlib import Path

import dj_database_url
from django.core.exceptions import ImproperlyConfigured

BASE_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BASE_DIR.parent.parent

DEBUG = os.environ.get("DJANGO_DEBUG", "1") == "1"

SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY", "")
if not SECRET_KEY:
    if not DEBUG:
        raise ImproperlyConfigured("DJANGO_SECRET_KEY must be set when DJANGO_DEBUG is 0")
    SECRET_KEY = "dev-only-insecure-key"

ALLOWED_HOSTS = os.environ.get("DJANGO_ALLOWED_HOSTS", "localhost,127.0.0.1,[::1]").split(",")

INSTALLED_APPS = [
    "django.contrib.contenttypes",
    "rest_framework",
    "corsheaders",
    "apps.clips",
    "apps.pipeline",
]

MIDDLEWARE = [
    "corsheaders.middleware.CorsMiddleware",
    "django.middleware.common.CommonMiddleware",
]

ROOT_URLCONF = "config.urls"
WSGI_APPLICATION = "config.wsgi.application"
ASGI_APPLICATION = "config.asgi.application"
APPEND_SLASH = False

DATABASES = {
    "default": dj_database_url.config(default=f"sqlite:///{BASE_DIR / 'db.sqlite3'}"),
}
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

USE_TZ = True
TIME_ZONE = "UTC"

# The frontend dev server (Vite) runs on another origin and calls this API.
CORS_ALLOWED_ORIGINS = os.environ.get(
    "CORS_ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
).split(",")
CORS_EXPOSE_HEADERS = ["Accept-Ranges", "Content-Range", "Content-Length"]

# No users or sessions yet, so the auth app is not installed: turn off DRF's auth.
REST_FRAMEWORK = {
    "DEFAULT_RENDERER_CLASSES": ["rest_framework.renderers.JSONRenderer"],
    "DEFAULT_AUTHENTICATION_CLASSES": [],
    "DEFAULT_PERMISSION_CLASSES": ["rest_framework.permissions.AllowAny"],
    "UNAUTHENTICATED_USER": None,
}

# Where clips live: one folder per clip holding clip.json, video.mp4 and layers/*.json.
# `python docs/layers/make_sample_clip.py data/samples/demo_fake` makes a fake one.
CLIPS_ROOT = Path(os.environ.get("CLIPS_ROOT", REPO_ROOT / "data" / "samples"))

# Uploads from the browser (POST /api/v1/clips) and the model runs they start (see
# apps/pipeline/jobs.py). The kaggle runner sends each run to a Kaggle GPU kernel and needs the
# kaggle CLI token plus KAGGLE_USERNAME; local runs the models on this machine.
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "1024"))
PIPELINE_RUNNER = os.environ.get("PIPELINE_RUNNER", "kaggle")
PIPELINE_DEVICE = os.environ.get("PIPELINE_DEVICE") or None  # local runner; None = each config
KAGGLE_USERNAME = os.environ.get("KAGGLE_USERNAME", "")
KAGGLE_RUN_TIMEOUT_H = float(os.environ.get("KAGGLE_RUN_TIMEOUT_H", "8"))  # queue + run
MAX_STREAMS = int(os.environ.get("MAX_STREAMS", "4"))  # live streams at once, one process each

# Our own log lines (run_models progress, failing modules) on the console; Django's stay as is.
LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {"plain": {"format": "%(asctime)s %(levelname)s %(name)s: %(message)s"}},
    "handlers": {"console": {"class": "logging.StreamHandler", "formatter": "plain"}},
    "loggers": {
        "apps": {"handlers": ["console"], "level": "INFO"},
        "sentinel_ml": {"handlers": ["console"], "level": "INFO"},
    },
}
