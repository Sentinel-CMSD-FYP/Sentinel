# Sentinel backend

Django + DRF. For now it serves recorded clips and their model layers, as described in
`docs/layers/README.md`. Live streaming, uploads and a database come later.

## Run it

```powershell
cd apps/backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt pytest pytest-django ruff
python ../../docs/layers/make_sample_clip.py ../../data/samples/demo_fake   # a fake clip
python manage.py runserver
```

Then open http://127.0.0.1:8000/api/v1/clips.

## Endpoints (`/api/v1`)

| method and path | returns |
|---|---|
| `GET /health` | `{"status": "ok"}` |
| `GET /clips` | every readable clip: id, name, fps, frame size, frame count, layer names |
| `GET /clips/<id>` | the clip's `clip.json` (layer manifest); `video_url` is set only if a video exists |
| `GET /clips/<id>/video` | the video, with byte-range support so players can seek |
| `GET /clips/<id>/layers/<layer>` | the layer file; `?from=&to=` keeps only those frames (inclusive) |

Layers are served as stored, so what a model writes is what the frontend gets. A layer is served only
if `clip.json` declares it. Errors are JSON, `{"detail": "..."}`, with status 404 or 400.

Not built yet: `POST /clips/<id>/layers/<layer>` (ingestion) and the live WebSocket.

## Clips on disk

```
<CLIPS_ROOT>/<clip_id>/clip.json
<CLIPS_ROOT>/<clip_id>/video.mp4
<CLIPS_ROOT>/<clip_id>/layers/<layer>.json
```

`CLIPS_ROOT` defaults to `data/samples/` in the repo (gitignored). Set the `CLIPS_ROOT` environment
variable to use another folder. In Docker the data folder is not mounted yet, so mount one and set
`CLIPS_ROOT` to use real clips.

## Configuration

| variable | default | meaning |
|---|---|---|
| `DJANGO_DEBUG` | `1` | `0` turns debug off and then requires `DJANGO_SECRET_KEY` |
| `DJANGO_SECRET_KEY` | dev-only key in debug | secret key |
| `DJANGO_ALLOWED_HOSTS` | `localhost,127.0.0.1,[::1]` | comma separated |
| `DATABASE_URL` | SQLite file | Postgres URL in CI and Docker; unused until there are models |
| `CORS_ALLOWED_ORIGINS` | the Vite dev server, `http://localhost:5173` | comma separated |
| `CLIPS_ROOT` | `data/samples` | folder holding the clips |

## Tests and checks

CI runs these in `apps/backend`:

```powershell
ruff check .
python manage.py makemigrations --check --dry-run
pytest
```

The tests build a small clip in a temporary folder, so they need no real data.
