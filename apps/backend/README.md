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

## Run models on a clip (`apps/pipeline`)

```powershell
# from the repo root, in the repo .venv (it has sentinel_ml, CPU torch and ultralytics)
pip install -r apps/backend/requirements.txt pytest-django
cd apps/backend
python manage.py run_models seq011 --modules density,infrastructure [--device cpu] [--replace]
```

It decodes `<CLIPS_ROOT>/<clip_id>/video.mp4` once (at most `n_frames` frames), gives every
frame to every module, and adds each finished layer with `docs/layers/add_layer.py`. That checks
the layer against the contract and the clip and writes `layers/<name>.json` plus its entry in
`clip.json`, so the API above serves it straight away. A layer the clip already has is refused
before any frame is decoded, unless you pass `--replace`. Modules today: `density` (Model A, DM-Count,
grid, every 25th frame) and `infrastructure` (Model C, YOLO26, polygons, every 125th frame).

`sentinel_ml` is imported only by this command, so the API and its Docker image run without
torch. The command's tests (`tests/test_run_models.py`) skip themselves where `sentinel_ml` is
missing, as in CI.

### The module interface (`ml/src/sentinel_ml/pipeline/`)

Every model follows one interface, `sentinel_ml.pipeline.Module`:

```python
class Module(Protocol):
    name: str                                          # layer name -> layers/<name>.json
    def process(self, frame_id: int, frame: np.ndarray) -> None: ...  # every frame, BGR uint8
    def layer(self, clip_id: str) -> dict: ...         # the finished layer (layer contract)
```

`run_clip(frames, modules, clip_id)` feeds `(frame_id, frame)` pairs, in order, to every module
and returns `{name: layer}`. A module decides itself which frames it runs on and keeps any
state it needs (previous frame for flow, tracks for tracking). Frame ids are 0 based.

To add a model (B points, D flow, E tracks):

1. Next to the model, write a class with `name`, `process` and `layer` (see `DensityModule` in
   `ml/src/sentinel_ml/density/layer.py`, about 30 lines).
2. Add a builder to `BUILDERS` in `ml/src/sentinel_ml/pipeline/registry.py` that loads it from
   its config in `ml/configs/` (resolve weight paths with `_from_repo`).
3. `python manage.py run_models <clip> --modules <name>`.

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
