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
| `POST /clips` | multipart `file` (mp4, mov, avi, mkv; at most `MAX_UPLOAD_MB`), optional `name` and optional camera fields (`altitude_m`, `fov_deg`, `fov_kind` horizontal or diagonal, `sensor_aspect`, `camera_model`, `altitude_source`): the video becomes a clip (`ingest_clip`) before the reply, 201 + its `clip.json`; 400 bad file or camera fields, 413 too big |
| `POST /clips/<id>/runs` | starts every model on the clip in the background, 202 + the run status; optional `replace` (JSON or form) to redo existing layers; 409 if a run is going or the clip has layers |
| `GET /clips/<id>/runs/<run_id>` | the run status (below) |
| `GET /runs/active` | `{"run": status}` of the run going now (any clip), `{"run": null}` if none; the upload panel asks it on page load, so a reload during a run keeps showing it |
| `POST /streams` | JSON `clip_id` and optional `modules` (list or `"flow,points"`, default all): replays that stored clip at real time as a live stream; 202 + the stream status; 400 bad modules, 404 unknown clip, 409 at `MAX_STREAMS` |
| `GET /streams` | `{"streams": [...]}`, every stream, newest first |
| `GET /streams/<id>` | the stream status (below) |
| `POST /streams/<id>/stop` | the stream stops reading and is saved as a clip; 202, or 409 if it has ended |
| `GET /streams/<id>/samples?after=<offset>` | `{"samples": [...], "next": offset, "state": ...}`: the samples made since `offset` (0 at first), in order |
| `GET /streams/<id>/video` | the live picture as MJPEG (`multipart/x-mixed-replace`), each part with `X-Frame-Id` and `X-Capture-Time`; ends with the stream |

Layers are served as stored, so what a model writes is what the frontend gets. A layer is served only
if `clip.json` declares it. Errors are JSON, `{"detail": "..."}`, with status 404 or 400.

## Live streams (`apps/pipeline/streams.py`, `ml/src/sentinel_ml/pipeline/live.py`)

A stream runs the models on frames as they arrive and hands out each layer sample at once. For
Tier 1 the source is a video file read at its own fps (a stand in for a drone feed): from the
API a stored clip's `video.mp4` (the request never names a path or URL), from the command line
any file, frame folder, `rtsp://` / `http://` URL or webcam index (URL and webcam untested).

```powershell
# from the repo root; the API needs no Kaggle settings for streams
cd apps/backend; ..\..\.venv\Scripts\python manage.py run_stream ..\..\data\samples\seq011\video.mp4 --modules all
# a folder of frames carries no frame rate: give it (default 30)
cd apps/backend; ..\..\.venv\Scripts\python manage.py run_stream path\to\frames --fps 25
```

* **Worker**: one process per stream (`manage.py run_stream --worker <id>`, started by the API;
  `run_stream <source>` runs in the foreground, Ctrl+C stops it). At most `MAX_STREAMS` at once,
  independent of the one model run at a time of `POST .../runs`. Device: `PIPELINE_DEVICE`.
* **Scheduling**: a reader thread paces the file and records every frame; each module chain
  (a module plus those that `need` it, e.g. points then tracks) has its own thread and queue of
  the frames it `wants`. A frame that waited longer than `MAX_LAG_S` (1 s) is dropped (the
  newest is kept), so a slow model lowers its rate instead of falling further behind. Skipped
  frames get no sample; the gaps rule shows it.
* **Samples** (`samples.jsonl` in `<CLIPS_ROOT>/.streams/<id>/`): one line per sample,
  `{"seq", "layer", "frame", "t_capture", "t_ready", "data"}`, `data` being that frame's value in
  the stored layer format; a layer's first sample also has `header` (the layer without
  `frames`) and `meta` (its clip.json entry as `add_layer` writes it: label, style,
  default_visible), and the status keeps all headers in `layers`. Poll with `?after=<next>`; a line
  still being written waits for the next poll.
* **Status** fields: `state` (`queued`, `running`, `done`, `failed`), `stage`, `fps`,
  `frame_size`, `frames_read`, `samples`, `layers`, `ran` (frames each module processed),
  `skipped` (frames each chain dropped), `late_s` (worst delay of the reader behind the file's
  pace), `errors` per failed module, `error`, `clip_id` once saved. Rewritten every second; a
  running stream silent for 60 s reads as failed.
* **End** (file end or stop): the recording (x264 crf 18, preset veryfast) and every finished
  layer become a normal clip `<CLIPS_ROOT>/<stream_id>` through `add_layer`, with
  `produced_by`, so it replays like any clip. A failing module is dropped as in `run_models`.
* A module used live needs `samples` and `wants(frame_id)` (all five have them).

The MJPEG picture ends only when a status read says the stream has ended (a read that meets the
worker replacing `status.json` on Windows is retried). The frontend's live page is described in
`apps/frontend/README.md`; latency numbers (CPU and Kaggle T4, 1 to 4 streams) are in
`docs/pipeline.md`, "Live latency".

## Upload and run from the browser (`apps/pipeline/views.py`, `jobs.py`, `kaggle_run.py`)

The frontend's "Upload a video and run the models" panel uses the three endpoints above.

* **Clip id**: the file name made safe (letters, digits, `_` and `-`, at most 40 characters; `my drone
  clip.mp4` becomes `my_drone_clip`), with `_2`, `_3` added when taken. The client's file name never
  reaches a path. The display name is the `name` field, else the file name.
* **Ingest** runs inside the upload request (about 40 frames/s on a desktop CPU), so a bad video is
  refused at once and the clip plays before any model has run.
* **Runs**: `POST .../runs` writes `<CLIPS_ROOT>/.runs/<run_id>/status.json` and starts
  `manage.py run_job <run_id>` as a separate process (output in `log.txt` next to it), so a run
  survives the request and a dev server restart. One run at a time on the whole server.
* **Status** fields: `state` (`queued`, `running`, `done`, `failed`), `stage` (what it is doing, e.g.
  "Kaggle: waiting for a GPU"), `progress` (`done`/`total` frames, local runner only), `layers`
  added, `errors` per failed module (the other layers are still added, as with `run_models`), `error`
  (why the whole run failed), times. The worker rewrites it at least every 30 s; a status older than
  5 min reads as failed (the worker was killed) and frees the server for a new run.
* **Runners** (`PIPELINE_RUNNER`):
  * `kaggle` (default, this PC has no CUDA GPU): the clip's video and clip.json go up as a new version
    of the private dataset `<user>/sentinel-run-clip`; once Kaggle says it is ready, the kernel
    `<user>/sentinel-run-models` (`apps/pipeline/kaggle_kernel/`) runs `run_models --modules all
    --device cuda --report` with the trained weights; the backend polls it, downloads its output and
    adds each layer with `add_layer` (checked again) plus its `produced_by`. The kernel log lands in
    `.runs/<run_id>/kaggle.log`. Needs the kaggle CLI token (`~/.kaggle/access_token`) and
    `KAGGLE_USERNAME`. The kernel runs the code in `<user>/sentinel-ml-src`, so after a change to
    `ml/` or the backend run `python eval/density_dmcount/kaggle/push.py src` first.
  * `local`: `run_models` on this machine (`PIPELINE_DEVICE`, default each config's `auto`), for a
    backend on a machine with a CUDA GPU.

Start the backend for uploads (PowerShell, repo root):

```powershell
$env:KAGGLE_USERNAME="ahmar004"; $env:PYTHONUTF8="1"
cd apps/backend; ..\..\.venv\Scripts\python manage.py runserver
```

Timing of a 300 frame 1080p clip through Kaggle: see `progress.md` step 27.

## Run models on a clip (`apps/pipeline`)

```powershell
# from the repo root, in the repo .venv (it has sentinel_ml, CPU torch, ultralytics, imageio-ffmpeg)
pip install -r apps/backend/requirements.txt pytest-django
pip install -e "ml[export]"   # ffmpeg for ingest_clip
cd apps/backend
# 1. make the clip: a mp4 file, or a folder of numbered frames (DroneCrowd)
python manage.py ingest_clip ../../data/dronecrowd/test_data-001/test_data/images seq011 --seq 011 --fps 25 --name "DroneCrowd test 11"
# 2. add the model layers
python manage.py run_models seq011 --modules all [--device cpu] [--replace]
```

`--modules all` runs every registered module (today density, flow, infrastructure, points,
tracks) in one decode; a comma separated list picks some of them. Progress is logged every 100
frames (frames done, frames/s, seconds left). A module that fails (cannot be built, raises on a
frame, or its layer is refused) is dropped with the modules that need it; the other layers are
still added and the command then exits with an error per failed module. The whole flow, the
failure rules and the demo clip rebuild are in `docs/pipeline.md`.

`ingest_clip <source> <clip_id> [--seq] [--fps] [--name]` writes `<CLIPS_ROOT>/<clip_id>/` with
`video.mp4` (frames piped once into ffmpeg: H.264, crf 18, yuv420p, faststart) and a `clip.json`
with no layers. It refuses an existing clip folder, odd frame sizes (H.264 needs even ones) and a
missing ffmpeg, and builds in a temporary folder, so a failed ingest leaves nothing behind. A frame
folder has no frame rate, so pass `--fps` (default 30). crf 18 because the models decode this
video: DM-Count counts on test clip 11 move 2.6% on average against the original jpgs at crf 18,
4.7% at ffmpeg's default crf 23.

Own footage: `--altitude 60 --dfov 82.1 [--sensor-aspect 4:3] [--camera-model "DJI Mini 3"]
[--altitude-source assumed|stated|telemetry]` (or `--hfov`) stores a clip.json `camera` block
(`apps/pipeline/camera.py`); the upload form has the same fields. A diagonal FOV (spec sheets) is
converted with the sensor's shape, default 4:3, not the 16:9 video's (a 16:9 video crops a 4:3
sensor top and bottom, so the horizontal FOV stays the sensor's; using 16:9 makes every metre about
7% too large). `run_models`, the Kaggle runner and live streams pass `camera` to flow, which then
writes `m_per_s`. Check a clip's metres with `eval/pipeline/geometry_check.py` (known lengths,
walking speeds).

On CPU (Ryzen 5 3600) clip 11 (300 frames, 1920x1080) takes 7 s to ingest, 48 s for `density`
(12 samples), 7 s for `infrastructure` (3 samples), 4 s for `flow` (12 samples, no weights) and
5 min 52 s for `points` (300 samples, 1.1 s per frame plus ~20 s to load). `tracks` adds
under 2 s on top of `points` (`--modules tracks` on clip 11: 6 min 53 s for both layers).

It decodes `<CLIPS_ROOT>/<clip_id>/video.mp4` once (at most `n_frames` frames), gives every
frame to every module, and adds each finished layer with `docs/layers/add_layer.py`. That checks
the layer against the contract and the clip and writes `layers/<name>.json` plus its entry in
`clip.json`, so the API above serves it straight away. A layer the clip already has is refused
before any frame is decoded, unless you pass `--replace`. Modules today: `density` (Model A, DM-Count,
grid, every 25th frame), `infrastructure` (Model C, YOLO26, polygons, every 125th frame), `flow`
(Model D, Farneback, grid of u, v, speed, every 25th frame) and `points` (Model B, DOT + Pixel
Distill, points at score >= 0.2, every frame), `tracks` (Model E, Kalman point tracker on B's
points, tracks `[track_id, x, y, score]`). `points` needs `pip install -e "ml[points]"` and the
uav-dot clone plus checkpoint in `ml/third_party/uav-dot` (see `docs/pipeline_models.md`).
`tracks` needs `points`, so asking for `tracks` builds one `points` module too and adds both
layers (B runs once); the clip's `fps` from `clip.json` turns E's 3 s miss limit into frames.

Each layer's `clip.json` entry gets `default_visible` from `SHOWN` in the registry (density,
infrastructure and tracks start on; points and flow start off). After the layers are added,
`clip.json` `produced_by` records per layer: `model` (with fp16/fp32 for A and C), `weights` and
`weights_md5` (first 12 hex) where there are weights, `every`, `device`, `seconds` (that module's
wall time; the shared decode is not counted), `frames` decoded and `date`. Entries of layers the
run did not touch are kept.

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
A module that works on another module's output lists it in `needs` (tracks: `("points",)`)
and holds that module; `build_modules(names, device, fps)` in the registry builds each needed
module once and puts it first, and `run_clip` refuses a list in the wrong order. Both take an
optional `errors` dict: given one, a failing module is noted there and left out instead of
raising (the backend always passes one). Live mode (`sentinel_ml.pipeline.live.run_live`)
also reads `samples` (the samples so far by frame id) and `wants(frame_id)` (whether `process`
works on that frame; frame id only).

To add a model:

1. Next to the model, write a class with `name`, `process`, `layer`, `wants` and `samples` (see
   `DensityModule` in `ml/src/sentinel_ml/density/layer.py`, about 40 lines).
2. Add a builder to `BUILDERS` in `ml/src/sentinel_ml/pipeline/registry.py` that loads it from
   its config in `ml/configs/` (resolve weight paths with `_from_repo`). If it needs another
   module, add it to `NEEDS` (its builder gets that module by name); if it needs the clip's fps,
   add it to `USES_FPS`. Wrap the module with `_about(...)` so clip.json says what made the
   layer, and add its name to `SHOWN` if it should start visible.
3. `python manage.py run_models <clip> --modules <name>`.

## Clips on disk

```
<CLIPS_ROOT>/<clip_id>/clip.json
<CLIPS_ROOT>/<clip_id>/video.mp4
<CLIPS_ROOT>/<clip_id>/layers/<layer>.json
```

The folder name and the `clip_id` inside `clip.json` must match. If you copy a clip folder, edit its
`clip_id` too: a mismatched clip is left out of the list (a warning is logged) and its URLs return 404
with a message saying so, instead of showing another clip's data.

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
| `MAX_UPLOAD_MB` | `1024` | largest video `POST /clips` accepts |
| `PIPELINE_RUNNER` | `kaggle` | where runs from the API go: `kaggle` (GPU kernel) or `local` |
| `PIPELINE_DEVICE` | each config (`auto`) | device for the `local` runner, e.g. `cpu` or `cuda` |
| `KAGGLE_USERNAME` | none | Kaggle account of the `kaggle` runner (the token has no name in it) |
| `KAGGLE_RUN_TIMEOUT_H` | `8` | give up on a Kaggle run after this long (GPU queue included) |
| `MAX_STREAMS` | `4` | live streams running at once |

## Tests and checks

CI runs these in `apps/backend`:

```powershell
ruff check .
python manage.py makemigrations --check --dry-run
pytest
```

The tests build a small clip in a temporary folder, so they need no real data, except
`tests/test_end_to_end.py`: 10 real DroneCrowd frames through ingest, every registered model
(real weights, CPU) and the API, about 25 s. It skips itself without the weights or the local
dataset, as in CI.
