# Pipeline: from a video to layers in the frontend

How a clip travels through Sentinel, how to add a model, and how to rebuild the demo clips.
Command details live in `apps/backend/README.md`; per model facts (weights, configs, rates, times)
in `docs/pipeline_models.md`; the layer formats in `docs/layers/`. Tier 1 targets against the
measured numbers, and what Tier 2 consumes: `docs/tier1_ticklist.md`; demo steps: `docs/demo_checklist.md`.

## How a clip flows

```
video file or frame folder
  │  manage.py ingest_clip <source> <clip_id>          (apps/backend/apps/pipeline/ingest.py)
  ▼
<CLIPS_ROOT>/<clip_id>/  clip.json + video.mp4 (H.264 crf 18) + layers/ (empty)
  │  manage.py run_models <clip_id> --modules all      (apps/backend/apps/pipeline/runner.py)
  │    build:   registry.build_modules     one Module per model, needed ones first (points before tracks)
  │    decode:  video.mp4 once, at most n_frames frames, progress logged every 100 frames
  │    run:     run_clip  every frame to every module; each module picks its own frames (every)
  │    write:   add_layer.py  checks each layer against the contract, writes layers/<name>.json
  │             and its clip.json entry; then clip.json produced_by (model, weights md5, seconds)
  ▼
Django clips API  /api/v1/clips, /clips/<id>, /clips/<id>/video, /clips/<id>/layers/<name>
  ▼
frontend (apps/frontend)  single view (all layers over the video) and split view (one panel each)
```

From the browser the same two steps run behind `POST /clips` (upload, ingest) and
`POST /clips/<id>/runs` (a background worker; with the default `kaggle` runner the models run on a
Kaggle GPU kernel and the layers are downloaded and added here). See "Upload and run from the
browser" in `apps/backend/README.md`.

A live stream takes the same modules on frames as they arrive: `POST /streams` (a stored clip
replayed at real time) or `manage.py run_stream <source>` reads the source, `run_live` gives
each module chain its own thread and drops frames a slow model cannot reach in time, every
sample goes out through `GET /streams/<id>/samples` (polled) and the picture through
`GET /streams/<id>/video` (MJPEG), and at the end the stream is saved as a normal clip with
its layers. The frontend's Live page shows every running stream with the same renderers, each
sample as soon as it arrives. See "Live streams" in `apps/backend/README.md` and "Live page" in
`apps/frontend/README.md`; latency below.

`CLIPS_ROOT` defaults to `data/samples/` (gitignored). Layers on a clip today: `density` (A, grid),
`points` (B), `infrastructure` (C, polygons), `flow` (D, grid of u, v, speed), `tracks` (E).

Own footage: give the camera at ingest (`ingest_clip --altitude 60 --dfov 82.1`, or the camera
fields of the upload form). It is kept as clip.json `camera`, and every run (local, Kaggle, live)
passes it to flow, which then writes m/s instead of px per frame. Check the metres with
`python eval/pipeline/geometry_check.py <clip dir> --lines <segments.yaml> --out <dir>`: marked
known lengths (parked cars about 4.5 m) give the altitude that would make them right, and track
and flow speeds are compared with walking (1.2 m/s).

## When a module fails

`run_models` keeps going without it. A module that cannot be built (missing weights or package),
raises on a frame, or returns a layer that `add_layer.py` refuses is dropped, and so is every
module that needs it (tracks when points fails). All other layers are added with their
`produced_by` entries. The command then exits with an error listing each failed module and why,
for example `density: build failed: AssertionError: Torch not compiled with CUDA enabled`; the
full traceback is in the log above it. A layer that failed keeps whatever version the clip had.

These still stop the run before any model work: an unknown module name, a missing clip or
video, a video size that differs from clip.json, and layers the clip already has without
`--replace`.

## Add a model

1. Write a class next to the model with `name`, `process(frame_id, frame)`, `layer(clip_id)`,
   and for live mode `wants(frame_id)` and `samples` (the `Module` protocol in `ml/src/sentinel_ml/pipeline/module.py`; `DensityModule` in
   `ml/src/sentinel_ml/density/layer.py` is a short example). It returns a layer in one of the
   contract types: grid, points, tracks, polygons.
2. Register a builder in `BUILDERS` (`ml/src/sentinel_ml/pipeline/registry.py`): config from
   `ml/configs/`, weights through `_from_repo`, wrapped with `_about(...)`. Add it to `NEEDS`,
   `USES_FPS` or `SHOWN` if it uses another module's output, the clip's fps, or should start
   visible.
3. Run `python manage.py run_models <clip> --modules <name>`, then check the layer in both
   frontend views. A new layer type also needs a renderer in `apps/frontend/src/lib/overlay.ts`.
4. Add unit tests in `ml/tests/`; the end to end test below picks up every registered module
   by itself, so add the weights path to its skip check if the model has weights.

No model specific code goes into the backend.

## Live latency (step 29, 2026.10.21)

Source: `live_demo48` (DroneCrowd 11, 23, 70, 71 back to back, 1200 frames, 25 fps, 1920x1080), all
five models, streams started together, numbers over the time all were live. Latency = frame
capture (read by the stream worker) to: sample ready (model done), received (browser poll) and
drawn (on the canvas). Kaggle numbers are a T4 with Kaggle's few vCPUs, not the RTX 4070 box.

| Where | Streams | Read fps per stream (25 needed) | density | points / tracks | infrastructure | flow | picture drawn |
| --- | --- | --- | --- | --- | --- | --- | --- |
| this PC, CPU | 1 | 25 | 8.7 s | 4.7 / 4.7 s | 2.4 s | 0.7 s | 0.05 s |
| this PC, CPU | 3 | ~12 (reader 50 s behind) | 29 s | 23 / 23 s | 13.6 s | 1.4 s | 0.11 s |
| this PC, CPU | 4 | 3 of 4 workers out of memory | | | | | |
| Kaggle T4 | 1 | 19.5 | 0.17 s | 1.4 / 1.4 s | 0.32 s | 1.3 s | |
| Kaggle T4 | 3 | 7.2 | 0.29 s | 2.1 / 2.2 s | 0.74 s | 4.2 s | |
| Kaggle T4 | 4 | 5.1 | 0.36 s | 2.5 / 2.6 s | 1.0 s | 5.5 s | |

Medians of capture to ready; the browser adds 0.1 to 0.2 s (poll every 250 ms, then draw within
one frame), measured on this PC. Infrastructure p90 on Kaggle is 9 to 17 s because its first call
warms up for ~13 s. GPU memory on the T4: 2.2 GB for 1 stream, 6.4 GB for 3, 8.5 GB for 4.

What it means:

* Models: on a GPU every model but flow is under 3 s at 1 to 4 streams; flow (Farneback, CPU)
  passes at 1 stream only. On CPU only flow and infrastructure (1 stream) pass; density and points
  arrive 5 to 9 s late, so the live view shows them as gaps (the rule hides a sample older than
  `max_age_frames` plus 3 s).
* CPU side is the bottleneck: each stream decodes every frame, records it (x264), encodes the
  preview JPEG and runs Farneback. On Kaggle this caps all streams together at ~20 frames/s, so
  even 1 stream reads below real time and 3 to 4 streams fall 1 to 3 min behind the source. On
  this PC (12 threads, models on CPU too) 3 streams fall 50 s behind; 4 streams exceed the Windows
  commit limit (each worker commits ~4.7 GB; 32 GB RAM + 2 GB page file).
* Fallback for Tier 1: one live stream at full rate (GPU), or 2 to 4 streams at a lower read rate
  (about 20 frames/s in total on Kaggle). To try on the 4070 box: cheaper recording (x264
  ultrafast or recording in its own thread), a read rate option, flow on a smaller frame. Not done
  (user choice); carried to step 31 as gaps.

Raw results: `runs/step29/cpu_{1,3,4}.json` (browser side, `measure.js`), Kaggle kernel
`eval/pipeline/kaggle/live_latency/` output in `runs/kaggle/live_latency/v1/` (`result.json`,
`latency.md`, per stream `samples.jsonl`), summary script `eval/pipeline/live_latency.py`.

## Checks

| Suite | Command | Notes |
| --- | --- | --- |
| ml | `.venv\Scripts\python -m pytest ml/tests` | |
| backend | `cd apps/backend; ..\..\.venv\Scripts\python -m pytest` | includes the end to end test |
| frontend | `cd apps/frontend; npm test; npm run lint` | |

The end to end test (`apps/backend/tests/test_end_to_end.py`) copies the first 10 frames of
DroneCrowd test clip 11, runs `ingest_clip`, `run_models --modules all --device cpu` with the
real weights, validates the clip folder, reads every layer through the API, and checks that the
frame 0 counts of density and points are within about 30% of the ground truth (211). About 25 s
on the Ryzen 5 3600. It skips itself when the weights, the points extra or the local DroneCrowd
copy are missing, so CI (no weights, no data) skips it and runs only the tests with fake modules.

## Rebuild the demo clips

The demo set is seq011 (test 11, day), seq023 (test 23, dense), seq070 (test 70, night) and
seq071 (val 71, dusk), 300 frames each with all five layers, about 17 min on a T4.

On a machine with an NVIDIA GPU (setup in `docs/run_tier1.md`):
`python eval/pipeline/build_demo_clips.py --device cuda` (DroneCrowd under `data/dronecrowd/`,
weights from `eval/pipeline/fetch_weights.py`; existing clips are skipped, `--rebuild` remakes
them). Without a local GPU, on Kaggle:

```powershell
$env:PATH="$PWD\.venv\Scripts;$env:PATH"; $env:PYTHONUTF8="1"; $env:KAGGLE_USERNAME="ahmar004"
python eval/density_dmcount/kaggle/push.py src -m "demo clips"   # code, configs, docs/layers
# wait until `kaggle datasets status ahmar004/sentinel-ml-src` says ready
python eval/density_dmcount/kaggle/push.py kernel eval/pipeline/kaggle/demo_clips
# when `kaggle kernels status ahmar004/sentinel-demo-clips` says COMPLETE:
kaggle kernels output ahmar004/sentinel-demo-clips -p runs/kaggle/demo_clips/vN
# copy runs/kaggle/demo_clips/vN/clips/seq0* to data/samples/
```

The kernel (`eval/pipeline/kaggle/demo_clips/run.py`) runs the same `ingest_clip` and
`run_models --modules all --device cuda` commands and `docs/layers/validate.py` per clip. It
needs the private datasets `sentinel-dmcount-ckpt` (A weights), `sentinel-b-ckpt` (uav-dot code
and B checkpoint, `push.py uavdot`) and the notebook output of `sentinel-infra-train` (C weights).
To edit the clip list, change `DEMO` in that file. On CPU the same two commands work locally,
but B makes each clip take about 6 to 7 min.
