# Pipeline model inventory

Every model reaches the frontend the same way: one `Module` (`name`, `process(frame_id, frame)`,
`layer(clip_id)`) registered in `ml/src/sentinel_ml/pipeline/registry.py`, run by
`python manage.py run_models <clip_id> --modules ...`, written with `add_layer()`, drawn by the
frontend in the single and split views. This file lists what each model has today and what is
still missing before it can join. Status as of 2026.10.21 (roadmap step 24).

## Summary

| Model | Module name | Layer type | Rate (frames) | CPU per call | T4 GPU per call | Owner | Status |
|---|---|---|---|---|---|---|---|
| A density (DM-Count) | `density` | grid 9x16, `people_per_cell` | every 25 | 5.5 s | 140 ms fp16 (317 fp32) | Ahmar | in pipeline |
| B points (DOT + Pixel Distill) | `points` | points `[x, y, score]`, score >= 0.2 | every frame | 1.1 s | 209 ms (model only) | Ahmed | in pipeline |
| C infrastructure (YOLO26 seg) | `infrastructure` | polygons | every 125 | 0.4 s | 41 ms fp16 | Ahmar | in pipeline |
| D flow (Farneback) | `flow` | grid 6x10, `u, v, speed`, px per frame (m/s with clip `camera`) | every 25, pair gap 5 | 180 ms per sample | runs on CPU | Mahid | in pipeline |
| E tracks (Kalman point tracker) | `tracks` | tracks `[track_id, x, y, score]`, score >= 0.35 | every frame B runs on | 5 ms | runs on CPU | Ahmed | in pipeline, needs `points` |
| Geometry (ground sampling) | none (library) | none | none | negligible | none | Mahid | used by D for m/s |

Timing sources: A and C from decision 0007 (step 19, 1920x1080, wrapper incl. postprocess); B from
`docs/project-log.md` (4.79 frames/s on Kaggle T4, model only, 3,000 val frames); D and E measured
on this PC (Ryzen 5 3600) on 2026.10.21: one flow sample on `seq011` = Farneback 87 ms at
960x540 (downscale 0.5, window 9) + camera motion fit 91 ms (at 480x270, window 15: 24 + 76 ms),
and the tracker 4.95 ms per frame with 130 fake points over 300 frames; B on CPU 1.1 s per frame
(1920x1080, 6 threads, step 23). None of these are RTX 4070 numbers.

CPU cost per second of 25 fps video on this PC: A 5.5 s (one sample), C about 0.1 s, D about 0.18 s,
E about 0.13 s. B at every frame would be 25 model calls per second, so B on CPU is the bottleneck
to settle in step 23 (rate, threshold, or a Kaggle run that writes the layer).

## A density (DM-Count)

* Entry: `sentinel_ml.density.layer.DensityModule`, built by `build_module("density", device)`.
* Weights: `data/weights/dmcount/finetune_e4.pth` (gitignored, md5 725694ba...; Kaggle dataset
  `ahmar004/sentinel-dmcount-ckpt` `best.pth`). Tier 1 model, see decision 0007.
* Config: `ml/configs/density_dmcount.yaml` (model, `half: true` on CUDA), `ml/configs/density_layer.yaml`
  (grid, rate, `max_age_frames` 25).
* Missing: nothing for Tier 1.

## B points (DOT + Pixel Distill)

* Module: `sentinel_ml.localization.layer.PointsModule`, built by `build_module("points", device)`
  (device defaults to cuda when available). It wraps `localization.infer.DotDetector(cfg, device)`
  (one RGB frame to (N, 3) `(x, y, score)` in original pixels) and keeps rows at or above
  `score_thr`. `infer.py` still writes the offline `<clip_id>_points.json` files (contracts v1.0).
* Code: `up-count/uav-dot` (Apache 2.0, commit a6d3664) cloned into gitignored
  `ml/third_party/uav-dot` (step 23): `git clone https://github.com/up-count/uav-dot` then
  `git checkout a6d3664`.
* Weights: the authors' released DroneCrowd checkpoint
  `ml/third_party/uav-dot/checkpoints/dot_pd_dronecrowd_51.00.ckpt` (330 MB, md5
  fcf6762ad393717741ffc7e6c467f021, Google Drive id `1wYa01jGYfrAun3SfxuWzcKAni3hzBOMV` from the
  repo README, `gdown <id> -O <path>`). No own training. Tier 1 = this release (user choice
  2026.10.21); mirror it in a private Kaggle dataset when a GPU run needs it.
* Configs: `ml/configs/infer_b.yaml` (model: net input 1920x1088 stretched, `min_score` 0.05; the
  registry accepts its keys as overrides, e.g. `checkpoint=...`) and `ml/configs/points_layer.yaml`
  (layer: `score_thr` 0.2, `every` 1, `max_age_frames` 1). 0.2 is B's threshold chosen on val
  (B-4, `docs/project-log.md` 14 Oct); E thresholds again at 0.35 (its own val choice).
* Deps: extra `points` in `ml/pyproject.toml` (`pip install -e "ml[points]"`:
  `segmentation-models-pytorch==0.3.4`, `timm==0.9.7`, pytorch-lightning, hydra-core; no torch
  change). smp downloads the MiT b2 encoder weights (94 MB, to the torch hub cache) the first time
  the model is built, so it needs internet once. The checkpoint loads through
  `restricted_torch_load` (allow list of pickle globals).
* Measured on this PC (step 23): load ~20 s, 1.1 s per 1920x1080 frame on CPU; seq011 every frame
  = 5 min 52 s through `run_models`. seq011 points per frame median 228 vs GT 207 (+11%), layer
  1.7 MB.

## C infrastructure (YOLO26 seg)

* Entry: `sentinel_ml.infrastructure.yolo26.InfraModule`, built by `build_module("infrastructure", device)`.
* Weights: `data/weights/yolo26/model_c_best.pt` (gitignored, md5 12213bdea884...; Kaggle kernel
  output `ahmar004/sentinel-infra-train`). Base COCO `yolo26s-seg.pt` for training only.
* Config: `ml/configs/infrastructure_yolo26.yaml` (imgsz 1280, NMS 0.5, `every` 125, fp16 on CUDA).
* Missing: nothing for Tier 1 (hover readout of class and score added in step 25).

## D flow (Farneback)

* Entry: `sentinel_ml.flow.layer.FlowModule(cfg, farneback, fps=None)`, registered as `flow`
  (`run_models <clip> --modules flow`). Every 25 frames it computes flow to the frame 5 later,
  removes camera shift, zoom and rotation (`flow/camera.py`), marks low texture cells invalid
  and pools onto the grid; a sample with ambiguous camera motion is left out (a gap). The sample
  is keyed by the pair's first frame. `flow/export.py` (whole clip folder) now runs the same
  module, so both give identical layers.
* Weights: none. Config: `ml/configs/flow_layer.yaml` (grid 6x10, every 25, gap 5, min_texture
  1.5, stabilise on, Farneback downscale 0.5 window 9 as in export.py). `flow_farneback.yaml`
  (0.25, window 15) is unchanged and used by E's camera shift.
* Units: px per frame, or m/s when the clip.json `camera` block (altitude, horizontal FOV) is set:
  `build_modules(..., fps, errors, camera)` passes it to the flow builder (step 30; ingest flags
  or the upload form write it). `produced_by.flow.units` records which.
* Known: DroneCrowd people barely move (seq011 mean 0.03 px per frame). Since step 25 the
  frontend scales arrows to max(peak, 0.5 px/frame), so this noise draws short arrows; tall
  buildings still give parallax arrows that camera removal cannot take out. Flow starts hidden.

## E tracks (Kalman point tracker)

* Entry today: `sentinel_ml.tracking.run_tracker.track_clip(points_doc, cfg, ...)` (offline, on a
  B points file) and `sentinel_ml.tracking.tracker.PointTracker(frame_size, cfg).update(points)`,
  one call per frame, returns `(track_id, x, y, score)` rows. The per frame form fits a module.
* Weights: none. Config: `ml/configs/tracker_e2_candidate.yaml` is Tier 1 (user choice 2026.10.21;
  frozen and scored in test run 2: score 0.35, min_hits 3, max_age 3.0 s, CMC off, altitude off).
  `max_age_s` must go through `resolve_config(cfg, fps, frame_stride, altitude_m)` first.
* Results (Ahmed, test run 2): count error 35.2% unfiltered, 37.9% filtered, IDF1 0.388, HOTA 0.300.
* Module (step 24): `sentinel_ml.tracking.layer.TracksModule`, name `tracks`, layer config
  `ml/configs/tracks_layer.yaml` (`max_age_frames` 1, rounding as `run_tracker`). It declares
  `needs = ("points",)` and gets the points module itself: `registry.build_modules` builds each
  needed module once and puts it first (`run_clip` refuses the wrong order), and per frame the
  tracker reads `PointsModule.rows(frame_id)`, so B runs once for both layers. Asking for
  `tracks` alone also adds the `points` layer. It tracks only the frames B ran on (stride = B's
  `every` in `resolve_config`), at E's own threshold 0.35 on B's 0.2 rows.
* `max_age_s` uses the clip's fps: `run_models` reads `fps` from `clip.json` and passes it to the
  builders in `registry.USES_FPS` (25 fps gives 75 frames, as in test run 2).
* The layer carries `"camera_motion_compensated": false` (extra top level key, allowed by the
  contract tools). Positions are B's detections in image pixels either way; `use_cmc` and
  `use_altitude` are refused by the module until the pipeline passes camera shifts and altitude.
* Same result as offline: the module's frames equal `run_tracker.track_clip` on the points layer
  (unit test and seq011 check).
* seq011 (step 24, CPU): `run_models seq011 --modules tracks --replace` 6 min 53 s (B is nearly
  all of it), points layer identical to step 23. 370 track ids vs 269 GT ids; per frame median 127
  tracked, 130 points >= 0.35, 207 GT; track span median 125 frames, 49 tracks cover the whole clip.
  GT to track match within 10 px: 242 of 269 GT ids matched, 43% keep one track id (median 2 ids
  per person). Ids are stable for long stretches but break, in line with test run 2 (IDF1 0.388).

## Geometry

* Entry: `sentinel_ml.geometry.ground` (`gsd`, `flow_to_ms`, `NadirCamera`, ...), pure functions
  for a straight-down camera over flat ground (decision 0006). Needs altitude and horizontal FOV;
  DroneCrowd and the YouTube clips carry neither, so it matters only for own footage (step 30).
* Step 30: clip.json `camera` block (`altitude_m`, `hfov_deg`, `altitude_source`, optional
  `model`, `dfov_deg`, `sensor_aspect`). Check script `eval/pipeline/geometry_check.py` (known
  lengths give the implied altitude; track and flow speeds in m/s against walking 1.2 m/s).
  E's `use_altitude` stays off (frozen E2 config, Ahmed's call).

## Branches

* `feat/ml-localization-tracking` (B + E baseline) reached main as PR #7 and is deleted on the remote.
* `feat/ml-tracking-e2` (E-2 tracker work: `tracking/cmc.py`, `tracking/altitude.py`,
  `max_age_s`, TrackEval and B+E benchmark scripts) merged into `density-dmcount` on 2026.10.21
  (no conflicts, left for the user to commit).
