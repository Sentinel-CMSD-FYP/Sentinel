# 0007 Models A (density, DMCount) and C (infrastructure, YOLO26): Tier 1 results

Date: 2026.10.21   Owner: Ahmar   Branch: `density-dmcount`
Context: sprint task [2-12]. Records what was built for Model A (crowd density) and Model C
(infrastructure segmentation), the numbers measured, the choices made and the gaps carried into Tier 2.
Details live in `README-DMCOUNT.md` (Model A), `eval/infrastructure_yolo26/LABELLING.md` (Model C) and
`docs/layers_density_infrastructure.md` (layer formats). Result files are under `data/outputs/` (not in git).
Hardware: no RTX 4070 was available. All GPU numbers are from a Kaggle Tesla T4, not the 4070.

## Results

### Model A: count error per bucket (DroneCrowd test, 30 clips, 9000 frames, all clips)

Mean absolute relative count error per frame / bias (negative = undercount). Buckets by GT count per frame
(decision 0005: sparse < 100, moderate 100 to 200, dense > 200). Targets are provisional (unsigned draft
Phase 0 sheet s4.2); per 0005 nothing is called pass or fail. Gap = fine tuned minus target, in points
(positive = above the target figure). Test was scored once per model.

| group | frames | zero shot (QNRF) | fine tuned (`finetune_e4`) | MAE fine tuned | Model B (points) | target | gap |
|---|---|---|---|---|---|---|---|
| overall | 9000 | 75.0% / -74.7% | 22.7% / -4.2% | 26.8 | 21.6% / -9.6% | | |
| sparse | 3427 | 85.3% / -85.3% | 25.1% / -14.9% | 17.3 | 24.2% / -14.9% | <= 15% | +10.1 |
| moderate | 3689 | 71.6% / -71.3% | 28.0% / +0.2% | 39.9 | 26.3% / -8.6% | <= 15% | +13.0 |
| dense | 1884 | 63.2% / -62.1% | 8.0% / +6.5% | 18.6 | 7.4% / -1.7% | <= 25% | -17.0 |
| day (27 clips) | 8100 | 72.9% | 15.8% / +4.7% | 22.9 | 18.9% | | |
| night (69, 70, 74) | 900 | 93.9% | 85.0% / -85.0% | 62.3 | 45.4% | | |

Day only (secondary): sparse 10.2% (gap -4.8), moderate 24.6% (gap +9.6), dense 8.0% (gap -17.0).
Val (10 held out train clips, 300 frames, used for checkpoint choice): zero shot 84.4% (MAE 135.9),
fine tuned 13.8% (MAE 15.37). fp16 gives the same val numbers (MAE 15.37), so fp16 is the default on CUDA.
Full tables: `data/outputs/density_count_error/compare_{val,test}.md`, `finetune_e4/notes.md`.

What the numbers say:
* Fine tuning removed the zero shot undercount on high altitude views: 19 of 30 clips are under 15%.
* Dense meets its figure with room; sparse misses it only through night (day only 10.2%).
* Moderate is the weakest bucket. Clips 17 and 18 (low altitude, large people with long shadows) overcount
  by about 70% (2 or 3 blobs per person, GT complete); night clips undercount 74 to 94%. Moderate without
  17/18 is 19.9%, without 17/18 and night 15.0%.
* Night is far worse than Model B (85.0% vs 45.4%).

### Model C: segmentation on the held out val images (30 images, 589 instances, scene split)

Weights `model_c_best.pt` (epoch 118 of 150, md5 12213bdea884...), imgsz 1280.

| class | val instances | mask mAP50 | mask `mAP50-95` | box mAP50 |
|---|---|---|---|---|
| all | 589 | 0.318 | 0.185 | 0.331 |
| obstruction | 392 | 0.778 | 0.464 | 0.793 |
| wall | 183 | 0.492 | 0.274 | 0.517 |
| barrier | 12 | 0.002 | 0.000 | 0.015 |
| gate_exit | 2 | 0.000 | 0.000 | 0.000 |

With the class wise NMS of the wrapper (IoU 0.5): mask mAP50 0.321, otherwise unchanged. fp16 and CPU fp32
give the same mAP. Sources: `data/outputs/infra_bench/notes.md`, `nms_check/nms_check.md`.

What the numbers say: vehicles and stalls (obstruction) are found well, also top down. The mAP is
pessimistic because the GT misses some buildings the model marks as wall. Barrier and gate_exit are not
learned (58 and 10 labels in total; barriers are a few pixels wide). Large flat grey roofs and dark walls
at night are missed.

### Speed (Kaggle T4, 1920x1080, not RTX 4070)

| item | fp32 | fp16 |
|---|---|---|
| DMCount (A), ms per frame | 317 (3.2 FPS) | 140 (7.1 FPS) |
| YOLO26 network alone (C) | 36.1 ms (27.7 FPS) | 23.9 ms (41.8 FPS) |
| C wrapper (NMS + polygons) | | 41 ms (24.5 FPS) |
| local CPU (Ryzen 5 3600) | A 5.5 s, C 0.3 s (wrapper 0.4 s) | |

Concurrent streams (one process per stream, full pipeline with decode, A every 25th frame = 1 Hz, C every
125th = every 5 s, fp16). A stream keeps up with live video when it reaches 25 video FPS.

| streams | video FPS per stream (min) | A ms under load | GPU memory loaded / peak |
|---|---|---|---|
| 1 | 54.9 | 129 | 1.37 / 1.9 GB |
| 2 | 45.6 | 177 | 2.7 / 3.8 GB |
| 3 | 37.4 | 224 | 4.1 / 5.6 GB |
| 4 | 30.3 | 295 | 5.5 / 7.5 GB |

fp32 cannot serve 4 streams (warm streams reached 18 FPS). A 4070 is estimated at roughly 2x a T4 (not
measured); 12 GB allows about 6 such streams by memory. Source: `data/outputs/fps_bench/notes.md`.

## Decisions

Model A
* DroneCrowd only (own footage later, for visual checks only). Kaggle reuses the public copy
  `onvithong/dronecrowddata` (checked byte identical to the official release by md5).
* DM-Count ported into `ml/src/sentinel_ml/density/` (MIT, commit cc5f213), not a submodule; started from
  the UCF QNRF weights.
* Ahmed's frozen split and loader adopted unchanged (val = 10 held out train clips); frame stride train 5,
  val 10, test all. Official `val_data` is a copy of test and is never used.
* Training: DM-Count QNRF defaults (crop 512, batch 10, Adam lr 1e-5, OT loss), best checkpoint by val MAE,
  early stop patience 4, 3 GPU h budget. Best = epoch 4.
* Count error protocol of 0005 adopted with our own script; frames with GT < 10 stay out of the percentages.
* Density decision (Ahmar, step 10): DMCount is the density head for dense zones (> 200 people), Model B
  points count sparse and moderate zones. `finetune_e4` is the Tier 1 model; no retraining from test findings.
* fp16 on CUDA by default (2.3x faster, same val error).

Model C
* 4 classes in id order: `gate_exit` (openings people pass through), `barrier` (fences, railings, hedges),
  `wall` (building footprints, solid walls), `obstruction` (stalls, kiosks, all vehicles, poles). They map
  onto the SRS cell attributes exit, barrier, non walkable, obstruction (FR9.3/FR9.4).
* Masks (instance segmentation), not boxes; `yolo26s-seg` from COCO, imgsz 1280.
* Labelling in Label Studio (local; no Docker for CVAT). 150 DroneCrowd images, 2927 polygons (wall 903,
  obstruction 1956, barrier 58, gate_exit 10). Batch 1 (30 images) drafted with SAM and reviewed by
  Ahmar; batches 2 to 4 (120 images) were drafted by a YOLO26 model trained on the labels so far and
  submitted without human review (Ahmar's choice). Rules: `LABELLING.md`.
* Train/val split by scene, not by image (cameras are static, so frames of one clip are near copies):
  56 scene groups from feature matching plus merges by eye, seeded balanced search. Train 120, val 30.
* Wrapper adds class wise NMS (YOLO26 is NMS free and gave duplicate walls) and fp16 on CUDA.

Integration (steps 16 to 19)
* Layers: density = `grid` layer `density`, 9x16 zones of 120 px, people per zone, every 25 frames,
  zones sum to the frame count. Infrastructure = new 4th layer type `polygons` (per object `cls`, `score`,
  `box`, `polygon`), every 125 frames. Formats: `docs/layers_density_infrastructure.md`.
* Backend stays Django (Mahid's 0003). Every model is a `Module` (`process(frame_id, frame)`,
  `layer(clip_id)`) in `sentinel_ml.pipeline`; commands `ingest_clip` (H.264 crf 18) and `run_models`
  write layers through `add_layer.py`. The frontend got a `polygons` renderer; both views were checked.
* H.264 compression shifts DMCount counts by 2.6% mean (5.1% max, bias -1.1%) against the JPEGs on clip 11.
  The test numbers above are JPEG based.

## Setup issues met (and fixes)

* Kaggle: the new KGAT token holds no user name (set `KAGGLE_USERNAME`); the CLI crashes on Windows without
  `PYTHONUTF8=1`; kernels had no internet until the account was phone verified; a kernel pushed right after a
  dataset update runs on the old version (wait for `kaggle datasets status` ready).
* Kaggle runs: twice a GPU kernel hung in RUNNING for hours with no output, so every GPU kernel now has a
  watchdog and a timestamped log; a cancelled run keeps no output; queue waits ran from 0 to 4 h (night
  pushes get a GPU fastest); P100 is unusable with torch cu128; a kernel must never list itself in
  `kernel_sources` (resume goes through a private checkpoint dataset instead).
* DM-Count's `demo.py` skips ImageNet normalisation; we normalise as in its test code. The model outputs 134
  rows for 1080 input against 135 GT cells.
* Google Drive refused `train_data.zip` to gdown (downloaded in the browser).
* Ultralytics `masks.xy` joins mask parts with bridge lines; the wrapper keeps the largest contour.
* Label Studio export writes a file, not a folder, if the target folder does not exist.
* Full list: `README-DMCOUNT.md` section 7 and `progress.md`.

## Alternatives and why not

* DOT density head: deferred; DMCount was the sprint's first choice, DOT stays a later benchmark.
* Starting from the NWPU, SHA or SHB weights: QNRF chosen in step 2; the others were not tried.
* Boxes for Model C: thin barriers and long walls fit boxes badly.
* FastAPI service: the team backend was already Django (0003).
* Pass/fail verdicts on the targets: the Phase 0 sheet is unsigned (0005).

## Gaps and open items for Tier 2

* Model A: low altitude views overcount (scale augmentation or multi scale training); night undercounts
  (low light data or augmentation); noisy val curve (LR decay, larger or averaged val); per zone dense
  threshold (the > 200 rule is per frame); own 40 to 80 m footage not yet seen by the model.
* Model C: barrier and gate_exit need many more labels (own footage with gates and barriers); a label fix
  pass on the known gaps (market stall rows, stadium stands, missed buildings, night clips); 120 of 150
  images were never reviewed by a person; only 30 val images, so mAP is noisy; partly overlapping walls
  with box IoU < 0.5 remain after NMS.
* Speed: measured on a T4, not the 4070; B, D and E were not running alongside; latency from capture to
  overlay is not measured (step 29); a 5th stream would likely fall behind on a T4.
* Team: Mahid to accept the `polygons` type (his `check_layer.py`, `validate.py` and schemas were extended
  here); Ahmed to confirm the density grid for Tier 2; polygons have no hover readout yet.
* Dusk and night scope is still open (0005).

Evidence: steps 2 to 19 in `progress.md`; result files `data/outputs/density_count_error/`,
`data/outputs/infra_bench/`, `data/outputs/fps_bench/`; Kaggle kernels `ahmar004/sentinel-dmcount-*`,
`sentinel-infra-*`, `sentinel-fps-bench`.
