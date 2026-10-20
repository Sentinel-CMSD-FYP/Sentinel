# Layers from Model A (density) and Model C (infrastructure)

Status: PROPOSED by Ahmar (step 16, 2026.10.20), for Mahid and Ahmed to confirm. Builds on the layer contract `docs/layers/layers.md` on main (Mahid, PROPOSED). Density uses the existing `grid` type unchanged. Infrastructure needs a new fourth type, `polygons`, because masks fit none of grid, points or tracks.

Samples (clip `seq011` = DroneCrowd test clip 11, real model output, 1920x1080, 300 frames at 25 fps): `docs/samples/seq011/density.json`, `docs/samples/seq011/infrastructure.json`. Made by `python eval/make_layer_samples.py --clip 11` (local CPU, about 1 min).

Shared rules (as in layers.md): original frame pixels, x = column, y = row, origin top left; frame ids are 0 based and used as string keys; a frame without an entry has no data (gaps stay gaps); one file per layer per clip, named after the layer.

## Density: `grid` layer `density`

```json
{"schema_version": 1, "clip_id": "seq011", "layer": "density", "type": "grid",
 "frame_size": [1920, 1080], "grid": {"rows": 9, "cols": 16},
 "units": "people_per_cell", "fields": ["density"], "max_age_frames": 25,
 "frames": {"0":  {"valid": [[1, 1, ...], ...], "density": [[0.0, 4.21, ...], ...]},
            "25": {...}}}
```

* `density[r][c]` = predicted number of people in zone (r, c). Zones are 120x120 px (`rows`, `cols` are configurable in `ml/configs/density_layer.yaml`, default 9x16 as in Mahid's sample). Zone (r, c) covers x from `c * W / cols` to `(c + 1) * W / cols`, y likewise.
* Frame count = sum over all zones (no separate field, the grid schema does not allow extra keys). Values are fractional, 3 decimals.
* Built from DM-Count's native map (8 px cells, 134x240 at 1080p) by summing each cell into the zone that holds its centre, so nothing is lost or counted twice. `valid` is 1 where at least one map cell lies in the zone (always all 1 at 9x16; the map stops 8 px short of the bottom edge).
* Rate: every 25th frame (1 Hz at 25 fps), `max_age_frames` 25. DM-Count takes 370 ms per frame on a Kaggle T4, so 1 Hz is what it can sustain per stream.
* Use (step 10 decision): DM-Count is the counting head for dense zones (> 200 people); Model B points count sparse and moderate zones.
* Checked: passes Mahid's `check_layer.py` (main) with clip length 300; the per frame sums equal the step 10 Kaggle predictions for clip 11 (197.9 vs GT 211 at frame 0).

## Infrastructure: new `polygons` layer `infrastructure`

```json
{"schema_version": 1, "clip_id": "seq011", "layer": "infrastructure", "type": "polygons",
 "frame_size": [1920, 1080], "classes": ["gate_exit", "barrier", "wall", "obstruction"],
 "max_age_frames": 125,
 "frames": {"0": [{"cls": "obstruction", "score": 0.91,
                   "box": [1020.7, 397.0, 1047.4, 464.0],
                   "polygon": [[1021.5, 397.5], [1021.5, 462.0], [1047.0, 397.5]]}, ...],
            "125": [...]}}
```

* `classes`: all class names the layer can hold (order = model class id). Each object: `cls` (one of `classes`), `score` in [0, 1] (3 decimals), `box` = [x0, y0, x1, y1], `polygon` = outline of the object, at least 3 points, closed implicitly (last point joins the first), inside the frame. Pixel values have 1 decimal. Objects are sorted by score, highest first. An empty list means the frame was run and nothing was found.
* Classes (step 11, map onto the SRS cell attributes FR9.3/FR9.4): `gate_exit` = exit, `barrier` = barrier, `wall` = non walkable, `obstruction` = obstruction. Rules: `eval/infrastructure_yolo26/LABELLING.md`.
* Rate: every 125th frame (5 s at 25 fps), `max_age_frames` 125. Infrastructure is static (fixed camera), so it need not run every frame.
* Model: fine tuned YOLO26s seg (step 14), imgsz 1280, conf 0.25, fp16 on CUDA, fp32 on CPU. A class wise NMS (box IoU 0.5) removes duplicate masks of one object (YOLO26 is NMS free). Val mask mAP50 0.318 without and 0.321 with this NMS (30 val images, `eval/infrastructure_yolo26/nms_check.py`). Weak classes: barrier and gate_exit are almost never found (too few labels), so the layer holds mostly wall and obstruction for now.
* Each polygon is the largest connected part of its mask, simplified with a 1 px tolerance (about 20 points per object). Ultralytics' own `masks.xy` joins all parts with bridge lines that cross buildings, so it is not used.
* Renderer needed (Mahid): draw each polygon outline (or fill) in a colour per class, optionally with `cls` and `score` on hover. Step 17 (2026.10.21) added the type to `docs/layers/check_layer.py`, `validate.py` and `schemas/` (`layer_polygons.schema.json`, `clip.schema.json` type enum), tests `ml/tests/test_layer_polygons.py`; Mahid to review.

## Python API

```python
from sentinel_ml.density import DMCount, DMCountConfig, DensityLayerConfig, density_layer
from sentinel_ml.infrastructure import InfraConfig, InfraModel, infrastructure_layer

frames = ((fid, bgr_frame) for fid, bgr_frame in ...)   # 0 based frame id, OpenCV BGR uint8
dens = density_layer(frames, DMCount(DMCountConfig.from_yaml(...)), "seq011",
                     DensityLayerConfig.from_yaml("ml/configs/density_layer.yaml"))
infra = infrastructure_layer(frames2, InfraModel(InfraConfig.from_yaml(
    "ml/configs/infrastructure_yolo26.yaml")), "seq011")
```

Each function takes every frame of the clip and runs its model only on frame ids that are a multiple of its `every`; it returns the layer as a dict ready for `json.dump`. Weights paths in the configs are relative to the repo root.

Since step 17 both are also modules of the common interface (`DensityModule`, `InfraModule`, built by name with `sentinel_ml.pipeline.build_module`), which the backend command `python manage.py run_models <clip> --modules density,infrastructure` runs (`apps/backend/README.md`).

## Open for the team

* Mahid: accept the `polygons` type (or ask for changes) and add a renderer; Ahmed: confirm the density grid fits Tier 2 risk features.
* Zone size: 120 px is about 4 to 8 m on the ground at 40 to 80 m altitude (assuming an 80 degree horizontal field of view). Ground cells come with geometry (decision 0006).
