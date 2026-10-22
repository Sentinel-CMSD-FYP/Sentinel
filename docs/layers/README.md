# Layer contract: model output to frontend

> STATUS: **PROPOSED**. Complements `docs/contracts.md` (Models B and E output formats), which
> still needs Ahmed's approval; this file extends it with a grid layer and the API around it.
> Do not change after the team agrees without a note in `docs/decisions/`.

The frontend plays a video and draws toggleable layers on top. Each model produces one layer. The
backend stores layers and serves them. Every layer is one of four **types**, so the frontend needs
only four renderers and adding a model does not need frontend work.

| type | one entry per frame holds | drawn as | example |
|---|---|---|---|
| `grid` | named matrices, one value per cell, plus a `valid` mask | heatmap or arrows | flow, density |
| `points` | rows of `[x, y, score]` | dots | localization (Model B) |
| `tracks` | rows of `[track_id, x, y, score]` | dots with trails | tracking (Model E) |
| `polygons` | objects `{cls, score, box, polygon}` plus a layer level `classes` list | outlines, one colour per class | infrastructure (Model C) |

## Conventions

Same as `docs/contracts.md`: coordinates in **original-frame pixels**, `x` = column, `y` = row,
origin top-left; frame ids are 0-based integers used as **string keys**; one JSON file per clip and
layer. Time is never stored: `t = frame / fps` from `clip.json`.

For now, grids are in **image space** (the cell grid of the camera frame). When geometry exists they
can become ground cells with the same fields, without changing the format.

## Rules the data must obey

- **Gaps stay gaps.** A frame that has no entry has no data. A cell with `valid: 0` has no data.
  Nothing is interpolated or carried forward. `max_age_frames` says how long a sample stays
  displayable; after that the frontend shows nothing.
- **Units are explicit.** Flow is `px_per_frame` in original-frame pixels until a ground conversion
  exists; then it becomes `m_per_s` and the `units` field says so. The flow exporter writes `m_per_s`
  when it is given `--altitude` and `--hfov` (a straight-down camera over flat ground is assumed;
  see `docs/decisions/0006-geometry-ground-sampling.md`). The pipeline's flow module writes `m_per_s`
  when clip.json has a `camera` block (`altitude_m`, `hfov_deg`, `altitude_source` assumed, stated
  or telemetry; optional `model`, `dfov_deg`, `sensor_aspect`), set by `ingest_clip --altitude` or
  the upload form.
- **Models may run slower than the video.** Flow and density at about 1 Hz are normal. Entries are
  sparse; they are not required for every frame.
- **Dwell before risk.** A risk layer (later) must not emit a score for a cell until its history fills
  the window.

## Files

```
clip.json                 metadata and layer manifest   (schemas/clip.schema.json)
layers/<layer>.json       one file per layer            (schemas/layer_<type>.schema.json)
video.mp4                 the footage
```

`video.mp4` must be **H.264 (AVC) with yuv420p pixels**, because browsers cannot play anything else.
OpenCV's default `mp4v` writer produces MPEG-4 Part 2, which fails in the browser with "no supported
streams"; re-encode with ffmpeg (`-c:v libx264 -pix_fmt yuv420p -movflags +faststart`).

`clip.json` lists the layers, their type, label, units and a `style` hint (for example
`{"render": "arrows"}` or `{"render": "heatmap", "field": "density"}`). The `layer` name must equal the
file name.

### Grid layer

```json
{
  "schema_version": 1, "clip_id": "seq011", "layer": "flow", "type": "grid",
  "frame_size": [1920, 1080], "grid": {"rows": 6, "cols": 10},
  "units": "px_per_frame", "fields": ["u", "v", "speed"], "max_age_frames": 25,
  "frames": {
    "0":  {"valid": [[1, 0, ...]], "u": [[...]], "v": [[...]], "speed": [[...]]},
    "25": {"valid": [[...]],       "u": [[...]], "v": [[...]], "speed": [[...]]}
  }
}
```

Every matrix is `rows x cols`. Values in cells with `valid: 0` are ignored.

### Points and tracks layers

```json
{"schema_version": 1, "clip_id": "seq011", "layer": "points", "type": "points",
 "frame_size": [1920, 1080], "max_age_frames": 5,
 "frames": {"0": [[747.0, 592.0, 0.91], [760.5, 601.2, 0.77]]}}
```

Tracks use `[track_id, x, y, score]` rows and may carry `altitude_m`. A `track_id` is unique within the
clip and never reused. `score` is in [0, 1].

## API (`/api/v1`)

```
GET  /clips                        list clips: id, name, fps, size, frame count, layers
GET  /clips/{id}                   clip.json (metadata and layer manifest)
GET  /clips/{id}/video             the video, with range requests so it can seek
GET  /clips/{id}/layers/{layer}    the layer file; optional ?from=&to= frame range
POST /clips/{id}/layers/{layer}    ingest a layer, validated before it is stored
GET  /health
WS   /ws/clips/{id}/               live push: {"layer", "frame", "data"} (later phase)
```

The frontend downloads each layer once and picks the entry for `round(video.currentTime * fps)`. The
WebSocket carries the same per-frame entries as the REST files.

## Validation

`validate.py` checks structure with the JSON Schemas and what schemas cannot: matrices that match
`rows x cols`, points inside the frame, unique track ids, frame ids inside the clip, and that every
layer file is declared in `clip.json` with the right type. The ingestion endpoint should call
`validate_layer`.

```
pip install jsonschema numpy opencv-python-headless imageio-ffmpeg
python docs/layers/make_sample_clip.py data/samples/demo_fake   # fake clip, validated on write
python docs/layers/validate.py data/samples/demo_fake           # validate any clip folder
python docs/layers/add_layer.py data/samples/demo_fake density.json --label "Density"
```

`add_layer.py` puts a model's layer into a clip folder and lists it in `clip.json`, so nobody edits the
list by hand. It refuses, and writes nothing, when the layer is invalid under the contract, was made for a
different `clip_id`, has a `frame_size` that differs from the clip's video (the model was run on a
different file or resolution), or uses frame numbers beyond the clip. It also refuses to overwrite an
existing layer unless `--replace` is given, and `--remove NAME` takes a layer out again. Grids start
visible and dots and tracks start hidden (`--visible` / `--hidden` override), and a grid with `u` and `v`
fields is drawn as arrows, any other grid as a heatmap of its first field (`--field` picks another).

Models B and E can hand over their files exactly as `docs/contracts.md` v1.0 defines them. `add_layer.py`
recognises that format (a string `schema_version` of `"1.0"` and no `type`), takes the layer type from the
file name (`<clip>_points.json`, `<clip>_tracks.json`, or `--kind`), and sets `max_age_frames` from the
file's `frame_stride` (`--max-age` overrides). Their extra header fields (`fps`, `frame_stride`,
`altitude_m`, `altitude_source`) are kept, and every other check above still applies. If their
`clip_id` differs from the clip folder's (for example `clip_011` and `dc_011`), add `--fix-clip-id`.

The fake clip (`make_sample_clip.py`) is generated, not committed (`data/` is gitignored). It is
simulated crowd data, not model output, with a deliberate gap in the flow layer (frames 100 to 149)
to exercise the gaps rule. Its video shows what the layers describe, so overlays line up.

## Phases

1. This contract and the fake clip.
2. Read-only backend serving clips, video and layers from files.
3. Ingestion with validation.
4. Live WebSocket, then dwell logic when risk arrives.

## Open questions

- The fifth model's output; it should fit one of the three types, otherwise a fourth type is needed.
- Flow's real grid size, and whether it ships per-cell `speed_var` or other Tier 2 inputs.
- Whether the video is served by the backend or from static storage.
