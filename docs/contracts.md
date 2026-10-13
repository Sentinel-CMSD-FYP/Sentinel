# Data contracts (Tier 1) — Models B and E

> STATUS: **PROPOSED, v1.0 draft**. Becomes FINAL only after the team agrees (ROADMAP task P0-5). Don't change after that without Ahmed's approval.
> Sample files: `docs/samples/clip_011_points.json`, `docs/samples/clip_011_tracks.json` (built from ground truth, not model output).

## Conventions
- Coordinates in **original frame pixels**, `x` = column, `y` = row, origin top-left. Not padded, not resized: if a model runs on a resized or padded input (DOT pads 1080 to 1088), points are mapped back before they are written.
- `frame_id`: integer, 0-based index within the clip, as a string key in JSON.
- One JSON file per clip: `<clip_id>_points.json`, `<clip_id>_tracks.json`.
- Readers should ignore keys they do not know, so fields can be added later.

## Header (both files)
| Field | Type | Meaning |
|---|---|---|
| `schema_version` | string | `"1.0"` |
| `clip_id` | string | e.g. `"clip_011"` |
| `frame_size` | `[w, h]` | original frame size in pixels |
| `fps` | number or `null` | frames per second of the clip; `null` if unknown |
| `altitude_m` | number | camera altitude in metres |
| `altitude_source` | string | `"assumed_constant"` or `"telemetry"` |

## Model B output: points
```json
{
  "schema_version": "1.0",
  "clip_id": "clip_001",
  "frame_size": [1920, 1080],
  "fps": 25,
  "altitude_m": 60.0,
  "altitude_source": "assumed_constant",
  "frames": {
    "0": [[x, y, score], [x, y, score]],
    "1": [[x, y, score]]
  }
}
```
`score` in [0, 1]. B saves all points with their scores; no threshold is applied when saving.

## Model E output: tracks
```json
{
  "schema_version": "1.0",
  "clip_id": "clip_001",
  "frame_size": [1920, 1080],
  "fps": 25,
  "altitude_m": 60.0,
  "altitude_source": "assumed_constant",
  "frames": {
    "0": [[track_id, x, y, score]],
    "1": [[track_id, x, y, score]]
  }
}
```
`track_id`: integer, unique within the clip, never reused.

## Ground truth (internal, for eval)
Loader returns `(image, points)` where `points` is an `(N, 2)` array of `(x, y)`. Tracking GT also carries a person ID per point.

## Changes
- **v1.0 draft (2026-10-14, approved by Ahmed):** added `schema_version`, `fps`, `altitude_source`; `altitude_m` now also in the points file; wrote down the "original pixels, not padded or resized" rule. Nothing removed or renamed.
- Assumptions to replace when real values exist: `altitude_m = 60.0` (midpoint of the 40–80 m range; DroneCrowd files carry no altitude) and `fps = 25` (given by Ahmed; not read from the dataset files).
- Possible later addition: per-frame altitude from telemetry, as a new optional field.
