# Data contracts (Tier 1) — Models B and E

> STATUS: **FINAL, v1.0** (team agreed, reported by Ahmed 2026-10-14). Don't change without Ahmed's approval.
> Sample files: `docs/samples/clip_011_points.json`, `docs/samples/clip_011_tracks.json` (built from ground truth, not model output).

## Conventions
- Coordinates in **original frame pixels**, `x` = column, `y` = row, origin top-left. Not padded, not resized: if a model runs on a resized or padded input (DOT pads 1080 to 1088), points are mapped back before they are written.
- `frame_id`: integer, 0-based index within the clip, as a string key in JSON. It stays the index in the original clip even when `frame_stride` > 1 (so strided frames read `"0"`, `"5"`, `"10"`, ...).
- One JSON file per clip: `<clip_id>_points.json`, `<clip_id>_tracks.json`.
- Readers should ignore keys they do not know, so fields can be added later.

## Header (both files)
| Field | Type | Meaning |
|---|---|---|
| `schema_version` | string | `"1.0"` |
| `clip_id` | string | e.g. `"clip_011"` |
| `frame_size` | `[w, h]` | original frame size in pixels |
| `fps` | number or `null` | frames per second of the clip; `null` if unknown |
| `frame_stride` | integer, optional | `1` (default) = every frame was processed; `5` = every 5th frame. Effective rate = `fps / frame_stride`, so speeds are right when frames are skipped |
| `altitude_m` | number | camera altitude in metres |
| `altitude_source` | string | `"assumed_constant"` or `"telemetry"` |

## Model B output: points
```json
{
  "schema_version": "1.0",
  "clip_id": "clip_001",
  "frame_size": [1920, 1080],
  "fps": 25,
  "frame_stride": 1,
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
  "frame_stride": 1,
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
- **v1.0 FINAL (2026-10-14, approved by Ahmed, team agreed):** added `schema_version`, `fps`, `frame_stride`, `altitude_source`; `altitude_m` now also in the points file; wrote down the "original pixels, not padded or resized" rule. Nothing removed or renamed.
- `altitude_m = 60.0` is an assumption (midpoint of the 40–80 m range; DroneCrowd files carry no altitude). Replace it when real values exist.
- `fps = 25` is the published DroneCrowd value (Wen et al., arXiv 2105.02440, p.3: "recorded at 25 frames per seconds (FPS) with a resolution of 1920×1080"). It is not stored in the dataset files.
- Possible later addition: per-frame altitude from telemetry, as a new optional field.
