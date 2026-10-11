# Data contracts (Tier 1) — Models B and E

> STATUS: **PROPOSED**. Becomes final only after the team kickoff agrees (ROADMAP task P0-5). Don't change after that without Ahmed's approval.

## Conventions
- Coordinates in **original frame pixels**, `x` = column, `y` = row, origin top-left.
- `frame_id`: integer, 0-based index within the clip, as a string key in JSON.
- One JSON file per clip: `<clip_id>_points.json`, `<clip_id>_tracks.json`.

## Model B output: points
```json
{
  "clip_id": "clip_001",
  "frame_size": [1920, 1080],
  "frames": {
    "0": [[x, y, score], [x, y, score]],
    "1": [[x, y, score]]
  }
}
```
`score` in [0, 1].

## Model E output: tracks
```json
{
  "clip_id": "clip_001",
  "frame_size": [1920, 1080],
  "altitude_m": 60.0,
  "frames": {
    "0": [[track_id, x, y, score]],
    "1": [[track_id, x, y, score]]
  }
}
```
`track_id`: integer, unique within the clip, never reused.

## Ground truth (internal, for eval)
Loader returns `(image, points)` where `points` is an `(N, 2)` array of `(x, y)`. Tracking GT also carries a person ID per point.