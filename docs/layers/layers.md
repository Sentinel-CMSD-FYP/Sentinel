# Sentinel layers: how your model's output plugs into the backend and frontend

Status: PROPOSED by Mahid. Read it, try it, and tell Mahid anything that does not fit before you
build against it. Nothing here is final until we all agree.

## The idea

The frontend plays a drone video and draws toggleable layers on top (for example optical flow
arrows, a density heatmap). Each model produces ONE layer. The backend stores the layers and the
frontend draws them.

To keep this simple, every layer is one of THREE types. The frontend only knows how to draw these
three, so if your output fits one of them, it works with no frontend changes:

  grid    a value (or vector) for every cell of a grid   drawn as a heatmap or arrows
  points  a list of [x, y, score] per frame               drawn as dots
  tracks  a list of [track_id, x, y, score] per frame     drawn as dots with trails

If your model's output does not fit any of these, tell Mahid BEFORE you build it.

## Rules for all layers

- Coordinates are in ORIGINAL frame pixels. x = column, y = row, origin top-left.
- Frame numbers are 0-based and used as STRINGS in JSON: "0", "25", "300".
- Time is never stored. Time in seconds = frame / fps.
- GAPS STAY GAPS: if you have no data for a frame, leave that frame out. If a grid cell has no
  data, mark it valid = 0. Never fill in, interpolate or repeat old values.
- Your model may run slower than the video (flow and density at about once per second is fine).
  You do not need an entry for every frame.
- Say your units honestly (see "units" below).
- One JSON file per layer per clip.

Every layer file starts with these fields:

  schema_version  always 1
  clip_id         the clip name, the same in every layer of that clip, e.g. "seq011"
  layer           your layer name: lowercase letters, digits, underscores, e.g. "density"
  type            "grid", "points" or "tracks"
  frame_size      [width, height] of the video in pixels, e.g. [1920, 1080]
  max_age_frames  how many video frames one entry stays on screen before it goes blank.
                  If you output once per second at 25 fps, use 25.
  frames          the data, keyed by frame number as a string

## Type 1: grid

For output that is one value per cell, such as density or flow. Extra fields:

  grid    {"rows": R, "cols": C}  the grid laid over the frame
  units   what the numbers mean, e.g. "people_per_cell" or "px_per_frame"
  fields  names of the matrices you provide, e.g. ["density"] or ["u", "v", "speed"]

Each entry in "frames" holds a "valid" matrix and one matrix per field. Every matrix is R rows of
C numbers. valid is 1 where the cell has a real observation and 0 where it does not (the numbers
in a valid=0 cell are ignored).

Example (density, 2 rows x 3 cols, one sample at frame 0 and one at frame 25):

  {
    "schema_version": 1, "clip_id": "seq011", "layer": "density", "type": "grid",
    "frame_size": [1920, 1080], "grid": {"rows": 2, "cols": 3},
    "units": "people_per_cell", "fields": ["density"], "max_age_frames": 25,
    "frames": {
      "0":  {"valid": [[1, 1, 0], [1, 1, 1]], "density": [[4, 9, 0], [2, 0, 1]]},
      "25": {"valid": [[1, 1, 1], [1, 1, 1]], "density": [[5, 8, 2], [3, 1, 0]]}
    }
  }

For flow, use fields ["u", "v", "speed"] and units "px_per_frame". u is movement to the right,
v is movement downward, in original-frame pixels per frame. When flow is converted to ground
speed the units will change to "m_per_s", so always fill in "units".

## Type 2: points

For detected head positions (localization). Each entry is a list of [x, y, score] rows.
score is between 0 and 1.

  {
    "schema_version": 1, "clip_id": "seq011", "layer": "points", "type": "points",
    "frame_size": [1920, 1080], "max_age_frames": 5,
    "frames": {"0": [[747.0, 592.0, 0.91], [760.5, 601.2, 0.77]], "5": [[748.2, 592.9, 0.90]]}
  }

## Type 3: tracks

For tracked people. Each entry is a list of [track_id, x, y, score] rows. track_id is a whole
number, unique within the clip, and never reused for a different person. Optional extra field:
"altitude_m" (the drone's height in metres).

  {
    "schema_version": 1, "clip_id": "seq011", "layer": "tracks", "type": "tracks",
    "frame_size": [1920, 1080], "altitude_m": 60.0, "max_age_frames": 5,
    "frames": {"0": [[7, 747.0, 592.0, 0.9], [8, 760.5, 601.2, 0.8]],
               "5": [[7, 748.2, 592.9, 0.9], [8, 761.1, 601.0, 0.8]]}
  }

## What to send Mahid

1. Your layer file(s), one per clip, named after the layer: density.json, points.json, ...
2. Your layer name, a short label for the toggle (e.g. "Density"), the type, and the units.
3. The clips you ran on, with their fps. Mahid will keep video and layers together in a folder:

     <clip_id>/clip.json          clip info: clip_id, fps, frame_size, n_frames, and a list of
                                  the layers (name, type, label, units)
     <clip_id>/video.mp4
     <clip_id>/layers/<layer>.json

   You do not have to write clip.json yourself; just tell Mahid the details.

   The video must be H.264 (yuv420p) or browsers will not play it. Most downloaded or phone
   videos already are; OpenCV's default "mp4v" writer is not.

## Check your file before sending it

Save the script at the bottom of this file as check_layer.py (it needs only Python 3, no
packages), then run:

  python check_layer.py density.json          check one layer
  python check_layer.py density.json 300      also check that no frame number is beyond the
                                              clip length (300 frames here)

It prints "ok", or lists what is wrong (wrong matrix size, point outside the frame, score above
1, duplicate track id, and so on).

## How it will be served (for information only)

The backend (Django) will offer these addresses. You do not need to build any of this.

  GET  /api/v1/clips                        list of clips and their layers
  GET  /api/v1/clips/<id>                   clip info and layer list
  GET  /api/v1/clips/<id>/video             the video
  GET  /api/v1/clips/<id>/layers/<layer>    your layer file
  POST /api/v1/clips/<id>/layers/<layer>    upload a layer (checked with the same rules)

The frontend downloads each layer once and shows the entry that matches the current video frame.

## Open questions

- What does the fifth model output? It should fit grid, points or tracks; if it does not, a
  fourth type is needed.
- Flow's real grid size, and whether flow also sends speed variance for the Tier 2 features.
- Ahmed: this extends docs/contracts.md (the points and tracks formats). Please confirm the
  formats above match what Models B and E will output.

## check_layer.py

```python
"""Check a Sentinel layer file. Standard library only.

    python check_layer.py layer.json [n_frames]
"""
import json
import re
import sys

NAME = re.compile(r"^[a-z][a-z0-9_]*$")


def num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def check_layer(doc, n_frames=None):
    """Return a list of problems; an empty list means the layer is valid."""
    errs = []
    kind = doc.get("type")
    if kind not in ("grid", "points", "tracks"):
        return [f"type must be grid, points or tracks, got {kind!r}"]
    if doc.get("schema_version") != 1:
        errs.append("schema_version must be 1")
    if not isinstance(doc.get("clip_id"), str) or not doc["clip_id"]:
        errs.append("clip_id must be a non-empty string")
    if not isinstance(doc.get("layer"), str) or not NAME.match(doc["layer"]):
        errs.append("layer must be lowercase letters, digits and underscores")
    fs = doc.get("frame_size")
    ok_size = isinstance(fs, list) and len(fs) == 2
    if not (ok_size and all(isinstance(v, int) and v > 0 for v in fs)):
        return errs + ["frame_size must be [width, height] in pixels"]
    w, h = fs
    age = doc.get("max_age_frames")
    if not (isinstance(age, int) and not isinstance(age, bool) and age >= 1):
        errs.append("max_age_frames must be an integer >= 1")
    frames = doc.get("frames")
    if not isinstance(frames, dict):
        return errs + ["frames must be an object keyed by frame number"]

    rows = cols = None
    fields = []
    if kind == "grid":
        g = doc.get("grid") or {}
        rows, cols = g.get("rows"), g.get("cols")
        if not (isinstance(rows, int) and isinstance(cols, int) and rows > 0 and cols > 0):
            errs.append("grid must have integer rows and cols")
            rows = cols = None
        fields = doc.get("fields")
        if not (isinstance(fields, list) and fields and all(isinstance(f, str) for f in fields)):
            errs.append("fields must be a non-empty list of names")
            fields = []
        if not isinstance(doc.get("units"), str):
            errs.append("units must be a string")

    for key, entry in frames.items():
        tag = f"frame {key}"
        if not key.isdigit():
            errs.append(f"{tag}: frame keys must be whole numbers as strings")
            continue
        if n_frames is not None and int(key) >= n_frames:
            errs.append(f"{tag}: beyond the clip length {n_frames}")
        if kind == "grid":
            if not isinstance(entry, dict):
                errs.append(f"{tag}: must be an object")
                continue
            for name in ["valid"] + fields:
                m = entry.get(name)
                if rows is None or not isinstance(m, list):
                    errs.append(f"{tag}: missing matrix '{name}'")
                elif len(m) != rows or any(not isinstance(r, list) or len(r) != cols for r in m):
                    errs.append(f"{tag}: '{name}' must be {rows}x{cols}")
                elif name == "valid" and any(v not in (0, 1) for r in m for v in r):
                    errs.append(f"{tag}: 'valid' may only hold 0 or 1")
                elif name != "valid" and any(not num(v) for r in m for v in r):
                    errs.append(f"{tag}: '{name}' must hold numbers")
            extra = set(entry) - set(["valid"] + fields)
            if extra:
                errs.append(f"{tag}: unexpected fields {sorted(extra)}")
        else:
            n = 3 if kind == "points" else 4
            if not isinstance(entry, list):
                errs.append(f"{tag}: must be a list of rows")
                continue
            ids = set()
            for row in entry:
                if not (isinstance(row, list) and len(row) == n and all(num(v) for v in row)):
                    errs.append(f"{tag}: each row needs {n} numbers")
                    break
                x, y = (row[0], row[1]) if kind == "points" else (row[1], row[2])
                if not (0 <= x < w and 0 <= y < h):
                    errs.append(f"{tag}: ({x:.0f}, {y:.0f}) is outside the {w}x{h} frame")
                    break
                if not 0 <= row[-1] <= 1:
                    errs.append(f"{tag}: score must be between 0 and 1")
                    break
                if kind == "tracks":
                    if row[0] != int(row[0]) or row[0] < 0 or row[0] in ids:
                        errs.append(f"{tag}: track_id must be a unique whole number >= 0")
                        break
                    ids.add(row[0])
    return errs


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    with open(sys.argv[1], encoding="utf-8") as fh:
        doc = json.load(fh)
    problems = check_layer(doc, int(sys.argv[2]) if len(sys.argv) > 2 else None)
    print("ok" if not problems else f"{len(problems)} problem(s):")
    for p in problems[:20]:
        print("  -", p)
    sys.exit(1 if problems else 0)
```
