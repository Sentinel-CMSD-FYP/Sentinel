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
    if kind not in ("grid", "points", "tracks", "polygons"):
        return [f"type must be grid, points, tracks or polygons, got {kind!r}"]
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
    classes = []
    if kind == "polygons":
        classes = doc.get("classes")
        if not (isinstance(classes, list) and classes and len(set(classes)) == len(classes)
                and all(isinstance(c, str) and c for c in classes)):
            errs.append("classes must be a non-empty list of unique names")
            classes = []

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
        elif kind == "polygons":
            errs += check_objects(entry, tag, classes, w, h)
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


def check_objects(entry, tag, classes, w, h):
    """Problems in one frame of a polygons layer (first problem per frame, like points)."""
    if not isinstance(entry, list):
        return [f"{tag}: must be a list of objects"]
    for obj in entry:
        if not (isinstance(obj, dict) and set(obj) == {"cls", "score", "box", "polygon"}):
            return [f"{tag}: each object needs exactly cls, score, box and polygon"]
        if obj["cls"] not in classes:
            return [f"{tag}: cls {obj['cls']!r} is not one of classes"]
        if not (num(obj["score"]) and 0 <= obj["score"] <= 1):
            return [f"{tag}: score must be between 0 and 1"]
        box = obj["box"]
        if not (isinstance(box, list) and len(box) == 4 and all(num(v) for v in box)
                and 0 <= box[0] <= box[2] <= w and 0 <= box[1] <= box[3] <= h):
            return [f"{tag}: box must be [x0, y0, x1, y1] inside the {w}x{h} frame"]
        poly = obj["polygon"]
        if not (isinstance(poly, list) and len(poly) >= 3
                and all(isinstance(p, list) and len(p) == 2 and all(num(v) for v in p)
                        for p in poly)):
            return [f"{tag}: polygon needs at least 3 [x, y] points"]
        if not all(0 <= x < w and 0 <= y < h for x, y in poly):
            return [f"{tag}: polygon has a point outside the {w}x{h} frame"]
    return []


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
