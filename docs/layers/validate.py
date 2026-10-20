"""Validate clip metadata and layer files against the layer contract.

Schemas catch structure; the checks below catch what JSON Schema cannot, such as a
matrix that does not match ``grid.rows x grid.cols`` or points outside the frame.
The backend ingestion endpoint is meant to call ``validate_layer``.

    python docs/layers/validate.py data/samples/demo_fake
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from jsonschema import Draft202012Validator

SCHEMAS = Path(__file__).parent / "schemas"
_LAYER_SCHEMA = {
    "grid": "layer_grid.schema.json",
    "points": "layer_points.schema.json",
    "tracks": "layer_tracks.schema.json",
    "polygons": "layer_polygons.schema.json",
}


def _schema_errors(doc: dict, schema_file: str) -> list[str]:
    schema = json.loads((SCHEMAS / schema_file).read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    return [f"{'/'.join(map(str, e.absolute_path)) or '<root>'}: {e.message}"
            for e in validator.iter_errors(doc)]


def validate_clip(doc: dict) -> list[str]:
    return _schema_errors(doc, "clip.schema.json")


def validate_layer(doc: dict, n_frames: int | None = None) -> list[str]:
    """Return a list of problems; an empty list means the layer is valid."""
    kind = doc.get("type")
    if kind not in _LAYER_SCHEMA:
        return [f"type must be one of {sorted(_LAYER_SCHEMA)}, got {kind!r}"]
    errors = _schema_errors(doc, _LAYER_SCHEMA[kind])
    if errors:
        return errors

    w, h = doc["frame_size"]
    for fid, entry in doc["frames"].items():
        if n_frames is not None and int(fid) >= n_frames:
            errors.append(f"frame {fid}: beyond clip length {n_frames}")

        if kind == "grid":
            rows, cols = doc["grid"]["rows"], doc["grid"]["cols"]
            for name in ("valid", *doc["fields"]):
                m = entry.get(name)
                if m is None:
                    errors.append(f"frame {fid}: missing '{name}'")
                elif len(m) != rows or any(len(r) != cols for r in m):
                    errors.append(f"frame {fid}: '{name}' is not {rows}x{cols}")
            extra = set(entry) - {"valid", *doc["fields"]}
            if extra:
                errors.append(f"frame {fid}: unexpected fields {sorted(extra)}")
        elif kind == "polygons":
            for obj in entry:
                x0, y0, x1, y1 = obj["box"]
                if obj["cls"] not in doc["classes"]:
                    errors.append(f"frame {fid}: cls {obj['cls']!r} is not one of classes")
                    break
                if not (0 <= x0 <= x1 <= w and 0 <= y0 <= y1 <= h):
                    errors.append(f"frame {fid}: box {obj['box']} outside {w}x{h}")
                    break
                if not all(0 <= x < w and 0 <= y < h for x, y in obj["polygon"]):
                    errors.append(f"frame {fid}: polygon point outside {w}x{h}")
                    break
        else:
            xi = 0 if kind == "points" else 1
            for row in entry:
                x, y = row[xi], row[xi + 1]
                if not (0 <= x < w and 0 <= y < h):
                    errors.append(f"frame {fid}: ({x:.0f}, {y:.0f}) outside {w}x{h}")
                    break
            if kind == "tracks":
                ids = [r[0] for r in entry]
                if len(ids) != len(set(ids)):
                    errors.append(f"frame {fid}: duplicate track ids")
    return errors


def validate_clip_dir(root: Path) -> dict[str, list[str]]:
    """Validate ``clip.json`` and every file in ``layers/`` under ``root``."""
    clip = json.loads((root / "clip.json").read_text(encoding="utf-8"))
    problems = {"clip.json": validate_clip(clip)}
    declared = {entry["layer"]: entry["type"] for entry in clip.get("layers", [])}
    for f in sorted((root / "layers").glob("*.json")):
        doc = json.loads(f.read_text(encoding="utf-8"))
        errs = validate_layer(doc, clip.get("n_frames"))
        if doc.get("layer") != f.stem:
            errs.append(f"layer name {doc.get('layer')!r} does not match file name {f.stem!r}")
        if declared.get(f.stem) != doc.get("type"):
            errs.append("not declared in clip.json, or declared with a different type")
        if doc.get("clip_id") != clip.get("clip_id"):
            errs.append("clip_id differs from clip.json")
        problems[f"layers/{f.name}"] = errs
    return problems


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    results = validate_clip_dir(Path(sys.argv[1]))
    for name, errs in results.items():
        print(f"{'ok  ' if not errs else 'FAIL'} {name}")
        for e in errs[:10]:
            print(f"       {e}")
    sys.exit(1 if any(results.values()) else 0)
