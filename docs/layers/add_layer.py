"""Add a model's layer file to a clip folder, or remove one, keeping clip.json in step.

A clip folder holds ``clip.json`` (which lists the layers), ``video.mp4`` and ``layers/*.json``.
Copying a layer in by hand and editing the list is easy to get wrong, so this does it and
checks the layer against the clip first. Standard library only.

    python docs/layers/add_layer.py data/samples/market_demo density.json --label "Density"
    python docs/layers/add_layer.py data/samples/market_demo density.json --replace
    python docs/layers/add_layer.py data/samples/market_demo --remove density

The layer is refused, and nothing is written, if it is not valid under the layer contract, if it
was made for a different video (different ``clip_id`` or ``frame_size``), if it uses frame numbers
beyond the clip, or if that layer already exists (unless ``--replace``).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from check_layer import check_layer  # noqa: E402

NAME = re.compile(r"^[a-z][a-z0-9_]*$")


class LayerError(Exception):
    """A problem the user can fix; its message says how."""


def _load(path: Path, what: str) -> dict:
    if not path.is_file():
        raise LayerError(f"{what} not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as e:
        raise LayerError(f"{what} is not valid JSON ({path}): {e}") from e
    if not isinstance(data, dict):
        raise LayerError(f"{what} must be a JSON object: {path}")
    return data


def _write(path: Path, text: str) -> None:
    """Write via a temporary file so a crash cannot leave a half-written file behind."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def manifest_entry(doc: dict, label: str | None, visible: bool | None, field: str | None) -> dict:
    """The clip.json line that tells the viewer how to show this layer."""
    name = doc["layer"]
    entry: dict = {
        "layer": name,
        "type": doc["type"],
        "label": label or name.replace("_", " ").title(),
    }
    if doc["type"] == "grid":
        fields = doc["fields"]
        entry["units"] = doc["units"]
        if field is not None and field not in fields:
            raise LayerError(f"--field {field!r} is not one of this layer's fields {fields}")
        if field is None and "u" in fields and "v" in fields:
            entry["style"] = {"render": "arrows", "u": "u", "v": "v"}
        else:
            entry["style"] = {"render": "heatmap", "field": field or fields[0]}
    # Grids start visible; dots and tracks start hidden because they crowd the picture.
    entry["default_visible"] = visible if visible is not None else doc["type"] == "grid"
    return entry


def add_layer(
    clip_dir: Path,
    layer_file: Path,
    *,
    name: str | None = None,
    label: str | None = None,
    visible: bool | None = None,
    field: str | None = None,
    replace: bool = False,
    fix_clip_id: bool = False,
    dry_run: bool = False,
) -> str:
    clip_dir = Path(clip_dir)
    clip = _load(clip_dir / "clip.json", "clip.json")
    doc = _load(Path(layer_file), "layer file")

    name = name or doc.get("layer")
    if not isinstance(name, str) or not NAME.match(name):
        raise LayerError(f"layer name {name!r} must be lowercase letters, digits and underscores")
    doc["layer"] = name

    if doc.get("clip_id") != clip["clip_id"]:
        if not fix_clip_id:
            raise LayerError(
                f"the layer says clip_id {doc.get('clip_id')!r} but this clip is "
                f"{clip['clip_id']!r}. If it was made on this clip's video, add --fix-clip-id."
            )
        doc["clip_id"] = clip["clip_id"]

    if doc.get("frame_size") != clip["frame_size"]:
        raise LayerError(
            f"the layer's frame_size {doc.get('frame_size')} differs from this clip's video "
            f"{clip['frame_size']}: it was made on a different video or resolution. Run the model "
            f"on {clip_dir / 'video.mp4'} itself, so frame numbers and pixel positions line up."
        )

    problems = check_layer(doc, clip["n_frames"])
    if problems:
        shown = "\n  - ".join(problems[:10])
        more = f"\n  ... and {len(problems) - 10} more" if len(problems) > 10 else ""
        raise LayerError(f"the layer is not valid ({len(problems)} problems):\n  - {shown}{more}")

    target = clip_dir / "layers" / f"{name}.json"
    layers = clip.get("layers", [])
    at = next((i for i, e in enumerate(layers) if e.get("layer") == name), None)
    if (at is not None or target.exists()) and not replace:
        raise LayerError(
            f"layer {name!r} already exists in this clip; use --replace to overwrite it"
        )

    entry = manifest_entry(doc, label, visible, field)
    if at is None:
        layers.append(entry)
    else:
        layers[at] = entry
    clip["layers"] = layers

    verb = "replaced" if at is not None else "added"
    summary = (f"{verb} layer {name!r} ({doc['type']}, {len(doc['frames'])} samples) "
               f"in {clip_dir}; layers now: {', '.join(e['layer'] for e in layers)}")
    if dry_run:
        return "dry run, nothing written: would have " + summary
    _write(target, json.dumps(doc))
    _write(clip_dir / "clip.json", json.dumps(clip, indent=2))
    return summary


def remove_layer(clip_dir: Path, name: str, *, dry_run: bool = False) -> str:
    clip_dir = Path(clip_dir)
    clip = _load(clip_dir / "clip.json", "clip.json")
    layers = clip.get("layers", [])
    target = clip_dir / "layers" / f"{name}.json"
    if not any(e.get("layer") == name for e in layers) and not target.exists():
        raise LayerError(f"layer {name!r} is not in this clip")
    clip["layers"] = [e for e in layers if e.get("layer") != name]
    summary = f"removed layer {name!r} from {clip_dir}; layers now: " + (
        ", ".join(e["layer"] for e in clip["layers"]) or "none")
    if dry_run:
        return "dry run, nothing written: would have " + summary
    target.unlink(missing_ok=True)
    _write(clip_dir / "clip.json", json.dumps(clip, indent=2))
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("clip_dir", type=Path, help="the clip folder (holds clip.json)")
    ap.add_argument("layer_file", type=Path, nargs="?", help="the layer JSON to add")
    ap.add_argument("--remove", metavar="NAME", help="remove this layer instead of adding one")
    ap.add_argument("--name", help="layer name (default: the 'layer' field of the file)")
    ap.add_argument("--label", help="name shown in the viewer (default: the layer name)")
    vis = ap.add_mutually_exclusive_group()
    vis.add_argument("--visible", dest="visible", action="store_true", default=None,
                     help="switched on when the clip opens (default for grids)")
    vis.add_argument("--hidden", dest="visible", action="store_false",
                     help="switched off when the clip opens (default for points and tracks)")
    ap.add_argument("--field", help="grid layers: which field a heatmap shows")
    ap.add_argument("--replace", action="store_true", help="overwrite a layer that exists")
    ap.add_argument("--fix-clip-id", action="store_true",
                    help="rewrite the layer's clip_id to match this clip")
    ap.add_argument("--dry-run", action="store_true", help="check and report, write nothing")
    a = ap.parse_args(argv)

    try:
        if a.remove:
            if a.layer_file:
                ap.error("give either a layer file or --remove, not both")
            print(remove_layer(a.clip_dir, a.remove, dry_run=a.dry_run))
        else:
            if not a.layer_file:
                ap.error("give a layer file to add, or --remove NAME")
            print(add_layer(a.clip_dir, a.layer_file, name=a.name, label=a.label,
                            visible=a.visible, field=a.field, replace=a.replace,
                            fix_clip_id=a.fix_clip_id, dry_run=a.dry_run))
    except LayerError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
