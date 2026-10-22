"""Fingerprint a DroneCrowd copy, and compare two fingerprints.

Used to check that the public Kaggle copy (onvithong/dronecrowddata) holds the
same files as our local copy from the official Google Drive links. Split folders
are found by name (``train_data``, ``test_data``, ``val_data`` holding ``images/``
and ``ground_truth/``), so the different folder nesting of the two copies does not
matter.

    python eval/density_dmcount/dataset_fingerprint.py make data/dronecrowd --out runs/fp_local.json
    python eval/density_dmcount/dataset_fingerprint.py compare runs/fp_local.json runs/fp_kaggle.json

The fingerprint holds per split frame and GT counts, the md5 of every GT file, the
md5 of a fixed image sample (frames 1, 150 and 300 of every clip) and the md5 of
``trainlist.txt`` / ``testlist.txt`` when present.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

SPLITS = ("train_data", "test_data", "val_data")
FRAME_RE = re.compile(r"^img(\d{3})(\d{3})\.jpg$")
SAMPLE_FRAMES = (1, 150, 300)


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_split_dirs(root: Path) -> dict[str, Path]:
    """Map split name -> folder holding ``images/`` and ``ground_truth/``."""
    found: dict[str, Path] = {}
    for img_dir in sorted(root.rglob("images")):
        d = img_dir.parent
        if not (d / "ground_truth").is_dir():
            continue
        for name in SPLITS:
            if name in d.name and name not in found:
                found[name] = d
    return found


def make(root: Path) -> dict:
    out: dict = {"root": str(root), "splits": {}, "lists": {}}
    for name, d in find_split_dirs(root).items():
        frames = sorted(p.name for p in (d / "images").iterdir() if FRAME_RE.match(p.name))
        gts = sorted((d / "ground_truth").glob("GT_img*.mat"))
        clips = sorted({int(FRAME_RE.match(f).group(1)) for f in frames})
        sample = [f"img{c:03d}{fr:03d}.jpg" for c in clips for fr in SAMPLE_FRAMES]
        out["splits"][name] = {
            "dir": str(d.relative_to(root)),
            "n_clips": len(clips),
            "n_frames": len(frames),
            "n_gt": len(gts),
            "gt_md5": {p.name: md5(p) for p in gts},
            "img_md5": {f: md5(d / "images" / f) for f in sample if (d / "images" / f).exists()},
        }
        print(f"{name}: dir={d.relative_to(root)} clips={len(clips)} frames={len(frames)} gt={len(gts)}")
    for lst in ("trainlist.txt", "testlist.txt"):
        hits = sorted(root.rglob(lst))
        if hits:
            out["lists"][lst] = md5(hits[0])
    print("lists:", out["lists"] or "none found")
    return out


def compare(a: dict, b: dict) -> bool:
    ok = True
    for name in sorted(set(a["splits"]) | set(b["splits"])):
        sa, sb = a["splits"].get(name), b["splits"].get(name)
        if not sa or not sb:
            print(f"{name}: only in {'A' if sa else 'B'}")
            ok = False
            continue
        for key in ("n_clips", "n_frames", "n_gt"):
            same = sa[key] == sb[key]
            ok &= same
            print(f"{name} {key}: A={sa[key]} B={sb[key]}{'' if same else '  MISMATCH'}")
        for key in ("gt_md5", "img_md5"):
            common = set(sa[key]) & set(sb[key])
            diff = sorted(k for k in common if sa[key][k] != sb[key][k])
            missing = len(set(sa[key]) ^ set(sb[key]))
            ok &= not diff and not missing
            print(f"{name} {key}: {len(common)} compared, {len(diff)} differ, {missing} only on one side"
                  + (f" e.g. {diff[:3]}" if diff else ""))
    for lst in sorted(set(a["lists"]) | set(b["lists"])):
        same = a["lists"].get(lst) == b["lists"].get(lst)
        print(f"{lst}: {'same' if same else 'DIFFERENT or missing'}")
    print("RESULT:", "identical" if ok else "NOT identical")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("root", type=Path)
    m.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("compare")
    c.add_argument("a", type=Path)
    c.add_argument("b", type=Path)
    args = ap.parse_args()
    if args.cmd == "make":
        fp = make(args.root)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(fp))
    else:
        ok = compare(json.loads(args.a.read_text()), json.loads(args.b.read_text()))
        raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
