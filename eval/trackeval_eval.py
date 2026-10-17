"""IDF1, HOTA, MOTA and ID switches with TrackEval, from our tracks (val first).

Tracks are converted to MOT text with 20x20 boxes centred on each point (the authors' evaluation
does the same), the ground truth is built from the .mat files in the same format (same ids as the
authors' MOT GT), and TrackEval's ``MotChallenge2DBox`` is run with ``DO_PREPROC`` off.

Metric threshold (the IoU a prediction needs to match a GT box in CLEAR and Identity): the
authors' evaluation script uses 0.0. The sanity checks show that this makes IDF1 and ID switches
meaningless (with randomly shuffled ids, IDF1 stays 1.0 and the switches are about 1000 times too few),
because at 0.0 every pair of boxes counts as a potential match. So the primary threshold here is the
standard 0.5 (for 20x20 boxes about a 6.7 px offset along one axis) and the secondary one is 0.3
(about a 10 px offset, like the F1 matching of Model B). HOTA does not use this threshold.

Each tracker is scored unfiltered and with ids kept only if they were output in at least
``min_len_frames`` frames (the offline filter of decision 0006). ``--sanity`` first runs the checks
that show TrackEval and the converters behave: ground truth as prediction must score 1, a consistent
relabelling must change nothing, shuffled ids must keep DetA but wreck IDF1 and add ID switches.

Units: pixels and frames. Coordinates: x = column, y = row, original frame pixels.
Val results are optimistic (B trained on those clips). The test split needs ``--allow-test``.

    python eval/trackeval_eval.py --config ml/configs/trackeval.yaml --split val --sanity
    python eval/trackeval_eval.py --config ml/configs/trackeval.yaml --split val
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

from sentinel_ml.localization import data as D
from sentinel_ml.tracking import TrackerConfig
from sentinel_ml.tracking.run_tracker import track_clip

BOX = 20  # box side in pixels (the points are the centres of 20x20 boxes)
BENCH = "MOT17-test"  # folder name TrackEval expects for BENCHMARK MOT17 and split test


# ---------- conversion to MOT text (pure functions, tested without TrackEval) ----------

def gt_rows(gt: dict[int, tuple[np.ndarray, np.ndarray]]) -> list[list[float]]:
    """MOT rows ``frame, id, x, y, w, h, 1, -1, -1, -1`` from ``{frame_id: (points, ids)}``.

    ``frame`` is 1-based, (x, y) the top-left corner of the 20x20 box around each point.
    """
    rows = []
    for fid in sorted(gt):
        pts, ids = gt[fid]
        for (x, y), pid in zip(pts, ids, strict=True):
            rows.append([fid + 1, int(pid), float(x) - BOX / 2, float(y) - BOX / 2, BOX, BOX,
                         1, -1, -1, -1])
    return rows


def tracks_rows(doc: dict, min_len: int | None = None) -> list[list[float]]:
    """MOT rows from a tracks document. With ``min_len``, ids output in fewer frames are dropped."""
    lengths = Counter(r[0] for rows in doc["frames"].values() for r in rows)
    rows = []
    for fid in sorted(doc["frames"], key=int):
        for tid, x, y, score in doc["frames"][fid]:
            if min_len is None or lengths[tid] >= min_len:
                rows.append([int(fid) + 1, tid, x - BOX / 2, y - BOX / 2, BOX, BOX, score,
                             -1, -1, -1])
    return rows


def oracle_rows(points_doc: dict, gt: dict, score_thr: float, radius: float = 10.0
                ) -> list[list[float]]:
    """Perfect association on Model B's points: the upper bound for any tracker.

    Keeps the detections with score >= ``score_thr``, gives each one matched to a GT head (within
    ``radius`` px, one-to-one, nearest first) that head's true id, and a unique new id to every
    unmatched detection. Only the association is perfect; missed people stay missed.
    """
    import point_metrics as pm  # eval/ is on sys.path when run as a script

    rows, fresh = [], 10**6
    for key in sorted(points_doc["frames"], key=int):
        fid = int(key)
        pts = np.asarray(points_doc["frames"][key], dtype=float).reshape(-1, 3)
        pts = pts[pts[:, 2] >= score_thr]
        xy, ids = gt[fid]
        match = dict(pm.match_points(pts[:, :2], xy, radius)) if len(pts) else {}
        for k, (x, y, score) in enumerate(pts):
            if k in match:
                tid = int(ids[match[k]])
            else:
                tid, fresh = fresh, fresh + 1
            rows.append([fid + 1, tid, x - BOX / 2, y - BOX / 2, BOX, BOX, score, -1, -1, -1])
    return rows


def write_mot(rows: list[list[float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for r in rows:
        head = f"{int(r[0])},{int(r[1])}"
        box = ",".join(f"{v:g}" for v in r[2:6])
        lines.append(f"{head},{box},{r[6]:g},-1,-1,-1")
    path.write_text(chr(10).join(lines) + (chr(10) if lines else ""))


def write_gt(root: Path, seq: str, rows: list[list[float]], n_frames: int) -> None:
    """GT file and seqinfo.ini in the layout TrackEval expects (same as the authors')."""
    base = root / "gt" / BENCH / seq
    write_mot(rows, base / "gt" / "gt.txt")
    (base / "seqinfo.ini").write_text("[Sequence]" + chr(10) + f"seqlength = {n_frames}")


def write_seqmap(root: Path, seqs: list[str]) -> Path:
    path = root / "gt" / "seqmap.txt"
    path.write_text("name" + chr(10) + chr(10).join(seqs) + chr(10))
    return path


# ---------- running TrackEval ----------

def load_trackeval(path: Path):
    """Import the TrackEval copy at ``path`` (raises ImportError if it is not there)."""
    if not (Path(path) / "trackeval").exists():
        raise ImportError(f"TrackEval not found at {path}")
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    import trackeval

    return trackeval


def run_trackeval(trackeval, root: Path, trackers: list[str], seqmap: Path,
                  threshold: float) -> dict:
    """Run HOTA, CLEAR and Identity for every tracker folder. Returns per-tracker summaries."""
    eval_cfg = trackeval.Evaluator.get_default_eval_config()
    eval_cfg.update(DISPLAY_LESS_PROGRESS=True, PRINT_RESULTS=False, OUTPUT_SUMMARY=False,
                    OUTPUT_EMPTY_CLASSES=False, OUTPUT_DETAILED=False, PLOT_CURVES=False,
                    TIME_PROGRESS=False, USE_PARALLEL=False, PRINT_CONFIG=False)
    data_cfg = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    data_cfg.update(GT_FOLDER=str(root / "gt"), TRACKERS_FOLDER=str(root / "trackers"),
                    SEQMAP_FILE=str(seqmap), SPLIT_TO_EVAL="test", DO_PREPROC=False,
                    PRINT_CONFIG=False, TRACKERS_TO_EVAL=trackers)
    mcfg = {"THRESHOLD": threshold, "PRINT_CONFIG": False}
    metrics = [trackeval.metrics.HOTA(dict(mcfg)), trackeval.metrics.CLEAR(dict(mcfg)),
               trackeval.metrics.Identity(dict(mcfg))]
    res, _ = trackeval.Evaluator(eval_cfg).evaluate(
        [trackeval.datasets.MotChallenge2DBox(data_cfg)], metrics)
    out = {}
    for name, per_seq in res["MotChallenge2DBox"].items():
        out[name] = summarise(per_seq)
    return out


def _flat(seq_res: dict) -> dict:
    p = seq_res["pedestrian"]
    return {"HOTA": float(np.mean(p["HOTA"]["HOTA"])), "DetA": float(np.mean(p["HOTA"]["DetA"])),
            "AssA": float(np.mean(p["HOTA"]["AssA"])), "IDF1": float(p["Identity"]["IDF1"]),
            "MOTA": float(p["CLEAR"]["MOTA"]), "IDSW": int(p["CLEAR"]["IDSW"]),
            "GT_IDs": int(p["Count"]["GT_IDs"]), "IDs": int(p["Count"]["IDs"]),
            "TP": int(p["CLEAR"]["CLR_TP"]), "FP": int(p["CLEAR"]["CLR_FP"]),
            "FN": int(p["CLEAR"]["CLR_FN"])}


def summarise(per_seq: dict) -> dict:
    """Per-sequence values, their mean over sequences, and TrackEval's combined values."""
    seqs = {s: _flat(r) for s, r in per_seq.items() if s != "COMBINED_SEQ"}
    mean = {k: float(np.mean([v[k] for v in seqs.values()])) for k in
            ("HOTA", "DetA", "AssA", "IDF1", "MOTA")}
    total = {k: int(sum(v[k] for v in seqs.values())) for k in
             ("IDSW", "GT_IDs", "IDs", "TP", "FP", "FN")}
    return {"clips": len(seqs), "mean_over_clips": mean, "totals": total,
            "combined": _flat(per_seq["COMBINED_SEQ"]), "per_clip": seqs}


# ---------- sanity checks ----------

def shuffle_ids(gt: dict, rng: np.random.Generator) -> dict:
    """Same boxes, but the ids are permuted among the objects of each frame."""
    return {fid: (pts, rng.permutation(ids)) for fid, (pts, ids) in gt.items()}


def relabel_ids(gt: dict, offset: int = 1000) -> dict:
    """Same boxes and a consistent new name for every id: nothing may change."""
    return {fid: (pts, ids + offset) for fid, (pts, ids) in gt.items()}


def drop_half(gt: dict, rng: np.random.Generator) -> dict:
    """Same ids, but a random half of the objects of each frame is missing."""
    out = {}
    for fid, (pts, ids) in gt.items():
        keep = rng.permutation(len(ids))[: len(ids) // 2]
        out[fid] = (pts[keep], ids[keep])
    return out


def gt_as_tracker(gt: dict) -> list[list[float]]:
    """GT rows written as a tracker output (score 1)."""
    return gt_rows(gt)


def sanity_variants(gt_by_clip: dict[int, dict], seed: int = 0) -> dict[str, dict[int, dict]]:
    rng = np.random.default_rng(seed)
    return {
        "gt_as_prediction": gt_by_clip,
        "consistent_relabel": {c: relabel_ids(g) for c, g in gt_by_clip.items()},
        "shuffled_ids": {c: shuffle_ids(g, rng) for c, g in gt_by_clip.items()},
        "half_detections_dropped": {c: drop_half(g, rng) for c, g in gt_by_clip.items()},
    }


def build_dataset(root: Path, gt_by_clip: dict[int, dict], trackers: dict[str, dict[int, list]],
                  threshold_dir: str = "") -> Path:
    """Write GT and every tracker's MOT files under ``root``. Returns the seqmap path."""
    if root.exists():
        shutil.rmtree(root)
    seqs = [f"{c:03d}" for c in sorted(gt_by_clip)]
    for c in sorted(gt_by_clip):
        gt = gt_by_clip[c]
        write_gt(root, f"{c:03d}", gt_rows(gt), max(gt) + 1)
    for name, per_clip in trackers.items():
        for c, rows in per_clip.items():
            write_mot(rows, root / "trackers" / BENCH / name / "data" / f"{c:03d}.txt")
    return write_seqmap(root, seqs)


def run_sanity(trackeval, gt_by_clip: dict[int, dict], root: Path, threshold: float) -> dict:
    variants = sanity_variants(gt_by_clip)
    trackers = {name: {c: gt_as_tracker(g) for c, g in per.items()}
                for name, per in variants.items()}
    seqmap = build_dataset(root, gt_by_clip, trackers)
    return run_trackeval(trackeval, root, list(trackers), seqmap, threshold)


def format_summary(name: str, s: dict) -> str:
    m, t, c = s["mean_over_clips"], s["totals"], s["combined"]
    return (f"{name:34s} HOTA {m['HOTA']:.3f} DetA {m['DetA']:.3f} AssA {m['AssA']:.3f} "
            f"IDF1 {m['IDF1']:.3f} MOTA {m['MOTA']:.3f} | IDSW {t['IDSW']} IDs {t['IDs']} "
            f"(GT {t['GT_IDs']}) | combined IDF1 {c['IDF1']:.3f} HOTA {c['HOTA']:.3f}")


# ---------- CLI ----------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--split", required=True, choices=["val", "test"])
    ap.add_argument("--sanity", action="store_true", help="run the sanity checks, then stop")
    ap.add_argument("--oracle", action="store_true",
                    help="also score the perfect-association oracle on B's points")
    ap.add_argument("--allow-test", action="store_true", help="needed for the test split")
    args = ap.parse_args()
    if args.split == "test" and not args.allow_test:
        raise SystemExit("the test split needs --allow-test (and counts as another look at test)")

    cfg = yaml.safe_load(args.config.read_text())
    trackeval = load_trackeval(Path(cfg["trackeval_path"]))
    data_cfg = D.load_config(Path(cfg["data_config"]))
    points_dir = Path(cfg["points_dir"]) / args.split
    docs = {int(f.name.split("_")[1]): json.loads(f.read_text())
            for f in sorted(points_dir.glob("clip_*_points.json"))}
    gts = {c: D.clip_gt_frames(data_cfg, args.split, c, sorted(int(k) for k in d["frames"]))
           for c, d in docs.items()}
    out_dir = Path(cfg["out_dir"]) / args.split
    key = "sanity_thresholds" if args.sanity else "thresholds"
    thresholds = [(f"iou_{t:g}", float(t)) for t in cfg[key]]
    result: dict = {"split": args.split, "clips": sorted(docs),
                    "note": "val is optimistic: B trained on these clips"}
    if args.sanity:
        sanity_clips = {c: gts[c] for c in sorted(gts)[: int(cfg["sanity_clips"])]}
        for label, thr in thresholds:
            res = run_sanity(trackeval, sanity_clips, out_dir / "sanity_work", thr)
            result[f"sanity_{label}"] = res
            print(f"--- sanity checks, {len(sanity_clips)} clips {sorted(sanity_clips)}, {label}")
            for name, s in res.items():
                print(format_summary(name, s))
    else:
        trackers: dict[str, dict[int, list]] = {}
        for name, path in cfg["trackers"].items():
            tcfg = TrackerConfig(**yaml.safe_load(Path(path).read_text())["tracker"])
            tdocs = {c: track_clip(d, tcfg) for c, d in docs.items()}
            trackers[name] = {c: tracks_rows(t) for c, t in tdocs.items()}
            trackers[f"{name}_min{cfg['min_len_frames']}"] = {
                c: tracks_rows(t, int(cfg["min_len_frames"])) for c, t in tdocs.items()}
        if args.oracle:
            for score_thr in cfg["oracle_score_thr"]:
                trackers[f"oracle_score_{score_thr:g}"] = {
                    c: oracle_rows(docs[c], gts[c], float(score_thr)) for c in docs}
        seqmap = build_dataset(out_dir / "work", gts, trackers)
        for label, thr in thresholds:
            res = run_trackeval(trackeval, out_dir / "work", list(trackers), seqmap, thr)
            result[label] = res
            print(f"--- {args.split} split, {len(docs)} clips, {label} (val is optimistic)")
            for name, s in res.items():
                print(format_summary(name, s))
    out_dir.mkdir(parents=True, exist_ok=True)
    name = "sanity.json" if args.sanity else ("results_oracle.json" if args.oracle
                                              else "results.json")
    (out_dir / name).write_text(json.dumps(result, indent=1) + chr(10))
    print(f"wrote {out_dir / name}")


if __name__ == "__main__":
    main()
