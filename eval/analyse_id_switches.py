"""IM-4: why TrackEval counts hundreds of ID switches on clips that look stable by eye.

Read-only analysis of the frozen run-2 tracks on DroneCrowd test clips (no tracker setting is
changed, nothing is tuned). It re-implements the CLEAR matching of TrackEval exactly (one-to-one
Hungarian matching on IoU of the 20x20 point boxes, with the previous-frame id preferred), checks
that its ID-switch count equals TrackEval's, and then explains the switches.

TrackEval counts an ID switch whenever the id matched to a ground-truth person differs from the id
last matched to that person, **even after a gap of any length**. So a track that dies and a new
track that starts on the same person is one "switch". Each switch is put in exactly one class:

  real_swap     the new id was, within ``SWAP_WINDOW`` frames, matched to a DIFFERENT person
  flip_back     the person returns to the id they had before the last one (A, B, A)
  new_id_direct consecutive frames, a new id takes over the same person (a duplicate track or a
                hand-over between two tracks on one head)
  after_gap     the person was unmatched for one or more frames, then matched with a new id

Units: frames (1-based, as in the MOT files), pixels (x = column, y = row), seconds = frame / fps.
    python eval/analyse_id_switches.py --config ml/configs/id_switches.yaml
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import linear_sum_assignment

SWAP_WINDOW = 5  # frames: an id seen on another person this recently counts as a takeover
CLASSES = ("real_swap", "flip_back", "new_id_direct", "after_gap")


def read_mot(path: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """MOT text -> ``{frame: (ids (n,), boxes (n, 4) as x, y, w, h in px)}``."""
    rows: dict[int, list[list[float]]] = defaultdict(list)
    for line in path.read_text().splitlines():
        if line.strip():
            f = line.split(",")
            rows[int(f[0])].append([float(v) for v in f[1:6]])
    return {t: (np.array([int(r[0]) for r in v]), np.array([r[1:5] for r in v]))
            for t, v in rows.items()}


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between boxes ``a`` (n, 4) and ``b`` (m, 4), each as x, y, w, h in pixels."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ax2, ay2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx2, by2 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.minimum(ax2[:, None], bx2[None]) - np.maximum(a[:, 0, None], b[None, :, 0])
    ih = np.minimum(ay2[:, None], by2[None]) - np.maximum(a[:, 1, None], b[None, :, 1])
    iw, ih = np.clip(iw, 0, None), np.clip(ih, 0, None)
    inter = iw * ih
    union = (a[:, 2] * a[:, 3])[:, None] + (b[:, 2] * b[:, 3])[None] - inter
    return inter / np.maximum(union, 1e-12)


def clear_match(gt: dict, trk: dict, threshold: float) -> list[dict]:
    """Per-frame matches exactly as TrackEval's CLEAR does. Returns a list of
    ``{"t", "gt", "trk", "iou"}`` (one per match), frames in increasing order."""
    eps = np.finfo(float).eps
    prev_ts: dict[int, int] = {}  # gt id -> tracker id matched in the previous frame
    out = []
    for t in sorted(set(gt) | set(trk)):
        if t not in gt or t not in trk:
            prev_ts = {}
            continue
        gids, gb = gt[t]
        tids, tb = trk[t]
        sim = iou_matrix(gb, tb)
        prev = np.array([[prev_ts.get(int(g), -1) == int(k) for k in tids] for g in gids])
        score = 1000 * prev + sim
        score[sim < threshold - eps] = 0
        rows, cols = linear_sum_assignment(-score)
        keep = score[rows, cols] > eps
        rows, cols = rows[keep], cols[keep]
        prev_ts = {int(gids[r]): int(tids[c]) for r, c in zip(rows, cols, strict=True)}
        out += [{"t": t, "gt": int(gids[r]), "trk": int(tids[c]), "iou": float(sim[r, c])}
                for r, c in zip(rows, cols, strict=True)]
    return out


def find_switches(matches: list[dict]) -> list[dict]:
    """ID switches (TrackEval's rule) with a class each. ``matches`` as from :func:`clear_match`."""
    last_trk: dict[int, int] = {}        # gt -> last matched tracker id
    last_t: dict[int, int] = {}          # gt -> frame of that match
    history: dict[int, list[int]] = {}   # gt -> run-length-compressed tracker id history
    owner: dict[int, tuple[int, int]] = {}  # tracker id -> (gt, frame) of its latest match
    switches = []
    for m in matches:
        g, k, t = m["gt"], m["trk"], m["t"]
        if g in last_trk and k != last_trk[g]:
            other = owner.get(k)
            if other is not None and other[0] != g and t - other[1] <= SWAP_WINDOW:
                cls = "real_swap"
            elif len(history[g]) >= 2 and history[g][-2] == k:
                cls = "flip_back"
            elif t - last_t[g] == 1:
                cls = "new_id_direct"
            else:
                cls = "after_gap"
            switches.append({"t": t, "gt": g, "old": last_trk[g], "new": k, "gap": t - last_t[g],
                             "iou": m["iou"], "cls": cls})
        if not history.get(g) or history[g][-1] != k:
            history.setdefault(g, []).append(k)
        last_trk[g], last_t[g], owner[k] = k, t, (g, t)
    return switches


def identity_stats(matches: list[dict], gt: dict, trk: dict) -> dict:
    """GT identities, predicted ids, fragmentation (ids per person) and purity."""
    per_person: dict[int, Counter] = defaultdict(Counter)
    per_track: dict[int, Counter] = defaultdict(Counter)
    for m in matches:
        per_person[m["gt"]][m["trk"]] += 1
        per_track[m["trk"]][m["gt"]] += 1
    frag = [len(c) for c in per_person.values()]
    matched = sum(sum(c.values()) for c in per_track.values())
    pure = sum(max(c.values()) for c in per_track.values())
    return {
        "gt_identities": len({int(i) for ids, _ in gt.values() for i in ids}),
        "predicted_ids": len({int(i) for ids, _ in trk.values() for i in ids}),
        "matched_people": len(per_person),
        "tracks_per_person_mean": float(np.mean(frag)) if frag else 0.0,
        "tracks_per_person_median": float(np.median(frag)) if frag else 0.0,
        "tracks_per_person_max": int(max(frag)) if frag else 0,
        "purity_matched_weighted": pure / matched if matched else 0.0,
    }


def nearest_neighbour_px(gt: dict, step: int = 25) -> dict:
    """Distance to the nearest other GT head (box centres, px), sampled every ``step`` frames."""
    d = []
    for t in sorted(gt)[::step]:
        c = gt[t][1][:, :2] + gt[t][1][:, 2:] / 2
        if len(c) > 1:
            dm = np.linalg.norm(c[:, None] - c[None], axis=2)
            np.fill_diagonal(dm, np.inf)
            d += list(dm.min(axis=1))
    d = np.array(d)
    return {"median_px": float(np.median(d)), "p10_px": float(np.percentile(d, 10)),
            "share_below_10px": float((d < 10).mean()), "share_below_20px": float((d < 20).mean())}


def analyse_clip(gt: dict, trk: dict, cfg: dict) -> dict:
    """Everything for one clip: stats, switches by class, first 10 s, threshold sensitivity."""
    cutoff = int(cfg["first_seconds"]) * int(cfg["fps"])
    primary = float(cfg["primary_threshold"])
    res: dict = {"by_threshold": {}}
    for thr in cfg["iou_thresholds"]:
        matches = clear_match(gt, trk, float(thr))
        sw = find_switches(matches)
        res["by_threshold"][f"{thr:g}"] = {
            "idsw": len(sw), "idsw_first_10s": sum(s["t"] <= cutoff for s in sw),
            "matches": len(matches),
            "by_class": {c: sum(s["cls"] == c for s in sw) for c in CLASSES}}
        if float(thr) == primary:
            margin = float(cfg["near_boundary_margin"])
            res["identity"] = identity_stats(matches, gt, trk)
            res["by_class_first_10s"] = {c: sum(s["cls"] == c and s["t"] <= cutoff for s in sw)
                                         for c in CLASSES}
            res["switch_gap_frames"] = {"direct_1": sum(s["gap"] == 1 for s in sw),
                                        "gap_2_to_25": sum(2 <= s["gap"] <= 25 for s in sw),
                                        "gap_over_25": sum(s["gap"] > 25 for s in sw)}
            ious = np.array([m["iou"] for m in matches])
            sw_iou = np.array([s["iou"] for s in sw]) if sw else np.array([])
            near = (ious >= primary) & (ious < primary + margin)
            sw_near = (sw_iou >= primary) & (sw_iou < primary + margin)
            res["boundary"] = {
                "match_share_iou_within_margin": float(near.mean()),
                "switch_share_iou_within_margin": float(sw_near.mean()) if len(sw) else 0.0,
                "median_iou_matches": float(np.median(ious)),
                "median_iou_at_switches": float(np.median(sw_iou)) if len(sw) else 0.0}
            res["switches_per_person_top5"] = Counter(s["gt"] for s in sw).most_common(5)
    res["gt_spacing"] = nearest_neighbour_px(gt)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    np.random.seed(int(cfg["seed"]))
    work = Path(cfg["work_dir"])
    results = json.loads(Path(cfg["results_json"]).read_text())
    official = results["iou_0.5"][cfg["tracker_name"]]["per_clip"]
    out = {"source": "DroneCrowd test, frozen run-2 tracks, read-only; clips already scored",
           "clips": {}}
    primary_key = f"{cfg['primary_threshold']:g}"
    for c in cfg["clips"]:
        seq = f"{c:03d}"
        gt = read_mot(work / "gt" / "MOT17-test" / seq / "gt" / "gt.txt")
        trk = read_mot(work / "trackers" / "MOT17-test" / cfg["tracker_name"] / "data"
                       / f"{seq}.txt")
        r = analyse_clip(gt, trk, cfg)
        r["trackeval_idsw_official"] = int(official[seq]["IDSW"])
        r["reproduces_trackeval"] = (r["by_threshold"][primary_key]["idsw"]
                                     == r["trackeval_idsw_official"])
        out["clips"][seq] = r
        print(json.dumps({seq: r}, indent=1))
    Path(cfg["out_json"]).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg["out_json"]).write_text(json.dumps(out, indent=1))
    print(f"wrote {cfg['out_json']}")


if __name__ == "__main__":
    main()
