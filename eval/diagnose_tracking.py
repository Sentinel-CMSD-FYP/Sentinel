"""Diagnose why the baseline tracker fragments identities on a clip (read-only analysis).

For each clip it runs the tracker on Model B's points (the tracker code is not changed; its
methods are wrapped while the analysis runs and restored afterwards) and reports:
  * how output tracks end (border kill, expired after > max_age misses, alive at the end) and,
    for expired ones, how far the nearest detection was at the first missed frame;
  * how many output ids sit on ground-truth heads ("real") and how many do not ("spurious"),
    and how many extra ids real people received;
  * whether short tracks (< min_len frames) have a predecessor or successor nearby;
  * Model B's detection continuity per GT person (dropouts, split by length vs max_age);
  * camera motion proxies: GT per-person step, GT common-mode motion, optical-flow estimate;
  * the border rule and the count error with and without the min-length filter.

Units: pixels and frames. Coordinates: (x, y) = (column, row), original frame pixels.
Ground truth is read from the .mat files only (no train images needed). The test split is
for diagnosis only: never tune on what you see here.

    python eval/diagnose_tracking.py --config ml/configs/tracker.yaml \
        --data-config ml/configs/dronecrowd_data.yaml --split test --clips 11 75
"""

from __future__ import annotations

import argparse
import contextlib
import json
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
import point_metrics as pm  # eval/ is on sys.path when this file is run as a script
import yaml

import sentinel_ml.tracking.tracker as T
from sentinel_ml.localization import data as D
from sentinel_ml.tracking import PointTracker, TrackerConfig

MIN_LEN = 45          # the authors' evaluation drops tracks shorter than this (frames)
LINK_FRAMES = 15      # a successor or predecessor must start or end within this many frames ...
LINK_PX = 30.0        # ... and this many pixels of the track end or start
MATCH_PX = 10.0       # a track point counts as on a GT head within this radius


@contextlib.contextmanager
def recording():
    """Wrap the tracker's methods to log births, misses and deaths. Yields the records.

    ``records[track_id]`` holds ``birth`` (frame), ``birth_state``, ``death`` as
    ``(cause, frame)`` with cause ``border`` or ``expired``, and ``first_miss`` as
    ``(frame, distance in px from the prediction to the nearest detection)``.
    The original methods are restored on exit.
    """
    records: dict[int, dict] = defaultdict(lambda: {"death": None, "first_miss": None})
    cur = {"frame": -1, "pts": np.zeros((0, 3))}
    orig_new, orig_update = PointTracker._new_track, PointTracker.update
    orig_predict, orig_missed = T.Track.predict, T.Track.mark_missed

    def new_track(self, xy, score, state):
        orig_new(self, xy, score, state)
        records[self.tracks[-1].id].update(birth=cur["frame"], birth_state=state.name)

    def update(self, points):
        pts = np.asarray(points, float).reshape(-1, 3)
        cur["pts"] = pts[pts[:, 2] >= self.cfg.score_thr]
        cur["frame"] += 1
        return orig_update(self, points)

    def predict(self, frame_size):
        before = self.state
        orig_predict(self, frame_size)
        if before is not T.State.DEAD and self.state is T.State.DEAD:
            records[self.id]["death"] = ("border", cur["frame"])

    def missed(self):
        first = self.missing_counter == 0
        before = self.state
        orig_missed(self)
        if first and records[self.id]["first_miss"] is None:
            d = cur["pts"]
            dist = (float(np.min(np.linalg.norm(d[:, :2] - np.array(self.kf.xy), axis=1)))
                    if len(d) else float("inf"))
            records[self.id]["first_miss"] = (cur["frame"], dist)
        if self.state is T.State.DEAD and before is not T.State.DEAD:
            records[self.id]["death"] = ("expired", cur["frame"])

    PointTracker._new_track, PointTracker.update = new_track, update
    T.Track.predict, T.Track.mark_missed = predict, missed
    try:
        yield records
    finally:
        PointTracker._new_track, PointTracker.update = orig_new, orig_update
        T.Track.predict, T.Track.mark_missed = orig_predict, orig_missed


def run_recorded(points_doc: dict, cfg: TrackerConfig):
    """Run the tracker on a points document. Returns ({frame_id: rows}, records)."""
    w, h = points_doc["frame_size"]
    outputs: dict[int, list] = {}
    with recording() as records:
        tracker = PointTracker((w, h), cfg)
        for fid in sorted(points_doc["frames"], key=int):
            pts = np.asarray(points_doc["frames"][fid], float).reshape(-1, 3)
            outputs[int(fid)] = tracker.update(pts)
    return outputs, dict(records)


def track_table(outputs: dict[int, list]) -> dict[int, list[tuple[int, float, float]]]:
    """{track_id: [(frame_id, x, y), ...]} for ids that were output."""
    out: dict[int, list] = defaultdict(list)
    for fid, rows in outputs.items():
        for tid, x, y, _ in rows:
            out[tid].append((fid, x, y))
    return dict(out)


def end_causes(tracks: dict, records: dict, gate_px: float) -> dict:
    """How output tracks end, and the nearest-detection distance class for expired ones."""
    ended: Counter = Counter()
    near: Counter = Counter()
    for tid in tracks:
        rec = records[tid]
        if rec["death"] is None:
            ended["alive_at_end"] += 1
            continue
        ended[rec["death"][0]] += 1
        if rec["death"][0] == "expired" and rec["first_miss"]:
            d = rec["first_miss"][1]
            near["within_gate" if d <= gate_px else "gate_to_2gate" if d <= 2 * gate_px
                 else "farther"] += 1
    return {"end_cause": dict(ended), "expired_first_miss_nearest_detection": dict(near),
            "tracks_created": len(records),
            "never_output": sum(1 for t in records if t not in tracks),
            "killed_by_border_any_state": sum(1 for r in records.values()
                                              if r["death"] and r["death"][0] == "border")}


def short_track_links(tracks: dict, min_len: int = MIN_LEN) -> dict:
    """For tracks shorter than ``min_len``: do they follow or precede another track nearby?"""
    first = {t: v[0] for t, v in tracks.items()}
    last = {t: v[-1] for t, v in tracks.items()}
    short = [t for t, v in tracks.items() if len(v) < min_len]
    pred = succ = isolated = 0
    for t in short:
        f0, x0, y0 = first[t]
        f1, x1, y1 = last[t]
        has_pred = any(o != t and 0 < f0 - last[o][0] <= LINK_FRAMES
                       and np.hypot(last[o][1] - x0, last[o][2] - y0) <= LINK_PX for o in tracks)
        has_succ = any(o != t and 0 < first[o][0] - f1 <= LINK_FRAMES
                       and np.hypot(first[o][1] - x1, first[o][2] - y1) <= LINK_PX
                       for o in tracks)
        pred += has_pred
        succ += has_succ
        isolated += (not has_pred and not has_succ)
    return {"short_tracks": len(short), "with_predecessor": pred, "with_successor": succ,
            "isolated": isolated}


def gt_identity(outputs: dict, gt: dict, n_gt: int) -> dict:
    """Split output ids into real (mostly on a GT head) and spurious; count extra ids.

    ``purity``: over the rows of real ids that sit on a GT head, the share that belong to the
    id's dominant GT person. 1.0 means no id ever jumped to a different person (no wrong merge).
    """
    on_gt: dict[int, Counter] = defaultdict(Counter)
    rows_n: Counter = Counter()
    for fid, rows in outputs.items():
        if not rows or fid not in gt:
            continue
        xy, ids = gt[fid]
        pts = np.array([[r[1], r[2]] for r in rows])
        for p, g in pm.match_points(pts, xy, MATCH_PX):
            on_gt[rows[p][0]][int(ids[g])] += 1
        for r in rows:
            rows_n[r[0]] += 1
    spurious = [t for t in rows_n if sum(on_gt[t].values()) / rows_n[t] < 0.5]
    real = [t for t in rows_n if t not in set(spurious)]
    per_person: dict[int, set] = defaultdict(set)
    for t in real:
        if on_gt[t]:
            per_person[on_gt[t].most_common(1)[0][0]].add(t)
    extra = sum(len(s) - 1 for s in per_person.values())
    matched = sum(sum(on_gt[t].values()) for t in real)
    dominant = sum(on_gt[t].most_common(1)[0][1] for t in real if on_gt[t])
    return {"output_ids": len(rows_n), "purity": dominant / matched if matched else 1.0, "spurious_ids": len(spurious), "real_ids": len(real),
            "gt_people": n_gt, "gt_people_tracked": len(per_person), "extra_ids_for_tracked": extra,
            "gt_people_never_tracked": n_gt - len(per_person),
            "ids_per_tracked_person": dict(sorted(Counter(len(s) for s in
                                                          per_person.values()).items())),
            "spurious_lengths": [rows_n[t] for t in spurious],
            "real_lengths": [rows_n[t] for t in real]}


def detection_continuity(points_by_frame: dict[int, np.ndarray], gt: dict, thr: float,
                         max_age: int) -> dict:
    """How steadily B detects each GT person (score >= thr, within MATCH_PX)."""
    detected: dict[int, dict[int, bool]] = defaultdict(dict)
    path: dict[int, list] = defaultdict(list)
    for fid in sorted(gt):
        xy, ids = gt[fid]
        p = pm.filter_by_score(points_by_frame.get(fid, np.zeros((0, 3))), thr)
        hit = {g for _, g in pm.match_points(p[:, :2] if len(p) else np.zeros((0, 2)), xy,
                                             MATCH_PX)}
        for k, gid in enumerate(ids):
            detected[int(gid)][fid] = k in hit
            path[int(gid)].append((fid, xy[k]))
    step = {}
    for gid, pts in path.items():
        d = [np.linalg.norm(pts[i + 1][1] - pts[i][1]) for i in range(len(pts) - 1)
             if pts[i + 1][0] == pts[i][0] + 1]
        step[gid] = float(np.median(d)) if d else 0.0
    slow_cut = float(np.percentile(list(step.values()), 25)) if step else 0.0
    keys = ("person_frames", "detected", "dropouts", "dropouts_longer_than_max_age",
            "dropouts_up_to_max_age")
    tot = {g: dict.fromkeys(keys, 0) for g in ("all", "slow")}  # zeros stay visible in reports
    for gid, fr in detected.items():
        seq = [fr[f] for f in sorted(fr)]
        groups = ["all"] + (["slow"] if step[gid] <= slow_cut else [])
        for g in groups:
            tot[g]["person_frames"] += len(seq)
            tot[g]["detected"] += sum(seq)
        i = 0
        while i < len(seq):
            if seq[i]:
                i += 1
                continue
            j = i
            while j < len(seq) and not seq[j]:
                j += 1
            if i > 0 and j < len(seq):  # a gap between two detections
                for g in groups:
                    tot[g]["dropouts"] += 1
                    tot[g]["dropouts_longer_than_max_age" if j - i > max_age
                           else "dropouts_up_to_max_age"] += 1
            i = j
    return {**tot, "slow_median_step_px": slow_cut}


def gt_motion(gt: dict) -> dict:
    """GT per-person step between consecutive frames and the common-mode motion per frame."""
    steps, common = [], []
    for fid in sorted(gt):
        if fid + 1 not in gt:
            continue
        a_xy, a_ids = gt[fid]
        b_xy, b_ids = gt[fid + 1]
        index = {int(g): k for k, g in enumerate(a_ids)}
        vecs = [b_xy[k] - a_xy[index[int(g)]] for k, g in enumerate(b_ids) if int(g) in index]
        steps += [float(np.linalg.norm(v)) for v in vecs]
        if vecs:
            common.append(np.linalg.norm(np.median(np.array(vecs), axis=0)))
    s = np.array(steps)
    c = np.array(common)
    return {"step_median_px": float(np.median(s)), "step_p95_px": float(np.percentile(s, 95)),
            "step_max_px": float(s.max()),
            "share_steps_above_20px": float((s > 20).mean()),
            "common_mode_median_px": float(np.median(c)), "common_mode_max_px": float(c.max())}


def camera_translation(image_paths: list[Path], downscale: int = 2) -> dict:
    """Optical-flow estimate of camera translation per frame (full-resolution pixels)."""
    import cv2

    prev, ts = None, []
    for path in image_paths:
        g = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, (g.shape[1] // downscale, g.shape[0] // downscale))
        if prev is not None:
            kp = cv2.goodFeaturesToTrack(prev, maxCorners=1000, qualityLevel=0.01,
                                         minDistance=1, blockSize=3)
            if kp is not None:
                nxt, ok, _ = cv2.calcOpticalFlowPyrLK(prev, g, kp, None)
                src, dst = kp[ok == 1], nxt[ok == 1]
                if len(src) > 4:
                    m, _ = cv2.estimateAffinePartial2D(src, dst, cv2.RANSAC)
                    if m is not None:
                        ts.append(float(np.hypot(m[0, 2], m[1, 2]) * downscale))
        prev = g
    t = np.array(ts)
    return {"frames_pairs": len(ts), "median_px": float(np.median(t)),
            "p95_px": float(np.percentile(t, 95)), "max_px": float(t.max())}


def border_stats(points_by_frame: dict, gt: dict, size: tuple[int, int], margin: int,
                 thr: float) -> dict:
    """GT heads and detections within ``margin`` px of the frame border."""
    w, h = size

    def near(xy):
        if len(xy) == 0:
            return np.zeros(0, bool)
        return np.minimum.reduce([xy[:, 0], xy[:, 1], w - 1 - xy[:, 0], h - 1 - xy[:, 1]]) < margin

    gt_n = gt_b = det_n = det_b = 0
    people = set()
    for fid, (xy, ids) in gt.items():
        b = near(xy)
        gt_n += len(xy)
        gt_b += int(b.sum())
        people |= {int(i) for i, flag in zip(ids, b, strict=True) if flag}
    for p in points_by_frame.values():
        p = p[p[:, 2] >= thr]
        det_n += len(p)
        det_b += int(near(p[:, :2]).sum())
    return {"gt_person_frames": gt_n, "gt_within_margin": gt_b, "gt_people_touching": len(people),
            "detections": det_n, "detections_within_margin": det_b}


def count_error(n_ids: int, n_gt: int) -> float:
    """|ids - GT| / GT in percent."""
    return 100.0 * abs(n_ids - n_gt) / n_gt


def analyse_clip(points_doc: dict, gt: dict, cfg: TrackerConfig, image_paths: list[Path] | None
                 ) -> dict:
    """Full diagnosis of one clip. ``gt`` is ``{frame_id: (xy, ids)}``."""
    n_gt = len({int(i) for _, ids in gt.values() for i in ids})
    size = tuple(points_doc["frame_size"])
    outputs, records = run_recorded(points_doc, cfg)
    tracks = track_table(outputs)
    lengths = {t: len(v) for t, v in tracks.items()}
    n_ids, n_filtered = len(tracks), sum(1 for n in lengths.values() if n >= MIN_LEN)
    by_frame = {int(k): np.array(v, float).reshape(-1, 3) for k, v in points_doc["frames"].items()}
    ident = gt_identity(outputs, gt, n_gt)
    sp, rl = ident.pop("spurious_lengths"), ident.pop("real_lengths")
    ident["median_len_spurious"] = float(np.median(sp)) if sp else 0.0
    ident["median_len_real"] = float(np.median(rl)) if rl else 0.0
    ident["spurious_shorter_than_min_len"] = sum(n < MIN_LEN for n in sp)
    ident["real_shorter_than_min_len"] = sum(n < MIN_LEN for n in rl)
    margin0_out, _ = run_recorded(points_doc, replace(cfg, border_margin_px=0))
    t0 = track_table(margin0_out)
    n0 = len(t0)
    n0_f = sum(1 for v in t0.values() if len(v) >= MIN_LEN)
    report = {
        "gt_unique_ids": n_gt, "output_ids": n_ids, "output_ids_min_len": n_filtered,
        "min_len_frames": MIN_LEN,
        "end": end_causes(tracks, records, float(2 * (cfg.size_px // 2))),
        "short_tracks": short_track_links(tracks),
        "identity": ident,
        "detection": detection_continuity(by_frame, gt, cfg.score_thr, cfg.max_age),
        "gt_motion": gt_motion(gt),
        "count_error_pct": {"no_filter": count_error(n_ids, n_gt),
                            "min_len_filter": count_error(n_filtered, n_gt)},
        "border": border_stats(by_frame, gt, size, cfg.border_margin_px, cfg.score_thr),
        "border_margin_0": {"output_ids": n0, "output_ids_min_len": n0_f,
                            "count_error_pct_no_filter": count_error(n0, n_gt),
                            "count_error_pct_min_len_filter": count_error(n0_f, n_gt)},
    }
    if image_paths:
        report["camera_translation_optical_flow"] = camera_translation(image_paths)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", type=Path, required=True, help="tracker.yaml")
    ap.add_argument("--data-config", type=Path, required=True, help="dronecrowd_data.yaml")
    ap.add_argument("--split", required=True, choices=["val", "test"])
    ap.add_argument("--clips", type=int, nargs="+", required=True)
    ap.add_argument("--points-dir", type=Path, help="default <points_dir>/<split> from the config")
    ap.add_argument("--flow-frames", type=int, default=150,
                    help="frames per clip for the optical-flow camera estimate (0 = skip)")
    ap.add_argument("--out", type=Path, help="write the report as JSON")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    tracker_cfg = TrackerConfig(**cfg["tracker"])
    data_cfg = D.load_config(args.data_config)
    points_dir = args.points_dir or Path(cfg["points_dir"]) / args.split
    reports = {}
    for clip in args.clips:
        doc = json.loads((points_dir / f"clip_{clip:03d}_points.json").read_text())
        frame_ids = sorted(int(k) for k in doc["frames"])
        gt = D.clip_gt_frames(data_cfg, args.split, clip, frame_ids)
        images = None
        if args.flow_frames:
            folder = Path(data_cfg["root"]) / (data_cfg["test_dir"] if args.split == "test"
                                              else data_cfg["train_dir"]) / "images"
            images = [folder / f"img{clip:03d}{f + 1:03d}.jpg" for f in range(args.flow_frames)]
            images = images if all(p.exists() for p in images[:2]) else None
        reports[f"clip_{clip:03d}"] = analyse_clip(doc, gt, tracker_cfg, images)
        print(json.dumps({f"clip_{clip:03d}": reports[f"clip_{clip:03d}"]}, indent=1))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(reports, indent=1) + "\n")


if __name__ == "__main__":
    main()
