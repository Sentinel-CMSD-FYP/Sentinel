"""Live mode: run the modules on a stream as it arrives and hand out each sample at once.

The offline runner (``run_clip``) gives every frame to every module in order and may take as
long as it needs. A live stream cannot wait for slow models, so here:

* a reader thread reads the source (a file is paced at its own fps, as the stand in for a
  drone feed) and passes every frame to ``on_frame`` (the backend records it and keeps a
  preview);
* each chain (a module plus the modules that ``need`` it, e.g. points then tracks) runs in its
  own thread on the frames it ``wants``, in frame order, so a slow model does not hold up a
  fast one;
* a chain that falls behind skips: a frame that waited in its queue longer than ``max_lag_s``
  is dropped (the newest one is always kept), so the delay stays near one model call and the
  sample rate drops instead. A skipped frame simply has no sample (the layer's gaps rule shows
  that).

A module used live needs ``samples`` (its samples so far, keyed by frame id, in the order they
were made) and may have ``wants(frame_id)`` (default: every frame), which must depend on the
frame id only, since the reader calls it. After each frame a module processes, its new samples
go to ``on_sample(layer, frame_id, data, header, t_capture)``; ``header`` is the layer without
its frames, given with the layer's first sample only, and ``t_capture`` is when the reader got
that frame (``time.time()``).

    result = run_live(frames, modules, "live_x", fps=25.0, on_sample=print)
    result.layers   # the finished layers, as run_clip returns them
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path

import cv2
import numpy as np

from sentinel_ml.flow.sources import open_frames
from sentinel_ml.pipeline.module import Module

logger = logging.getLogger(__name__)


@dataclass
class LiveResult:
    layers: dict[str, dict]  # name -> finished layer, as run_clip returns
    n_frames: int  # frames read (and passed to on_frame)
    errors: dict[str, str] = field(default_factory=dict)  # failed module -> why
    times: dict[str, float] = field(default_factory=dict)  # module -> seconds in its model
    ran: dict[str, int] = field(default_factory=dict)  # module -> frames it processed
    skipped: dict[str, int] = field(default_factory=dict)  # chain root -> frames dropped
    late_s: float = 0.0  # worst delay of the reader behind the file's own pace


def open_source(source: str, fps: float | None = None
                ) -> tuple[Iterator[np.ndarray], float, tuple[int, int], bool]:
    """``(frames, fps, (width, height), realtime)`` for a file, frame folder, URL or webcam.

    A file or folder is read as fast as asked, so ``realtime`` is True and ``run_live`` paces
    it; a URL (``rtsp://``, ``http://``) or a webcam index (``"0"``) delivers frames at its own
    pace.
    """
    if not (source.isdigit() or "://" in source):
        try:
            frames, src_fps, size = open_frames(Path(source), fps=fps)
        except SystemExit as e:  # open_frames reports a bad source this way
            raise ValueError(str(e)) from e
        return frames, src_fps, size, True
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        raise ValueError(f"cannot open {source}")
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    def read() -> Iterator[np.ndarray]:
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    return
                yield frame
        finally:
            cap.release()

    return read(), fps or cap.get(cv2.CAP_PROP_FPS) or 30.0, size, False


def chains(modules: list[Module]) -> list[list[Module]]:
    """Each module with the modules that need it, in the given order (needed ones first)."""
    root_of: dict[str, str] = {}
    groups: dict[str, list[Module]] = {}
    for m in modules:
        needs = getattr(m, "needs", ())
        missing = [n for n in needs if n not in root_of]
        if missing:
            raise ValueError(f"module {m.name!r} needs {missing} to run before it")
        roots = {root_of[n] for n in needs}
        if len(roots) > 1:
            raise ValueError(f"module {m.name!r} needs modules of different chains {sorted(roots)}")
        root = roots.pop() if roots else m.name
        root_of[m.name] = root
        groups.setdefault(root, []).append(m)
    return list(groups.values())


def wants(m: Module, frame_id: int) -> bool:
    fn = getattr(m, "wants", None)
    return True if fn is None else fn(frame_id)


def run_live(frames: Iterable[np.ndarray], modules: list[Module], clip_id: str, *, fps: float,
             realtime: bool = True, max_lag_s: float = 1.0,
             on_frame: Callable[[int, np.ndarray, float], None] | None = None,
             on_sample: Callable[[str, int, object, dict | None, float | None], None] | None = None,
             stop: Callable[[], bool] | None = None) -> LiveResult:
    """Run ``modules`` on ``frames`` (BGR, ids from 0) as they arrive; see the module doc.

    ``stop()`` is asked before each frame; True (or Ctrl+C) ends the reading, and the chains
    still finish the frames queued for them. A failing module is dropped with the ones that
    need it, as ``run_clip`` does with ``errors``; an exception in the reader or ``on_frame``
    is raised once the chains are done.
    """
    names = [m.name for m in modules]
    if not names:
        raise ValueError("no modules to run")
    if len(set(names)) != len(names):
        raise ValueError(f"module names must be unique, got {names}")
    groups = chains(modules)
    cond = threading.Condition()
    out = threading.Lock()  # on_sample and the error bookkeeping, one thread at a time
    queues: dict[str, deque] = {g[0].name: deque() for g in groups}
    result = LiveResult(layers={}, n_frames=0, times=dict.fromkeys(names, 0.0),
                        ran=dict.fromkeys(names, 0), skipped=dict.fromkeys(queues, 0))
    failed: dict[str, str] = {}
    captured: dict[int, float] = {}
    sent = dict.fromkeys(names, 0)
    headed: set[str] = set()
    stopping = threading.Event()
    state = {"finished": False, "error": None}

    def fail(m: Module, why: str) -> None:
        with out:
            failed[m.name] = why
            for d in modules:  # in order, so a chain of needs is dropped in one pass
                if d.name not in failed and set(getattr(d, "needs", ())) & failed.keys():
                    failed[d.name] = f"needs {sorted(set(d.needs) & failed.keys())}, which failed"

    def trim(root: str) -> None:
        """Drop the frames that waited too long, keeping the newest (call holding ``cond``)."""
        q, limit = queues[root], time.monotonic() - max_lag_s
        while len(q) > 1 and q[0][2] < limit:
            q.popleft()
            result.skipped[root] += 1

    def emit(m: Module) -> None:
        samples = m.samples
        new = len(samples) - sent[m.name]
        if new <= 0:
            return
        items = list(islice(reversed(samples.items()), new))[::-1]  # the newest ``new`` ones
        sent[m.name] += new
        header = None
        if m.name not in headed:
            headed.add(m.name)
            header = {k: v for k, v in m.layer(clip_id).items() if k != "frames"}
        with out:
            for key, data in items:
                if on_sample:
                    fid = int(key)
                    on_sample(m.name, fid, data, header, captured.get(fid))
                header = None

    def reader() -> None:
        t0 = time.monotonic()
        try:
            for fid, frame in enumerate(frames):
                if stopping.is_set() or (stop and stop()):
                    break
                if realtime:
                    wait = t0 + fid / fps - time.monotonic()
                    if wait > 0:
                        time.sleep(wait)
                    else:
                        result.late_s = max(result.late_s, -wait)
                t = time.time()
                captured[fid] = t
                if on_frame:
                    on_frame(fid, frame, t)
                result.n_frames = fid + 1
                with cond:
                    for g in groups:
                        if any(wants(m, fid) for m in g if m.name not in failed):
                            queues[g[0].name].append((fid, frame, time.monotonic()))
                        trim(g[0].name)
                    cond.notify_all()
        except BaseException as e:
            state["error"] = e
        finally:
            with cond:
                state["finished"] = True
                cond.notify_all()

    def chain(group: list[Module]) -> None:
        q = queues[group[0].name]
        while True:
            with cond:
                while not q and not state["finished"]:
                    cond.wait()
                if not q:
                    return
                trim(group[0].name)
                fid, frame, _ = q.popleft()
            for m in group:
                if m.name in failed or not wants(m, fid):
                    continue
                t = time.perf_counter()
                try:
                    m.process(fid, frame)
                    result.ran[m.name] += 1
                    emit(m)
                except Exception as e:
                    logger.exception("module %r failed on frame %d", m.name, fid)
                    fail(m, f"process (frame {fid}) failed: {type(e).__name__}: {e}")
                finally:
                    result.times[m.name] += time.perf_counter() - t

    threads = [threading.Thread(target=reader, name="live-reader", daemon=True)]
    threads += [threading.Thread(target=chain, args=(g,), name=f"live-{g[0].name}", daemon=True)
                for g in groups]
    for th in threads:
        th.start()
    for th in threads:
        while th.is_alive():
            try:
                th.join(0.5)  # a timeout, so Ctrl+C reaches this thread
            except KeyboardInterrupt:
                logger.info("stopping: the queued frames are still processed")
                stopping.set()

    for m in modules:
        if m.name in failed:
            continue
        try:
            result.layers[m.name] = m.layer(clip_id)
        except Exception as e:
            logger.exception("module %r failed in layer", m.name)
            fail(m, f"layer failed: {type(e).__name__}: {e}")
    for name in failed:
        result.layers.pop(name, None)
    result.errors = dict(failed)
    if state["error"] is not None:
        raise state["error"]
    return result
