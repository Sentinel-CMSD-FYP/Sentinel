import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from sentinel_ml.flow.export import ExportConfig, cell_texture, centre_mask, export_flow
from sentinel_ml.flow.sources import open_frames

CHECKER = Path(__file__).resolve().parents[2] / "docs" / "layers" / "check_layer.py"


def check_layer(doc, n_frames=None):
    """The contract's own checker, so the exporter is held to the real rules."""
    if not CHECKER.is_file():
        pytest.skip("docs/layers/check_layer.py is not available here")
    spec = importlib.util.spec_from_file_location("check_layer", CHECKER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.check_layer(doc, n_frames)


def texture(h=240, w=320, seed=0):
    rng = np.random.default_rng(seed)
    noise = rng.uniform(0, 255, (h, w)).astype(np.float32)
    return cv2.GaussianBlur(noise, (0, 0), 2.0).astype(np.uint8)


def sequence(n=30, dx=2, base=None, h=240, w=320):
    """Frames of a texture sliding right by ``dx`` pixels per frame (reflected at the edge)."""
    base = texture(h, w) if base is None else base
    frames = []
    for i in range(n):
        m = np.float32([[1, 0, dx * i], [0, 1, 0]])
        g = cv2.warpAffine(base, m, (w, h), borderMode=cv2.BORDER_REFLECT)
        frames.append(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR))
    return frames


def write_folder(path: Path, frames):
    path.mkdir()
    for i, f in enumerate(frames):
        cv2.imwrite(str(path / f"img{i:04d}.png"), f)  # png: no compression noise
    return path


# These tests slide the whole frame, which is camera motion, so the correction is off here;
# the tests below it turn it on.
CFG = ExportConfig(rows=3, cols=4, every=10, gap=5, min_texture=0.5, stabilise=False)


def load(out: Path):
    clip = json.loads((out / "clip.json").read_text())
    layer = json.loads((out / "layers" / "flow.json").read_text())
    return clip, layer


def test_exports_a_frame_folder_with_correct_flow(tmp_path):
    src = write_folder(tmp_path / "frames", sequence(30, dx=2))
    out = tmp_path / "clip"
    s = export_flow(src, out, fps=10.0, config=CFG)

    clip, layer = load(out)
    # samples at 0, 10, 20; each needs the frame 5 later, and 20 + 5 < 30
    assert sorted(layer["frames"], key=int) == ["0", "10", "20"]
    assert layer["max_age_frames"] == 10
    assert layer["units"] == "px_per_frame"
    assert layer["grid"] == {"rows": 3, "cols": 4}
    assert s["frames"] == clip["n_frames"] == 30
    assert clip["fps"] == 10.0 and clip["frame_size"] == [320, 240]

    entry = layer["frames"]["10"]
    valid = np.array(entry["valid"], bool)
    assert valid.any()
    # 2 px per frame to the right, reported in pixels per frame (not per gap)
    assert np.median(np.array(entry["u"])[valid]) == pytest.approx(2.0, abs=0.4)
    assert np.median(np.array(entry["v"])[valid]) == pytest.approx(0.0, abs=0.4)
    assert np.median(np.array(entry["speed"])[valid]) == pytest.approx(2.0, abs=0.4)


def test_output_obeys_the_layer_contract(tmp_path):
    out = tmp_path / "clip"
    export_flow(write_folder(tmp_path / "f", sequence(30)), out, fps=10.0, config=CFG)
    clip, layer = load(out)
    assert check_layer(layer, clip["n_frames"]) == []
    assert clip["layers"][0]["layer"] == "flow" and clip["layers"][0]["type"] == "grid"
    assert clip["video_url"] == f"/api/v1/clips/{out.name}/video"
    assert (out / "video.mp4").stat().st_size > 0


def test_cells_without_texture_are_marked_invalid(tmp_path):
    base = texture()
    base[:, 160:] = 128  # right half is flat: no reliable flow there
    out = tmp_path / "clip"
    export_flow(write_folder(tmp_path / "f", sequence(30, dx=0, base=base)), out, fps=10.0,
                config=ExportConfig(rows=3, cols=4, every=10, gap=5, min_texture=1.5))
    valid = np.array(load(out)[1]["frames"]["10"]["valid"])
    assert valid[:, 0].all() and valid[:, 1].all()  # textured left half
    assert not valid[:, 2].any() and not valid[:, 3].any()  # flat right half


def test_exports_an_mp4_the_same_way(tmp_path):
    video = tmp_path / "in.mp4"
    w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (320, 240))
    if not w.isOpened():
        pytest.skip("this OpenCV build cannot write mp4")
    for f in sequence(30, dx=2):
        w.write(f)
    w.release()

    out = tmp_path / "clip"
    export_flow(video, out, config=CFG)  # fps comes from the file
    clip, layer = load(out)
    assert clip["fps"] == pytest.approx(10.0, abs=0.1)
    assert clip["n_frames"] == 30
    entry = layer["frames"]["10"]
    valid = np.array(entry["valid"], bool)
    assert np.median(np.array(entry["u"])[valid]) == pytest.approx(2.0, abs=0.6)  # lossy codec


def test_start_and_max_frames_renumber_from_zero(tmp_path):
    out = tmp_path / "clip"
    cfg = ExportConfig(rows=3, cols=4, every=10, gap=5, min_texture=0.5, start=5, max_frames=20)
    export_flow(write_folder(tmp_path / "f", sequence(40)), out, fps=10.0, config=cfg)
    clip, layer = load(out)
    assert clip["n_frames"] == 20
    assert sorted(layer["frames"], key=int) == ["0", "10"]  # the layer matches the cut video


def test_wide_video_is_shrunk_to_even_dimensions(tmp_path):
    out = tmp_path / "clip"
    cfg = ExportConfig(rows=2, cols=2, every=10, gap=5, min_texture=0.5, max_width=161)
    export_flow(write_folder(tmp_path / "f", sequence(30)), out, fps=10.0, config=cfg)
    w, h = load(out)[0]["frame_size"]
    assert w <= 161 and w % 2 == 0 and h % 2 == 0
    assert load(out)[1]["frame_size"] == [w, h]


def test_frame_folder_filters_by_sequence_prefix(tmp_path):
    folder = tmp_path / "images"
    folder.mkdir()
    for name in ("img011001.png", "img011002.png", "img012001.png"):
        cv2.imwrite(str(folder / name), sequence(1)[0])
    (folder / "notes.txt").write_text("not an image")
    frames, fps, size = open_frames(folder, seq="011", fps=25.0)
    assert fps == 25.0 and size == (320, 240)
    assert len(list(frames)) == 2
    assert len(list(open_frames(folder, seq="012", fps=1.0)[0])) == 1
    assert len(list(open_frames(folder)[0])) == 3  # no prefix: every image, text file ignored


def test_missing_input_fails_clearly(tmp_path):
    with pytest.raises(SystemExit, match="no image frames"):
        open_frames(tmp_path)
    with pytest.raises(SystemExit, match="cannot open"):
        open_frames(tmp_path / "nope.mp4")


@pytest.mark.parametrize("kwargs", [{"gap": 0}, {"every": 0}, {"rows": 0}, {"cols": 0}])
def test_bad_config_is_rejected(kwargs):
    with pytest.raises(ValueError):
        ExportConfig(**kwargs)


def zooming(n=30, z=0.006, h=240, w=320):
    """Frames of a texture zooming in by a fraction ``z`` per frame about the centre."""
    base = texture(h, w)
    frames = []
    for i in range(n):
        m = cv2.getRotationMatrix2D((w / 2, h / 2), 0, (1 + z) ** i)
        g = cv2.warpAffine(base, m, (w, h), borderMode=cv2.BORDER_REFLECT)
        frames.append(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR))
    return frames


def test_camera_zoom_is_removed_and_reported(tmp_path):
    src = write_folder(tmp_path / "f", zooming())
    cfg = ExportConfig(rows=3, cols=4, every=10, gap=5, min_texture=0.5)
    raw = export_flow(src, tmp_path / "raw", fps=10.0, config=ExportConfig(
        rows=3, cols=4, every=10, gap=5, min_texture=0.5, stabilise=False))
    fixed = export_flow(src, tmp_path / "fixed", fps=10.0, config=cfg)

    assert raw["mean_speed_px_per_frame"] > 0.25  # the zoom shows up as motion everywhere
    assert fixed["mean_speed_px_per_frame"] < 0.1  # and is gone once subtracted
    assert fixed["mean_speed_before_px_per_frame"] == pytest.approx(
        raw["mean_speed_px_per_frame"], rel=0.05)

    cam = json.loads((tmp_path / "fixed" / "camera.json").read_text())
    first = cam["frames"]["0"]
    assert first["zoom_percent_per_s"] == pytest.approx(0.006 * 10 * 100, rel=0.25)  # ~6 %/s
    assert first["used"] is True
    assert fixed["mean_zoom_percent_per_s"] == pytest.approx(6.0, rel=0.25)
    assert not (tmp_path / "raw" / "camera.json").exists()  # nothing removed, nothing to report


def test_stabilised_output_still_obeys_the_contract(tmp_path):
    out = tmp_path / "clip"
    export_flow(write_folder(tmp_path / "f", zooming()), out, fps=10.0,
                config=ExportConfig(rows=3, cols=4, every=10, gap=5, min_texture=0.5))
    clip, layer = load(out)
    assert check_layer(layer, clip["n_frames"]) == []
    assert clip["produced_by"]["stabilise"] is True


def test_ambiguous_camera_motion_leaves_a_gap_not_a_guess(tmp_path):
    """Nearly half the frame moves and the rest is still: which part is the camera?"""
    base = texture()
    frames = []
    for i in range(30):
        moving = np.roll(base, 2 * i, axis=1)
        g = base.copy()
        g[:, :140] = moving[:, :140]  # 44% of the width slides, the rest stays put
        frames.append(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR))
    out = tmp_path / "clip"
    s = export_flow(write_folder(tmp_path / "f", frames), out, fps=10.0, config=ExportConfig(
        rows=3, cols=4, every=10, gap=5, min_texture=0.5, min_inliers=0.9))
    assert load(out)[1]["frames"] == {}  # nothing is claimed
    assert s["camera_samples_dropped"] == 3
    cam = json.loads((out / "camera.json").read_text())
    assert all(c["used"] is False and c["inlier_fraction"] < 0.9 for c in cam["frames"].values())


def test_a_frame_wide_convergence_is_read_as_zoom_out():
    """Known limit, written down so nobody is surprised: from flow alone, everything moving
    toward the centre cannot be told apart from the camera zooming out."""
    from sentinel_ml.flow import FlowField
    from sentinel_ml.flow.camera import remove_camera_motion

    x = ((np.arange(80) + 0.5) / 80 - 0.5) * 320
    y = ((np.arange(60) + 0.5) / 60 - 0.5) * 240
    xx, yy = np.meshgrid(x, y)
    converging = FlowField((-0.01 * xx).astype(np.float32), (-0.01 * yy).astype(np.float32))
    clean, cam = remove_camera_motion(converging, (320, 240))
    assert cam.zoom == pytest.approx(-0.01, abs=1e-4)
    assert np.abs(clean.u).max() < 1e-3  # removed, as camera motion


def test_centre_mask():
    assert centre_mask(4, 4, 1.0).all()
    inner = centre_mask(4, 4, 0.5)
    assert inner.sum() == 4 and inner[1:3, 1:3].all()
    assert centre_mask(3, 3, 0.5).sum() == 1 and centre_mask(3, 3, 0.5)[1, 1]


def test_centre_fraction_marks_outer_cells_invalid(tmp_path):
    out = tmp_path / "clip"
    export_flow(write_folder(tmp_path / "f", sequence(30)), out, fps=10.0, config=ExportConfig(
        rows=4, cols=4, every=10, gap=5, min_texture=0.5, centre_fraction=0.5))
    valid = np.array(load(out)[1]["frames"]["10"]["valid"], bool)
    assert valid[1:3, 1:3].all()
    assert valid.sum() == 4  # only the middle 2x2


@pytest.mark.parametrize("kwargs", [{"centre_fraction": 0}, {"centre_fraction": 1.5},
                                    {"min_inliers": -0.1}, {"min_inliers": 1.1}])
def test_bad_camera_settings_are_rejected(kwargs):
    with pytest.raises(ValueError):
        ExportConfig(**kwargs)


def test_cell_texture_separates_texture_from_flat():
    img = texture()
    img[:, 160:] = 90
    t = cell_texture(img, 2, 2)
    assert t[:, 0].min() > 1.5
    assert t[:, 1].max() < 0.2
