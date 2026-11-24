"""eval/pipeline/fetch_weights.py and build_demo_clips.py without network or data."""

import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest

EVAL = Path(__file__).resolve().parents[2] / "eval" / "pipeline"


def load(name):
    spec = importlib.util.spec_from_file_location(name, EVAL / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


fw = load("fetch_weights")
bd = load("build_demo_clips")


def test_check_reports_missing_wrong_and_good_files(tmp_path, monkeypatch, capsys):
    good = tmp_path / "data" / "a.pth"
    good.parent.mkdir(parents=True)
    good.write_bytes(b"weights")
    (tmp_path / "data" / "b.pt").write_bytes(b"other")
    monkeypatch.setattr(fw, "REPO", tmp_path)
    monkeypatch.setattr(fw, "RELEASE", [
        ("a.pth", "data/a.pth", hashlib.md5(b"weights").hexdigest()),
        ("b.pt", "data/b.pt", hashlib.md5(b"expected").hexdigest())])
    monkeypatch.setattr(fw, "B_CKPT", ("ckpt/b.ckpt", "drive id", "0" * 32))
    monkeypatch.setattr(sys, "argv", ["fetch_weights.py", "--check"])
    with pytest.raises(SystemExit) as e:
        fw.main()
    assert e.value.code == 1
    out = capsys.readouterr().out.splitlines()
    assert out == ["ok        data/a.pth", "wrong md5 data/b.pt", "missing   ckpt/b.ckpt"]


def test_every_release_file_matches_a_config_path():
    configs = (EVAL.parents[1] / "ml" / "configs")
    text = (configs / "density_dmcount.yaml").read_text() + \
        (configs / "infrastructure_yolo26.yaml").read_text()
    assert all(dest in text for _, dest, _ in fw.RELEASE)
    assert fw.B_CKPT[0] in (configs / "infer_b.yaml").read_text()


def test_build_refuses_an_unknown_clip(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["build_demo_clips.py", "--clips", "seq011,seq999"])
    with pytest.raises(SystemExit, match="seq999"):
        bd.main()
