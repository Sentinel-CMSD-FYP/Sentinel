from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sentinel_ml.density import DMCount, DMCountConfig, DMCountNet, overlay  # noqa: E402

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "density_dmcount.yaml"


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return DMCount(DMCountConfig(weights=None, device="cpu"))


def test_state_dict_keys_match_released_weights():
    keys = set(DMCountNet().state_dict())
    # Names used in the released model_*.pth files.
    for k in ("features.0.weight", "features.34.bias", "reg_layer.0.weight",
              "reg_layer.2.weight", "density_layer.0.weight"):
        assert k in keys
    assert len(keys) == 38


@pytest.mark.parametrize("h, w", [(256, 320), (1080, 1920)])
def test_density_is_one_eighth_and_sums_to_count(model, h, w):
    frame = np.random.default_rng(0).integers(0, 255, (h, w, 3), dtype=np.uint8)
    dm = model.predict(frame)
    assert dm.density.shape == ((h // 16) * 2, (w // 16) * 2)
    assert dm.density.dtype == np.float32
    assert (dm.density >= 0).all()
    assert dm.count == pytest.approx(float(dm.density.sum()))
    assert dm.frame_shape == (h, w)


def test_rejects_grey_frames(model):
    with pytest.raises(ValueError):
        model.predict(np.zeros((64, 64), dtype=np.uint8))


def test_overlay_matches_frame(model):
    frame = np.full((128, 160, 3), 100, dtype=np.uint8)
    out = overlay(frame, model.predict(frame))
    assert out.shape == frame.shape and out.dtype == np.uint8


def test_config_from_yaml():
    cfg = DMCountConfig.from_yaml(CONFIG)
    assert cfg.weights.endswith("finetune_e4.pth")
    assert cfg.device == "auto"
