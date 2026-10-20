from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from sentinel_ml.density import DensityLayerConfig, DensityMap, density_layer, zone_grid  # noqa: E402

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "density_layer.yaml"


def dm_1080(seed: int = 0) -> DensityMap:
    # DM-Count's map for a 1920x1080 frame is 134 x 240.
    d = np.random.default_rng(seed).random((134, 240), dtype=np.float32)
    return DensityMap(density=d, frame_shape=(1080, 1920))


class FakeModel:
    def __init__(self) -> None:
        self.calls = 0

    def predict(self, frame: np.ndarray) -> DensityMap:
        self.calls += 1
        d = np.zeros((2 * (frame.shape[0] // 16), 2 * (frame.shape[1] // 16)), np.float32)
        d[0, 0] = 2.5  # top left zone
        d[-1, -1] = 1.0  # bottom right zone
        return DensityMap(density=d, frame_shape=frame.shape[:2])


@pytest.mark.parametrize("rows, cols", [(9, 16), (18, 32), (5, 7), (1, 1)])
def test_zones_sum_to_count(rows, cols):
    dm = dm_1080()
    people, valid = zone_grid(dm, rows, cols)
    assert people.shape == valid.shape == (rows, cols)
    assert people.sum() == pytest.approx(dm.count, rel=1e-6)
    assert valid.all()


def test_each_zone_sums_its_own_cells():
    dm = dm_1080()
    people, _ = zone_grid(dm, 9, 16)
    # 120 px zones = 15 map cells per side; the last zone row gets the 14 cells left of 134.
    assert people[0, 0] == pytest.approx(dm.density[:15, :15].sum(), rel=1e-6)
    assert people[8, 15] == pytest.approx(dm.density[120:, 225:].sum(), rel=1e-6)


def test_zone_without_cells_is_invalid():
    # 200 rows on a 1080 frame are finer than the 8 px map cells, so some zones get no cell.
    people, valid = zone_grid(dm_1080(), 200, 16)
    assert not valid.all()
    assert people[~valid].sum() == 0


def test_layer_follows_grid_contract():
    frames = ((i, np.zeros((1080, 1920, 3), np.uint8)) for i in range(60))
    model = FakeModel()
    layer = density_layer(frames, model, "seq011", DensityLayerConfig(every=25))
    assert model.calls == 3
    assert list(layer["frames"]) == ["0", "25", "50"]
    assert layer["frame_size"] == [1920, 1080]
    assert layer["grid"] == {"rows": 9, "cols": 16}
    assert (layer["type"], layer["layer"], layer["units"]) == ("grid", "density", "people_per_cell")
    assert layer["fields"] == ["density"] and layer["schema_version"] == 1
    entry = layer["frames"]["0"]
    assert set(entry) == {"valid", "density"}
    assert len(entry["density"]) == 9 and all(len(r) == 16 for r in entry["density"])
    assert entry["density"][0][0] == 2.5 and entry["density"][8][15] == 1.0
    assert {v for r in entry["valid"] for v in r} == {1}


def test_layer_rejects_mixed_sizes_and_no_frames():
    model = FakeModel()
    mixed = [(0, np.zeros((1080, 1920, 3), np.uint8)), (25, np.zeros((720, 1280, 3), np.uint8))]
    with pytest.raises(ValueError):
        density_layer(mixed, model, "c")
    with pytest.raises(ValueError):
        density_layer([(3, np.zeros((64, 64, 3), np.uint8))], model, "c")


def test_config_file_loads():
    cfg = DensityLayerConfig.from_yaml(CONFIG)
    assert (cfg.rows, cfg.cols, cfg.every, cfg.max_age_frames) == (9, 16, 25, 25)
    with pytest.raises(ValueError):
        DensityLayerConfig(rows=0)
