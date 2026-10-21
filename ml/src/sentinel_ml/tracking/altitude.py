"""Per-clip altitude proxy for DroneCrowd benchmarking.

DroneCrowd has no altitude data. The authors of the UP-COUNT tracker (up-count/uav-dot-track,
``src/utils/video_reader.py``) use two stand-in values: 50 for clips with large objects and 100
for the rest. Large objects mean the drone is closer, so a wider matching gate. This is a size
proxy, not a measured altitude, and it only changes the matching gate (see ``resolve_config``).

For the pipeline itself the contract's constant altitude (60 m, assumed) is used instead.
"""

from __future__ import annotations

from pathlib import Path

import yaml

PROXY_ALTITUDE_M = {"large": 50.0, "small": 100.0}


def altitude_from_size_tags(tags: dict[int, str]) -> dict[int, float]:
    """Map ``{clip: "small" | "large"}`` to ``{clip: altitude in metres}``."""
    bad = {c: t for c, t in tags.items() if t not in PROXY_ALTITUDE_M}
    if bad:
        raise ValueError(f"unknown object-size tags {bad}; use one of {sorted(PROXY_ALTITUDE_M)}")
    return {int(c): PROXY_ALTITUDE_M[t] for c, t in tags.items()}


def load_size_tags(path: Path, split: str) -> dict[int, str] | None:
    """Object-size tags for ``split``, or None if the file is missing or not fully filled in."""
    if not Path(path).exists():
        return None
    tags = yaml.safe_load(Path(path).read_text()).get(split) or {}
    if not tags or any(v is None for v in tags.values()):
        return None
    altitude_from_size_tags(tags)  # validates the values
    return {int(c): v for c, v in tags.items()}
