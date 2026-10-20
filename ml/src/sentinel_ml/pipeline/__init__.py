"""Common module interface and runner (backend scaffold, step 17)."""

from sentinel_ml.pipeline.module import Module, run_clip
from sentinel_ml.pipeline.registry import available, build_module

__all__ = ["Module", "available", "build_module", "run_clip"]
