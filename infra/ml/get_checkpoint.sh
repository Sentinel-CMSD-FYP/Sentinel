#!/usr/bin/env bash
# Run INSIDE the container, from /workspace. Downloads the authors' DOT+PD DroneCrowd checkpoint.
set -euo pipefail
mkdir -p ml/third_party/uav-dot/checkpoints
gdown 1wYa01jGYfrAun3SfxuWzcKAni3hzBOMV -O ml/third_party/uav-dot/checkpoints/dot_pd_dronecrowd_51.00.ckpt
ls -l ml/third_party/uav-dot/checkpoints
