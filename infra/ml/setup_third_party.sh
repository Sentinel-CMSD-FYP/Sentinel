#!/usr/bin/env bash
# Clone the two third-party repos at the commits used on the laptop, fetch the DOT+PD checkpoint.
# Run on the PC host from the repo root:  bash infra/ml/setup_third_party.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
TP=ml/third_party; mkdir -p "$TP"

if [ ! -d "$TP/uav-dot/.git" ]; then git clone https://github.com/up-count/uav-dot "$TP/uav-dot"; fi
git -C "$TP/uav-dot" checkout a6d3664b50cc0cc6f1a08848eb294b71753b6600

if [ ! -d "$TP/uav-dot-track/.git" ]; then git clone https://github.com/up-count/uav-dot-track "$TP/uav-dot-track"; fi
git -C "$TP/uav-dot-track" checkout 505a7289b9602524383ae4260912f8e783000ccf
# Only TrackEval is needed (eval/trackeval_eval.py). Its submodule URL is SSH, so point it at https.
git -C "$TP/uav-dot-track" config submodule.eval/TrackEval.url https://github.com/up-count/TrackEval.git
git -C "$TP/uav-dot-track" submodule update --init eval/TrackEval

echo "Third-party repos ready. TrackEval commit:"; git -C "$TP/uav-dot-track/eval/TrackEval" rev-parse HEAD
echo "(laptop had 5477c170b741f95a311dcea99a7099fca805e7e2)"
