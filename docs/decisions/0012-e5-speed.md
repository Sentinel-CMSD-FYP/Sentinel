# 0012 E-5 B+E speed on the RTX 4000 Ada
Date: 2026-11-23   Owner: Ahmed
Context: the only B+E speed number so far (4.9 to 5.3 FPS) came from a Kaggle T4. The target (>= 5 FPS per stream, 3 to 4 concurrent streams) is defined for the RTX 4000 Ada. `eval/benchmark_be.py` had never run on it.
Decision: measure on the target GPU with the frozen run-2 tracker config; report both the 1-stream and the multi-stream result; the speed target is MET for 1 stream and NOT MET for 3 and 4 streams.
Evidence:
- Command: `bash infra/pc/gpu-job.sh docker compose -p sentinel-$USER -f infra/ml/compose.gpu.yaml run --rm -T ml env PYTHONPATH=ml/src python eval/benchmark_be.py --infer-config ml/configs/infer_b.yaml --tracker-config ml/configs/tracker_selected_e2.yaml --split test --streams 1 3 4 --frames 250 --warmup 10 --device cuda --label alone_rtx4000ada`
- Raw result: `data/outputs/benchmark/alone_rtx4000ada.json` (gitignored). Source label for every number: RTX 4000 Ada (driver 595.99.02), torch 2.1.0+cu121 (CUDA 12.1), DroneCrowd test clips 11, 15, 16, 17 (1920x1080, net input 1920x1088), 250 timed frames per stream after 10 untimed warm-up, frame_stride 1, fp16 OFF (fp32), tracker_selected_e2.yaml, images read from disk inside the timed loop, threads in one process with one shared model B, device synchronised before each timer stop.
- GPU before the run: 291 MiB used, 0 % util, no compute processes (nvidia-smi). Nothing else was running; this is the "alone" run. A run with teammates' jobs was NOT done.

| Streams | Lowest-stream avg FPS | Lowest-stream p95 FPS | All streams together | 5 FPS per stream |
|---|---|---|---|---|
| 1 | 10.20 | 10.12 | 10.20 | met |
| 3 | 4.43 (streams 4.43 / 4.54 / 4.82) | 3.59 | 13.79 | NOT met |
| 4 | 3.32 (streams 3.32 / 3.33 / 3.51 / 3.53) | 2.74 | 13.69 | NOT met |

- 1 stream, mean ms per frame: read 11, B 84, E 3 (total 98). B is the cost.
- Total throughput saturates at about 13.7 to 13.8 FPS from 3 streams on: adding streams only splits the same GPU/process capacity.
- nvidia-smi dmon (2 s samples): in the 79 samples with non-zero SM use, mean SM 87.6 %, max 100 %; peak framebuffer 1953 MB of 20 GB (memory is not the limit). The other 116 samples are idle time before and between passes.
- Not measured / caveats: fp16, other resolutions, separate processes per stream, a run alongside teammates' jobs. The thread-based design (one Python process, GIL) may add overhead that real multi-process streams would not; not tested.
Alternatives & why not: no Kaggle interim number (a T4 result says little about this GPU); E-6 not started (needs approval).
Open issues:
- GAP: 3 and 4 concurrent streams are below 5 FPS per stream (4.43 and 3.32 lowest). 1 stream passes at 2x the target.
- Recommendation for E-6 (one change at a time, re-measure FPS AND F1 each time): fp16 for B first, then frame_stride 2 (contract-compatible), then lower net resolution (likely costs F1). 4 streams at 5 FPS needs about 20 FPS total, i.e. roughly 1.5x more than the saturated 13.7.
- Team latency (<= 3 s) is not covered by this test.
