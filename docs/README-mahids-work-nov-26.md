# Mahid's work, 26 November 2026

Branch `feat/ml-speed-e6`, pull request #11 (https://github.com/Sentinel-CMSD-FYP/Sentinel/pull/11): CI green, **not merged yet**. Work done in Ahmed's folder on the lab GPU PC (RTX 4000 Ada).

## 1. Done today
| Item | Result | Details |
|---|---|---|
| Supervisor document on the tracking count target | Written; includes that the draft said 25% and 35% was set after a first test result (38.7%). Measured 37.9% | `docs/tracking-count-target-issue.md`, short note `docs/supervisor-note-tracking-target.md` |
| PR #10 | Merged into `main` with a merge commit (`4f5e4bb`) | |
| E-6 step 1, fp16 for Model B | Opt-in, off by default. B+E lowest stream avg FPS: 1 stream 10.01 -> 12.98, 3 streams 4.50 -> 5.65, 4 streams 3.33 -> 4.35 (target 5: still missed). Val F1 0.8635 -> 0.8629 | `docs/decisions/0014-e6-speed-protocol.md` |
| Latency, all five models (target <= 3 s) | Met on a replayed clip: points median 1.15 s (1 stream), 1.38-1.40 s (3), 1.52-1.54 s (4); worst 2.07 s | `docs/decisions/0015-latency-1-stream.md` |
| 10 minute live run, 1 stream | No delay drift; stream worker memory grew about 31 MB per minute (3740 -> 4020 MB) | decision 0015 |
| Repeatable latency scripts | `eval/pipeline/run_latency.sh`, `latency_headless.py`, `latency_workers_local.py` | smoke-run once |

## 2. Limits (say these with the numbers)
- Latency excludes any drone video link or network; recordings replayed on one PC, headless browser.
- With 4 streams B and E process only about 1 frame in 9 to 10; with 1 stream about 2 in 5. **Tracking accuracy at live frame rates has not been measured.**
- fp16 was not tried on the test split or for tracking metrics; the team pipeline still uses fp32.
- Validation F1 is optimistic (the released detector trained on those clips).
- Current focus is one stream; multi-stream work is future work.

## 3. Open, next session
1. Merge PR #11 (merge commit or rebase-and-merge, not squash).
2. Get the supervisor's answer: 25% or 35%, night/dusk scope, signature.
3. Frame-skip tracking test on validation (proposed, not started; no GPU needed).
4. Update `docs/report-corrections.md`: its speed sentences predate fp16 and its latency text still says "not measured".
5. Stale "RTX 4070" text in `README.md`, `docs/tier1_ticklist.md`, `README-DMCOUNT.md` (shared or Ahmar's).
6. Ask for: the report's LaTeX path; numeric targets for Models C, D and geometry.

## 4. Tell the team
- Pipeline owner: memory growth in long live runs (likely samples kept in memory to save the clip; not tested).
- Ahmar: the `tier1-models` GitHub release was not found; `docs/run_tier1.md` depends on it.
- GitHub reported 23 vulnerabilities on `main` (4 critical); not looked into.

## 5. Working from another computer
```bash
git clone https://github.com/Sentinel-CMSD-FYP/Sentinel.git && cd Sentinel
git checkout feat/ml-speed-e6        # or main after PR #11 is merged
```
Reading documents and CPU tests work anywhere. **Not in git** (only on the lab PC): `CLAUDE.md`, `ROADMAP.md`, the report PDF, `docs/cc-prompt-immediate.md`, everything under `data/` (dataset, weights, raw results, clips), `ml/third_party/`, `.venv`, Node. GPU runs need the lab PC.

## Earlier work
`docs/README-2026-11-24-session.md` (IM-1 to IM-5) and `docs/project-log.md`.
