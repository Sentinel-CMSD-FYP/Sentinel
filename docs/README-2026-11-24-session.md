# Session summary, 2026-11-23 to 2026-11-24 (temporary file)

Temporary hand-over note for the "do immediately" block (`docs/cc-prompt-immediate.md`, tasks IM-1 to IM-5). Delete or fold into `docs/project-log.md` when no longer needed. Every number below comes from a command that was run; sources are in the linked files.

## What was done
| Task | Result | Where |
|---|---|---|
| IM-1 B+E speed on the RTX 4000 Ada | 1 stream 10.20 FPS avg (p95 10.12), target 5: met. 3 streams lowest 4.43 (p95 3.59) and 4 streams lowest 3.32 (p95 2.74): gap. DroneCrowd test clips, 250 timed frames, fp32, GPU idle, no teammates' jobs | `docs/decisions/0012-e5-speed.md`, `eval/benchmark_be.py` |
| IM-2 honest status table and report wording | Every Tier 1 row is Met, Gap or Not measured, with a source; exact replacement sentences for the report | `docs/tier1-status.md`, `docs/report-corrections.md` |
| IM-3 supervisor note | 182 words; tracking count error 37.9% filtered (35.2% unfiltered) vs 35% target (25% stretch); cause is Model B's detections (oracle adds only 0.066 IDF1). **Waiting for the supervisor's answer** | `docs/supervisor-note-tracking-target.md` |
| IM-4 ID-switch analysis, clips 22 to 24 | TrackEval's 510, 579, 558 switches reproduced exactly and are real (about 2 per person, about 2 tracks per person); the 0.5 IoU threshold is not the cause; ground-truth heads about 7 px apart vs a 20 px box (hypothesis, untested). ID stability is now "Met by eye only" | `eval/analyse_id_switches.py`, `ml/configs/id_switches.yaml`, `docs/decisions/0013-id-switch-analysis.md` |
| IM-5 housekeeping | E-1, E-4, E-5 ticked in the local `ROADMAP.md`; `origin/main` merged (no conflicts); `ruff` clean; pytest 457 passed, 8 skipped, 2 data-dependent tests pass in the GPU container; PR #10 opened, CI green | PR #10 |

## Setup changes made along the way
- GitHub login on this PC: `mah1dkund1` is active (token has no `workflow` scope); `Ahmed-Ijaz-A` is logged in but inactive.
- Repo git identity: `mah1dkund1 <mahidk595@gmail.com>`. Two earlier commits (`76e8cbb`, `3d79655`) keep the author `Ahmed Ijaz`.
- `CLAUDE.md` (local, gitignored): tracking target row now reads "35% (25% stretch), pending supervisor sign-off".
- `docs/decisions/0001` to `0013` are now committed (main dropped the gitignore line). Numbers 0003, 0006, 0007 are also used by different files from main.

## Open for next session
1. Merge PR #10 (https://github.com/Sentinel-CMSD-FYP/Sentinel/pull/10): use a merge commit or rebase-and-merge, not squash.
2. Take the supervisor note to the supervisor: 25% or 35% in the signed Phase 0 sheet. Then update `CLAUDE.md` and `docs/tier1-status.md`.
3. Stale "RTX 4070" text in `README.md`, `docs/tier1_ticklist.md` (also still marks speed and ID stability as passing) and `README-DMCOUNT.md`. Shared or Ahmar's files; propose edits first.
4. E-6, the speed fix for 3 to 4 streams (fp16, then frame stride 2, then lower resolution; one change at a time, re-measure FPS and F1). Needs approval.
5. The latency test (not part of IM-1 to IM-5; `main` now has the pipeline and `test_live_latency.py`).
6. Questions for Ahmed: path to the report's LaTeX source; numeric targets for Models C and D and geometry.
7. Not done: speed with teammates' jobs running; renumbering the duplicate decision numbers; validation-only proposals in decision 0013.
