# Tier 1 demo checklist (sprint [3-6])

About 10 minutes. Results to quote: `docs/tier1_ticklist.md`. Setup on a new machine: `docs/run_tier1.md`.

## Before (the day before)

* [ ] Demo clips present and valid: `data/samples/seq011`, `seq023`, `seq070`, `seq071` (and `live_demo48`
  for the live part). Check: `python docs/layers/validate.py data/samples/seq011` (each clip). Rebuild:
  `python eval/pipeline/build_demo_clips.py --device cuda`, or the Kaggle kernel in `docs/pipeline.md`.
* [ ] Optional: move test only clips (`own_footage_test`, `live_2026*`, `dronecrowd_test_23`) out of
  `data/samples/` so the clip list shows the demo set only.
* [ ] All three test suites green (`docs/run_tier1.md` section 6).
* [ ] Browser window not minimised during the live part (a minimised Chrome draws at ~1.5 frames/s).

## Start

1. Terminal 1, backend (repo root first):
   `cd apps/backend; ..\..\.venv\Scripts\python manage.py runserver`
   On the GPU machine set `PIPELINE_RUNNER=local` and `PIPELINE_DEVICE=cuda` first; on this CPU PC keep
   the default runner (Kaggle) only if an upload is shown, and set `KAGGLE_USERNAME=ahmar004`,
   `PYTHONUTF8=1`.
2. Terminal 2, frontend: `cd apps/frontend; npm run dev`, open http://localhost:5173.
3. Check: the clip list loads, no errors in the browser console.

## Show, in this order

1. **seq011 (day, test 11), Single view.** Play. Default layers: density (heat map), infrastructure
   (polygons), tracks. Hover a polygon (class and score). Turn on points and flow, then off again.
   Say: A counts dense zones, B points count sparse and moderate ones; C finds walls and vehicles.
2. **seq011, Split by model.** One panel per model, same frame. Say: every model reaches the frontend
   through the same `Module` interface and layer contract.
3. **seq023 (dense, GT about 257).** Single view. Density close to GT (first sample 270); dense is the
   bucket that meets its target (8.0% against 25%).
4. **seq070 (night).** Show the failure openly: density about 5 against GT 42, few tracks. Night is out
   of scope for Tier 1 and is a Tier 2 gap.
5. **seq071 (dusk, val 71).** Reported without a target (A 19.3% undercount on val dusk).
6. **Live.** Live page, "Go live with this clip" on `live_demo48`, open it in Single view. Point at the
   latency: on a GPU every model but flow is under 3 s; on this CPU PC density and points come 5 to
   9 s late, so they show as gaps (the gaps rule, never stale values). Stop the stream.
7. **Upload (optional, only on the GPU machine or with time to spare).** Upload a short clip with
   altitude and FOV; the run takes about 1.5 min on a GPU, about 5 min through Kaggle.
8. **Numbers.** Open `docs/tier1_ticklist.md`: what passes, what fails, what waits for the 4070.

## After

* [ ] Stop live streams (Live page, Stop) and both servers (Ctrl+C).
* [ ] Recording of the run (31b, on the GPU machine).
