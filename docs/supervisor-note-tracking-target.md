# Note for the supervisor: tracking count-error target

**Measured.** Tracking count error is 37.9% with the 45-frame filter (35.2% without it). Source: DroneCrowd test split, 30 clips, 5,092 people, run-2 frozen settings, second look on test, not blind.

**Target.** The report says 35%, with 25% as a stretch goal. Both are missed. Tracking quality is IDF1 0.388, HOTA 0.300.

**Cause.** It is the head detections from Model B, not the tracker. When we gave the tracker perfect identities (an oracle, not a real tracker), IDF1 rose only from 0.388 to 0.454 (+0.066). On test, B finds too few people on night, dusk and large-scale clips. Source: project log, 18 Oct, and report Table 7.5.

**Plan.** Improve B in Tier 2 (train on our own split). Before any training we fix a success rule and a cap on Kaggle GPU hours, so the effort is bounded. Night stays out of scope; dusk is reported separately.

**Decision needed.** Which count-error target goes into the signed Phase 0 sheet: 35% or 25%? Our draft says 35% with 25% as stretch, and the sheet is unsigned.
