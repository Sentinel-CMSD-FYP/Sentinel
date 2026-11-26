# Note for the supervisor: tracking count-error target

**Measured.** 37.9% with the 45-frame filter (35.2% without). DroneCrowd test, 30 clips, 5,092 people, frozen settings, second look, not blind.

**Target history.** The draft Phase 0 sheet said 25%. On 18 October, after a first test result of 38.7%, we moved to 35% with 25% as stretch. We miss both.

**Cause.** Mainly Model B's head detections. Perfect identities (an oracle, not a real tracker) raise IDF1 only from 0.388 to 0.454.

**Plan.** Improve B in Tier 2, with a success rule and Kaggle-hours cap fixed before training.

**Decisions needed.**
1. Target in the signed Phase 0 sheet: 25%, or 35% with 25% stretch?
2. Approve night out of scope (not excluded at kickoff) and dusk reported separately?
3. Sign the sheet.

Details: `docs/tracking-count-target-issue.md`.
