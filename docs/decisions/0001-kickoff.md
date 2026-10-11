# 0001 Team kickoff decisions
Date: 2026-10-11   Owner: Ahmed
Context: P0-1. Ahmed brought the kickoff checklist to Mahid and Ahmar. These are the answers as Ahmed reported them.

Decision:
1. **Dataset:** DroneCrowd is primary. UP-COUNT / UP-COUNT-TRACK is secondary. Team uses the same copy and folder layout (all three questions: yes).
2. **Train/val/test split:** UNDECIDED. Leading option is the official split, if the dataset has one (checked in P0-3). Not frozen yet. Which clips go in validation is also undecided. Rule from CLAUDE.md stays: tune on validation only, touch test once at the end.
3. **JSON formats:** OK as proposed in `docs/contracts.md`. If Mahid or Ahmar need extra fields, add them later. Claude flags it if this becomes a blocker.
4. **GPU schedule:** No fixed schedule. Kaggle is a possible fallback for GPU work. If something cannot run on Kaggle, decide then. The 4070 stays one heavy job at a time.
5. **Night footage:** Not explicitly out of scope. If the dataset has night frames they are used for training and testing. No night footage is added on purpose.
6. **Decision support only:** Ahmed was unsure what this meant. It means the system gives early warnings to a human operator and does not act on its own. It is a framing for the project, not a task for B or E. Ahmed left it to Claude and the project docs.
7. **Point F1 target:** UNDECIDED. The Phase 0 sheet has no number. We report it, and Ahmed may set one with the team later.

Evidence: Ahmed's reply in session on 2026-10-11. No commands run.

Alternatives & why not: None discussed.

Open issues:
- Split must be frozen before B-1 (data loader). P0-3 will show whether official split files exist.
- Altitude source for E is undecided. Check in P0-3 whether the dataset has altitude.
- Kaggle limits (GPU hours, internet, session length) not checked. Not yet needed.
- `contracts.md` is still PROPOSED until P0-5.
