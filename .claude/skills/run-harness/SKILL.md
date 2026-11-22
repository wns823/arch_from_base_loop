---
name: run-harness
description: Conductor procedure for one harness run on one validation case (or Stage 2). Loops allocator → exploiter/explorer subagent → runner, re-invokes the search-space agent on stagnation/saturation/max-interval, stops by rule, then finalizes (test once, grade by code). Use when the user asks to run the harness on a case (a built-in case or the user's own case made with `agh new-case`).
argument-hint: "<case> [run_id]"
---

The run is driven by **loop code**, not by an agent (REQ-016). The main session only starts it and reports.

1. `harness/bin/agh init-run <case> [run_id]`: creates `runs/<case>/<run>/` (workspace, boundary.json, LEDGER) and trains the base as candidate 000.
2. `harness/bin/agh loop <case> <run>` (Python, long-running; start it in the background or tmux). Each iteration:
   - allocator (code) decides: SSA, a candidate (exploiter or explorer, with parent and target), or stop
   - SSA → `claude -p --agent search-space` writes `space_v<k>.json` → the loop validates it against the boundary (re-asks at most twice if invalid)
   - candidate → `claude -p --agent exploiter|explorer` writes `cands/<id>/` → **the loop automatically** checks the boundary and size limits, trains in the sandbox with the fixed budget, computes S_val, and updates the LEDGER and coverage (a failed candidate is handed back to a new agent call with the error log, at most twice)
   - stop rule (code): budget N, or 2 SSA rounds without improvement
   - after the loop: **automatically** finalize (best S_val → S_test once)
3. `final.json` already holds the grade (code only): S_test, ratio to target, verdict, leak check (reproduce mode). `harness/bin/agh judge <case> <run>` only re-prints it (G when a gate verdict exists).
   For a new case run `harness/bin/agh check-case <case>` first (no agents; size check, short training, time estimate).
4. Report to the user: S_val trajectory, S_test vs target, which modules the winner changed (`best/`), agent contributions, failures, leaks (`harness/bin/agh status <case> <run>`). The user can steer a running search by writing `runs/<case>/<run>/steer.md`.

Never rerun finalize or judge to change a number.
