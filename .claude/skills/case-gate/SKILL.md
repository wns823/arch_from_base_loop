---
name: case-gate
description: Calibrate one validation case before any harness run — pilot the training budget, run H vs A0 (3 seeds), per-item ablations (2 seeds), reachability check, V0-neg check — and record the verdict. Deterministic scripts only, no LLM decisions. Use when preparing a case listed in validation/VALIDATION.md.
argument-hint: "<case>"
---

This is test calibration, not the harness. No subagent chooses anything here.

1. Pilot: run H and A0 at increasing `train_samples` (2 seeds) until mean(S_H) − mean(S_A0) > 2σ_seed; the smallest such budget becomes `budget.yaml: train_samples`.
2. Gate: H and A0 × 3 seeds at that budget → usable iff the gap > 2σ_seed (VALIDATION 4.1).
3. Per-item ablations of H × 2 seeds → items with drop > 2σ are "detectable" (denominator of R).
4. Reachability: H must be a diff on the base inside the editable files and within size limits.
5. Freeze: record H's params and FLOPs (size limits = 1.25×), the budget and the verdict in `~/ag_harness_private/gate/<case>/verdict.json`; freeze the answer key if not yet frozen.
Gate scripts live in `~/ag_harness_private/gate/<case>/` and run through `harness/run_job.py` (sandbox, GPU lease).
