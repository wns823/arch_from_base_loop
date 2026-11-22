---
name: harness-status
description: Show the state of the ag_harness project — which cases are calibrated (gate verdicts), which harness runs exist and their progress, GPU lease, and the next step per VALIDATION.md stage gates. Use when the user asks for status or progress.
---
Run `harness/bin/agh status` and summarise: per case gate verdict and budget; per run candidates done / N, best S_val, SSA versions, stop reason; stage-gate progress (A4000 → Spark → H100, VALIDATION 4.3); GPU lease holder. Do not read ~/ag_harness_private beyond what `agh status` prints.
