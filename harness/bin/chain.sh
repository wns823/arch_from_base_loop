#!/usr/bin/env bash
# Run N independent harness runs of one case back to back: init-run → loop (search + finalize) → one RESULT line each.
# usage: harness/bin/chain.sh <case> [policy=ucb] [n_runs=3] [prefix=fid]
# runs are named <prefix>_<policy>_r<n>; log: reports/<prefix>_<case>_<policy>.log
set -u
case=$1; pol=${2:-ucb}; n=${3:-3}; pre=${4:-fid}
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
AGH="$ROOT/harness/bin/agh"
log=reports/${pre}_${case}_${pol}.log
mkdir -p reports
for r in $(seq 1 "$n"); do
  run=${pre}_${pol}_r$r
  echo "=== $case $run start $(date +%F' '%T)" >> "$log"
  AGH_POLICY=$pol "$AGH" init-run "$case" "$run" > /dev/null 2>&1 || { echo "init failed $run" >> "$log"; continue; }
  "$AGH" loop "$case" "$run" > "runs/$case/$run/loop.log" 2>&1 || echo "loop error $run" >> "$log"
  python3 - "$case" "$run" >> "$log" 2>&1 <<'PY'
import json, sys
c, r = sys.argv[1:3]
f = json.load(open(f"runs/{c}/{r}/final.json"))
rt = f.get("ratio_to_target")
print("RESULT", c, r, "best", f["best"], "n", f["n_candidates"], "S_val", round(f["S_val"], 4),
      "S_test", f["S_test"] and round(f["S_test"], 4), "ratio", rt and round(rt, 3), f.get("verdict"),
      "leak", (f.get("leak") or {}).get("leak"))
PY
done
echo "=== $case $pol chain done $(date +%F' '%T)" >> "$log"
