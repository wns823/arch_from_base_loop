#!/usr/bin/env bash
# Copy case data from a server that has it (case data is not in the repo).
# usage: harness/bin/sync_data.sh <user@host>:<repo path on that host> <case>... [--to <dir for big data>]
#   e.g. harness/bin/sync_data.sh me@labserver:/path/to/ag_harness V2 V6
#        harness/bin/sync_data.sh me@labserver:/path/to/ag_harness V3 --to /data/$USER/ag_data
# Small cases are copied into cases/<case>/data. A case whose data is a symlink on the source (V3: 81 GB cache) is
# copied into <dir>/<case>_data (default: ./data_store) and cases/<case>/data is pointed at it; the sandbox binds the
# symlink target automatically.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
SRC=$1; shift
TO="$ROOT/data_store"; CASES=()
while [ $# -gt 0 ]; do
  case $1 in --to) TO=$2; shift 2 ;; *) CASES+=("$1"); shift ;; esac
done
host=${SRC%%:*}; path=${SRC#*:}
for c in "${CASES[@]}"; do
  target=$(ssh "$host" "readlink -f '$path/cases/$c/data'")
  if [ "$target" = "$path/cases/$c/data" ]; then
    mkdir -p "cases/$c/data"
    rsync -a --info=progress2 "$host:$target/" "cases/$c/data/"
  else
    mkdir -p "$TO/${c}_data"
    rsync -a --info=progress2 "$host:$target/" "$TO/${c}_data/"
    rm -f "cases/$c/data"; ln -s "$(cd "$TO/${c}_data" && pwd)" "cases/$c/data"
  fi
  echo "$c: $(du -sh "cases/$c/data/" | cut -f1)"
done
