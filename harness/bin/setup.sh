#!/usr/bin/env bash
# One-time setup on a new server: venv + packages, private dir, then `agh doctor`.
# usage: harness/bin/setup.sh            (uses uv if installed, else python3 -m venv)
#        TORCH_INDEX=https://download.pytorch.org/whl/cu121 harness/bin/setup.sh   (pick a CUDA build)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PYVER="${PYVER:-3.12}"
if [ ! -x .venv/bin/python ]; then
  if command -v uv >/dev/null; then uv venv --python "$PYVER" .venv
  else python3 -m venv .venv; fi
fi
PIP=(.venv/bin/python -m pip)
if ! .venv/bin/python -m pip --version >/dev/null 2>&1; then
  command -v uv >/dev/null && PIP=(uv pip --python .venv/bin/python) || .venv/bin/python -m ensurepip
fi
if [ -n "${TORCH_INDEX:-}" ]; then "${PIP[@]}" install torch --index-url "$TORCH_INDEX"; fi
"${PIP[@]}" install -r requirements.txt
mkdir -p -m 700 "$HOME/ag_harness_private"
mkdir -p runs reports
chmod +x harness/bin/agh harness/bin/chain.sh harness/bin/sync_data.sh
echo "--- doctor"
harness/bin/agh doctor || { echo "fix the FAIL lines above"; exit 1; }
