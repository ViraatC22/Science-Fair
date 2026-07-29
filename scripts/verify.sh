#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

python_bin="${PYTHON:-python3}"
cache_root="${TMPDIR:-/tmp}/slime-mold-sim-verify"
export PYTHONPYCACHEPREFIX="$cache_root/pycache"
export MPLCONFIGDIR="$cache_root/matplotlib"

"$python_bin" -m pip check
"$python_bin" -m compileall -q backend Streamlit_App tests
"$python_bin" -m unittest discover -s tests -v
