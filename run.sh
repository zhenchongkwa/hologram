#!/usr/bin/env bash
# Launch the Hologram Glitch Band from Git Bash / WSL / macOS / Linux.
#   ./run.sh                 start it
#   ./run.sh --effect vhs     any of run.py's arguments pass straight through
set -euo pipefail
cd "$(dirname "$0")"

# Windows venvs put the interpreter in Scripts/, everywhere else it's bin/.
if [ -x ".venv/Scripts/python.exe" ]; then
    PY=".venv/Scripts/python.exe"
elif [ -x ".venv/bin/python" ]; then
    PY=".venv/bin/python"
else
    cat >&2 <<'EOF'

Python environment not found at .venv

Create it with:
    py -3.12 -m venv .venv                                    (Windows)
    python3.12 -m venv .venv                                  (other)
    .venv/Scripts/python.exe -m pip install -r requirements.txt

EOF
    exit 1
fi

if [ ! -f "models/hand_landmarker.task" ]; then
    echo "Hand landmark model missing - downloading it now..."
    "$PY" scripts/download_model.py
    echo
fi

cat <<'EOF'
 Hologram Glitch Band
 ---------------------------------------------------------------
  Raise BOTH hands. The shape is drawn through four fingertips:
  each hand's thumb and index finger. Move them to reshape it.

  TAP thumb to index on BOTH hands  -  next effect
  n  next mode - shape, cube, slit-scan
  b  straight to cube mode - a 3D box, open both palms for a rose
  g  bloom - bright areas bleed light
  [ ]  prev/next      p  one shape per hand      s  screenshot
  r  record           ?  all keys                q  quit
 ---------------------------------------------------------------

EOF

exec "$PY" run.py "$@"
