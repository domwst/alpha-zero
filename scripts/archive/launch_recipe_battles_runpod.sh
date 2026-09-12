#!/usr/bin/env bash
set -uo pipefail
cd /workspace/alpha-zero-battles || exit 1
source scripts/runpod_env.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKIO_WORKER_THREADS=16
run="$PWD/runs/recipe-battles"
exec 9>"$run/.launcher-lock"
flock -n 9 || exit 1
bash scripts/monitor_gpu.sh "$run/gpu.csv" 1000 &
gpu_monitor=$!
cleanup() { kill "$gpu_monitor" 2>/dev/null || true; }
trap cleanup EXIT
date -u +%Y-%m-%dT%H:%M:%SZ > "$run/launcher-started-at.txt"
python3 -u scripts/archive/run_recipe_battles.py --output-dir "$run" --binary "$PWD/validated/alz"
result=$?
printf '%s\n' "$result" > "$run/exit-code.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$run/finished-at.txt"
if [ "$result" -ne 0 ]; then
  python3 - <<'PY'
import sys
sys.path.insert(0, 'scripts')
from experiment_io import read,write
from pathlib import Path
path=Path('runs/recipe-battles/status.json')
if not path.exists() or read(path).get('stage') != 'failed':
 write(path, {'stage':'failed', 'error':'Queue validation failed; see launcher.log'})
PY
fi
exit "$result"
