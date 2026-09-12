#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/runpod_env.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKIO_WORKER_THREADS=16
export CARGO_BUILD_JOBS=2 CARGO_TARGET_DIR="$PWD/target"
./run.sh cargo build --release --locked --bin alz
./run.sh cargo test --release --locked --workspace
fork_manifest=$(find "$CARGO_HOME/git/checkouts" -path '*/7e7d436/Cargo.toml' -print -quit)
test -n "$fork_manifest"
./run.sh cargo test --release --offline --manifest-path "$fork_manifest" --test fused_adam_tests --no-run
PYTHONPATH=scripts python3 -m unittest scripts/test_pooling_followup.py scripts/test_replay_followups.py scripts/test_handoff_fallback.py scripts/test_experiment_dashboard.py
python3 - "$fork_manifest" <<'PY'
import datetime, json, sys
from pathlib import Path
sys.path.insert(0, 'scripts')
from experiment_io import digest, write
repo = Path.cwd()
write(repo / 'pooling-validation-plan.json', {
    'architecture_command': ['bash', str(repo / 'scripts/run_cuda_tests_runpod.sh'), 'architecture'],
    'cache_command': ['bash', str(repo / 'scripts/run_cuda_tests_runpod.sh'), 'cache'],
    'backend_command': ['bash', str(repo / 'scripts/run_cuda_tests_runpod.sh'), 'backend', sys.argv[1]],
})
write(repo / 'pooling-build-ready.json', {
    'completed_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'binary_sha256': digest(repo / 'target/release/alz'),
    'validation_plan_sha256': digest(repo / 'pooling-validation-plan.json'),
    'cuda_checks': 'scheduled after value-head queue; no GPU work during build',
})
PY
