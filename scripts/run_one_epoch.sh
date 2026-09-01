#!/usr/bin/env bash
set -euo pipefail

run_root=${1:?usage: run_one_epoch.sh RUN_ROOT}
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck source=runpod_env.sh
source "$repo_dir/scripts/runpod_env.sh"

run_root=$(cd -- "$run_root" && pwd)
latest_checkpoint=$(
  find "$run_root/checkpoints" -mindepth 1 -maxdepth 1 -type d \
    -printf '%f\n' | awk '/^[0-9]+$/' | sort | tail -1
)
if [[ -z "$latest_checkpoint" ]]; then
  printf 'no numeric checkpoint found below %s/checkpoints\n' "$run_root" >&2
  exit 1
fi

latest_epoch=$((10#$latest_checkpoint))
# `--epochs` is an exclusive upper bound. Resume starts at latest_epoch + 1.
export EPOCHS=$((latest_epoch + 2))
export TOKIO_WORKER_THREADS=${TOKIO_WORKER_THREADS:-24}

cpu_list=${RUNPOD_CPU_LIST:-}
if [[ -n "$cpu_list" ]]; then
  exec taskset --cpu-list "$cpu_list" \
    "$repo_dir/scripts/run_long_training.sh" "$run_root"
fi

exec "$repo_dir/scripts/run_long_training.sh" "$run_root"
