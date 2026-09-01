#!/usr/bin/env bash
set -euo pipefail

run_dir=${1:?usage: run_profiled.sh RUN_DIR COMMAND [ARG ...]}
shift
if (($# == 0)); then
  echo "a command is required" >&2
  exit 2
fi

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
mkdir -p "$run_dir"
run_dir=$(cd -- "$run_dir" && pwd)
monitor_interval_ms=${GPU_MONITOR_INTERVAL_MS:-200}
if [[ -z ${ALZ_LOG_ANSI+x} ]]; then
  if [[ -t 1 ]]; then
    export ALZ_LOG_ANSI=always
  else
    export ALZ_LOG_ANSI=auto
  fi
fi

{
  printf 'started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'host=%s\n' "$(hostname)"
  printf 'working_directory=%s\n' "$repo_dir"
  printf 'git_commit=%s\n' "$(git -C "$repo_dir" rev-parse HEAD 2>/dev/null || printf unknown)"
  printf 'git_status=%s\n' "$(git -C "$repo_dir" status --short 2>/dev/null | tr '\n' ';')"
  if [[ -f "$repo_dir/target/release/alz" ]]; then
    printf 'alz_binary_sha256=%s\n' "$(sha256sum "$repo_dir/target/release/alz" | awk '{ print $1 }')"
  fi
  printf 'command='
  printf '%q ' "$@"
  printf '\n'
  printf 'tokio_worker_threads=%s\n' "${TOKIO_WORKER_THREADS:-auto}"
  printf 'omp_num_threads=%s\n' "${OMP_NUM_THREADS:-auto}"
  printf 'mkl_num_threads=%s\n' "${MKL_NUM_THREADS:-auto}"
  printf 'gpu_monitor_interval_ms=%s\n' "$monitor_interval_ms"
  printf 'log_ansi=%s\n' "$ALZ_LOG_ANSI"
  nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,power.limit --format=csv,noheader
  "$repo_dir/run.sh" python -c \
    'import torch; print(f"torch={torch.__version__}"); print(f"torch_cuda={torch.version.cuda}"); print(f"cuda_available={torch.cuda.is_available()}")'
} >"$run_dir/manifest.txt"

"$repo_dir/scripts/monitor_gpu.sh" "$run_dir/gpu.csv" "$monitor_interval_ms" &
gpu_monitor_pid=$!
"$repo_dir/scripts/monitor_host.sh" "$run_dir/host.csv" "$monitor_interval_ms" &
host_monitor_pid=$!
cleanup() {
  kill "$gpu_monitor_pid" "$host_monitor_pid" 2>/dev/null || true
  wait "$gpu_monitor_pid" 2>/dev/null || true
  wait "$host_monitor_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

cd "$repo_dir"
set +e
RUST_BACKTRACE=1 "$repo_dir/run.sh" "$@" 2>&1 | tee "$run_dir/stdout.log"
status=${PIPESTATUS[0]}
set -e

printf 'finished_utc=%s\nexit_status=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$status" >>"$run_dir/manifest.txt"
exit "$status"
