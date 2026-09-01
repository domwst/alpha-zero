#!/usr/bin/env bash
set -euo pipefail

run_root=${1:?usage: check_training_status.sh RUN_ROOT}
run_root=$(cd -- "$run_root" && pwd)
segment_file="$run_root/current-segment.txt"
segment_dir=$(<"$segment_file")

printf 'run_root=%s\nsegment=%s\n' "$run_root" "$segment_dir"
printf '\nProcesses:\n'
pgrep -af '/target/release/alz train' || true

printf '\nRecent training output:\n'
tail -n 12 "$segment_dir/stdout.log"

printf '\nRecent GPU telemetry:\n'
tail -n 6 "$segment_dir/gpu.csv"

if [[ -s "$segment_dir/host.csv" ]]; then
  printf '\nRecent host telemetry:\n'
  tail -n 6 "$segment_dir/host.csv"
fi

printf '\nHost memory:\n'
free -h

printf '\nGPU now:\n'
nvidia-smi \
  --query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu \
  --format=csv,noheader

if [[ -s "$run_root/stats/epochs.jsonl" ]]; then
  printf '\nLatest completed epoch:\n'
  tail -n 1 "$run_root/stats/epochs.jsonl" | jq '{
    epoch,
    games,
    self_play_seconds,
    evaluations_per_second,
    moves_per_second,
    network_average_batch_size,
    training,
    checkpoint_seconds,
    epoch_seconds
  }'
else
  printf '\nNo epoch has completed yet.\n'
fi
