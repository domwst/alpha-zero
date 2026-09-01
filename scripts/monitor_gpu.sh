#!/usr/bin/env bash
set -euo pipefail

output=${1:?usage: monitor_gpu.sh OUTPUT.csv [INTERVAL_MS]}
interval_ms=${2:-200}
mkdir -p "$(dirname -- "$output")"

printf '%s\n' \
  'timestamp,index,gpu_util_percent,memory_util_percent,memory_used_mib,power_w,sm_clock_mhz,memory_clock_mhz,temperature_c,pstate' \
  >"$output"

exec nvidia-smi \
  --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,power.draw,clocks.current.sm,clocks.current.memory,temperature.gpu,pstate \
  --format=csv,noheader,nounits \
  --loop-ms="$interval_ms" >>"$output"
