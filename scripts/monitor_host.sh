#!/usr/bin/env bash
set -euo pipefail

output=${1:?usage: monitor_host.sh OUTPUT.csv [INTERVAL_MS] [WATCH_PID]}
interval_ms=${2:-1000}
watch_pid=${3:-}
mkdir -p "$(dirname -- "$output")"
interval_seconds=$(awk -v milliseconds="$interval_ms" 'BEGIN { print milliseconds / 1000 }')

printf '%s\n' \
  'timestamp,load_1m,mem_total_kib,mem_available_kib,swap_total_kib,swap_free_kib,watched_rss_kib' \
  >"$output"

while [[ -z "$watch_pid" || -r "/proc/$watch_pid/status" ]]; do
  mem_total=0
  mem_available=0
  swap_total=0
  swap_free=0
  while read -r key value _unit; do
    case "$key" in
      MemTotal:) mem_total=$value ;;
      MemAvailable:) mem_available=$value ;;
      SwapTotal:) swap_total=$value ;;
      SwapFree:) swap_free=$value ;;
    esac
  done </proc/meminfo

  read -r load_1m _rest </proc/loadavg
  watched_rss=0
  if [[ -n "$watch_pid" && -r "/proc/$watch_pid/status" ]]; then
    while read -r key value _unit; do
      if [[ "$key" == 'VmRSS:' ]]; then
        watched_rss=$value
        break
      fi
    done <"/proc/$watch_pid/status"
  fi

  printf '%s,%s,%s,%s,%s,%s,%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "$load_1m" "$mem_total" "$mem_available" \
    "$swap_total" "$swap_free" "$watched_rss" >>"$output"
  sleep "$interval_seconds"
done
