# Concurrent training smoke test — 7 September 2026

Completed: **1.997× aggregate training throughput** with two trainers.
See the [results and resource measurements](../reports/concurrent-training-smoke-2026-09-07.md).

The user requested a disposable training run for approximately two epochs
alongside the active capacity experiment, to measure total throughput.

`scripts/concurrent_training_smoke.py` waits for two completed standalone
epochs of the active 16-block × 32-channel model, then starts a fresh copy
for exactly two epochs. Both use the same executable, seed, replay sources
60–69, dataset split, batch size 256, standard Adam, learning rate 0.001,
weight decay 0.0001, GELU, original pooling, deep value head, and device replay
cache. MPS configuration is unchanged. The production queue continues running.
Disposable checkpoints are excluded from all architecture comparisons.

Pod output:
`/workspace/alpha-zero-followups/runs/value-heads-20260906/capacity-concurrency-smoke-20260907`.
The independent tmux session is `alz-concurrency-smoke-20260907`.
The probe stops after two epochs; its supervisor also stops it if the main
queue changes stage or the probe exceeds one hour. Only the probe process
group is terminated by that supervisor.

The main run's first two completed epochs provide the standalone baseline.
Native epoch JSON records training samples and training duration. The report
uses total samples divided by total duration within each group of epochs,
rather than averaging rounded rates. Main epochs that are not wholly inside
the probe's training window are excluded from concurrent throughput estimates.
The sum of the two concurrent worker rates is compared with the solo rate.
This is a steady-state smoke estimate, not an exact sample count in one common
wall-clock interval. Validation, checkpoint writes, initial data loading, and
cache preparation are excluded from the native training-duration metric;
probe process duration and logs retain those overheads separately.

GPU utilization, allocated memory, power, and temperature are sampled every
five seconds during the probe. Existing capacity telemetry provides the solo
reference. Results should be interpreted together with per-worker slowdown,
memory use, and the number of fully overlapping main epochs. Increased GPU
utilization alone does not establish increased useful throughput.
