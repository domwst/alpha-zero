# RunPod RTX 3090 deployment and Kata benchmark — 2026-09-01

## Decision

The RTX 3090 pod is suitable for the current Kata training run. Relative to the
eight normal completed Kata epochs on the Nebius L40S host, it delivered 43.2%
fewer evaluations per second and took 68.8% longer per epoch. At one quarter of
the hourly price, that is still about 2.27× more evaluations per dollar and
2.37× more completed epochs per dollar for this measured workload.

The host has ample capacity for one Kata trainer: peak pod memory was 50.57 GiB
of a 116.42 GiB limit, GPU memory peaked at 1,384 MiB of 24 GiB, and the CPU
cgroup did not add any throttled periods during the benchmark. The workload is
not VRAM-bound and averages modest CPU use; its main limitation remains small,
serial MCTS inference requests and loss of batch fill in the end-of-epoch tail.

## Persistent deployment

The repository, checkpoints, run data, toolchains, dependency caches, virtual
environment, and build output live below `/workspace/alpha-zero` or sibling
paths below `/workspace`; they survive a pod restart.

Two scripts make a fresh container reproducible:

- `scripts/runpod_env.sh` redirects Cargo, rustup, uv, and local binaries into
  `/workspace`.
- `scripts/setup_runpod.sh` installs OS prerequisites, uv, Python 3.14, the
  pinned `nightly-2026-08-27` Rust toolchain, locked Python dependencies, and
  builds/tests the release executable.

The bootstrap completed with Rust 1.100 nightly, uv 0.12.8, Python 3.14.7,
PyTorch 2.13.0+cu130, and a healthy CUDA allocation. The native release binary
SHA-256 was
`a27af37832bc22373e3c673c423e5145c5f66e60838283473d205c4b254df711`.

## Host

- GPU: NVIDIA RTX 3090, 24 GiB, 350 W limit, PCIe Gen4 x16 capability.
- Driver/CUDA: 580.126.20 / CUDA 13 runtime.
- CPU: AMD EPYC 7H12 host, 256 logical CPUs visible, 27.2-vCPU cgroup quota.
- Memory: 116.42 GiB cgroup limit.
- Persistent filesystem: `/workspace`.

The benchmark was NUMA-pinned to CPU IDs 64–95 (the GPU-local NUMA node), with
16 Tokio workers and one OpenMP/MKL thread.

## Controlled preflight

All values use Kata v1, CUDA, and batch 256. Medians are used for the isolated
three-repeat measurements.

| Workload | Nebius L40S | RunPod RTX 3090 | Change |
| --- | ---: | ---: | ---: |
| Isolated inference | 122,684 examples/s | 34,595 examples/s | -71.8% |
| Isolated optimizer | 26,549 samples/s | 11,905 samples/s | -55.2% |
| 500 games × 16 simulations, P500 | 38,491 eval/s | 22,646 eval/s | -41.2% |
| Same shallow self-play | 2,422 moves/s | 1,425 moves/s | -41.2% |

The RTX 3090 inference repeats varied from 31.7k to 47.2k examples/s, so the
isolated inference ratio is less representative than self-play. Tokio 16 won
the shallow comparison by 2.9% over Tokio 24 and was retained.

## Complete production epoch

The RunPod process restored Kata checkpoint 18 and ran exactly epoch 19 with
700 games, 2,000 simulations per move, B256/P500, a 1 ms partial-batch timeout,
and training batch 256. Checkpoint 19 was successfully written and restored
metadata identifies `kata_v1`.

For a stable baseline, the Nebius column is the median of completed epochs
10–12 and 14–18. The stalled epoch 13 is excluded.

| Metric | Nebius L40S median | RunPod RTX 3090 epoch 19 | Change |
| --- | ---: | ---: | ---: |
| Evaluations/s | 26,200 | 14,881 | -43.2% |
| Moves/s | 15.81 | 9.05 | -42.8% |
| Self-play time | 1,238.8 s | 2,097.9 s | +69.4% |
| Optimizer samples/s | 16,233 | 9,795 | -39.7% |
| Optimizer time | 26.72 s | 43.35 s | +62.3% |
| Whole epoch | 1,269.0 s | 2,142.4 s | +68.8% |

RunPod epoch 19 performed 31,218,903 evaluations across 18,986 moves. At the
matched 480-second heartbeat it sustained 26,352 eval/s, versus 40,458 eval/s
for the interrupted Nebius epoch-19 attempt: a 34.9% reduction while both had
hundreds of active games. The final aggregate fell to 14,881 eval/s because the
last few unusually long games ran at tiny batches. Average batch size was 80.68.

This is why normalized throughput agrees with the estimated 35–45% slowdown,
while wall-clock duration is about 69% longer: a 43% lower rate implies roughly
1.76× elapsed time for equal work, and the exact game trajectories are not
bitwise stable across the two GPU/software builds.

## Resource telemetry

- Pod memory peak: 50.57 GiB (43.4% of the 116.42 GiB limit).
- GPU memory peak: 1,384 MiB; self-play normally used about 776 MiB.
- Whole-run GPU utilization: 33.3% average, 80% maximum.
- Dense self-play window: 35.4% average GPU utilization and 225.7 W average
  power.
- Optimizer window: 66.0% average GPU utilization, 300.7 W average power.
- Maximum temperature/power: 64 °C / 335.3 W.
- CPU throttling: zero additional throttled periods during the benchmark.

There is no thermal, VRAM, RAM, or CPU-quota blocker. Raising raw GPU batch size
would not address the low-fill tail; the most promising scheduling experiment
for future epochs is starting all 700 games concurrently, since memory headroom
is sufficient and it would avoid starting the second 200-game wave late.

## Artifacts

- Remote run: `/workspace/alpha-zero/runs/kata-v1-50e-20260901`
- Remote benchmark segment:
  `/workspace/alpha-zero/runs/kata-v1-50e-20260901/segments/20260901T194130Z-26184`
- Remote preflight: `/workspace/alpha-zero/runs/rtx3090-preflight-20260901`
- Local mirror: `remote-results/runpod-rtx3090-20260901`
- Checkpoint-19 model SHA-256:
  `a3075649dff086d29154bcb0ea10da074ca0f51fa3f0153f77ab31873a155afd`
