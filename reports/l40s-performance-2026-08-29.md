# L40S performance tuning report — 2026-08-29

## Decision

Use the memory-safe target-depth scheduler for training, while retaining the
throughput-only optimum as an explicitly measured reference:

| Setting | Former baseline | Throughput optimum | Production long run |
| --- | ---: | ---: | ---: |
| Inference batch size | 128 | 128 | 128 |
| Concurrent games | 160 | 384 | **160** |
| Partial-batch timeout | 100,000 µs | 5,000 µs | **100,000 µs** |
| Tokio worker threads | automatic (16 here) | 16 | 16 |
| Training batch size | 1,024 | 256 | **256** |

The controlled self-play comparison used identical seeds, 384 games, and 32
simulations per move. The throughput optimum's median was 17,604 evaluations/s,
versus 16,686 evaluations/s for the former baseline: a 5.50% improvement.
Target-depth launches then showed continued host-memory growth at P384 and P256
on this 62 GiB/no-swap VM. P256 measured only 0.8% faster than P160, so production
uses the three-repeat P160 point for substantially greater multi-day headroom.

The primary limitation is not L40S compute or GPU memory. This 10-block,
32-channel network is small, while each game performs MCTS serially. Nsight and
`nvidia-smi` show CPU-side search, batch formation, synchronization, and many
small CUDA launches leaving the GPU mostly idle. Deep MCTS tree expansion does,
however, make host RAM the practical limit on game parallelism.

## Environment

- Workload: 19×19 five-in-a-row, 10 residual blocks, 32 channels.
- Host: Nebius VM, Intel Xeon Gold 6338, 8 physical cores / 16 logical CPUs,
  62 GiB RAM.
- GPU: NVIDIA L40S, 46,068 MiB, 350 W limit.
- Driver: 580.173.02.
- PyTorch: 2.13.0+cu130; CUDA runtime 13.0.
- Nsight Systems: 2025.3.2.474.
- Source base: `021a5f726d6deed9bcc26bf7d90c74cfc2b4dbd2`, plus the changes described here.

All throughput candidates were warmed before measurement. Inference and training
batches were repeated three times. Scheduler finalists and the former baseline
were repeated three times with identical per-repeat seeds. Optimizer/OpenMP
thread counts were held at one so they did not compete with MCTS workers.

## Changes made

### Measurement and reproducibility

- Added `benchmark inference`, `benchmark training`, and `benchmark self-play`
  commands with machine-readable JSON output.
- Added network-batcher metrics: invocation/request counts, batch histogram,
  cancellations, queue and request latency, transfer/forward/service time.
- Added per-epoch JSON and append-only `epochs.jsonl` metrics with phase timing,
  normalized value/policy losses, replay size, and scheduler statistics.
- Replaced entropy-created per-game RNGs with deterministic, independently
  derived epoch/game/training/render seeds.
- Added reproducible sweep, GPU-monitoring, Nsight, and profiled-run scripts.
- Added continuous host load, available-memory, and swap telemetry
  after target-depth MCTS exposed host RAM as a production constraint.
- Added time-based training heartbeats so multi-hour self-play phases are visibly
  alive before the first game finishes.

### Training behavior and operations

- Decoupled game parallelism, inference batch size, and batch timeout. The prior
  batch-padding/ramp coupling made these controls difficult to measure or reason
  about independently.
- Changed production defaults to batch 128, 160 concurrent games, 100 ms
  timeout, and training batch 256. The measured B128/P384/5 ms throughput
  optimum remains documented for hosts with more RAM or shallower searches.
- Corrected training losses to be sample-weighted and normalized, including the
  policy cross-entropy reduction.
- Made `--epochs N` an absolute, resume-safe target. Restoring epoch 17 with
  `--epochs 50` now trains epochs 18–49 rather than 50 additional epochs.
- Added a segmented long-run wrapper. A restart preserves all earlier stdout/GPU
  telemetry while reusing the checkpoint, replay, game, and stats directories.
- Adapted `run.sh` for Linux CUDA/PyTorch in the Nix environment: locked `uv`
  setup, Torch/NVIDIA runtime paths, Nix `libstdc++`, system CUDA driver and zlib,
  required Torch CUDA preloads, release rpath, safe quoting, and dynamic discovery
  of the virtual environment's Python version.

## Results

### Raw inference

This includes host-to-device input transfer, forward execution, device-to-host
outputs, and synchronization.

| Batch | Median examples/s | Median iteration |
| ---: | ---: | ---: |
| 1 | 866 | 1.155 ms |
| 16 | 13,330 | 1.200 ms |
| 32 | 26,697 | 1.199 ms |
| 64 | 51,342 | 1.247 ms |
| 128 | 95,546 | 1.340 ms |
| 256 | **128,012** | 2.000 ms |
| 512 | 104,952 | 4.878 ms |
| 1,024 | 72,475 | 14.129 ms |
| 2,048 | 30,546 | 67.046 ms |
| 4,096 | 29,239 | 140.085 ms |

Batch 256 is the isolated network peak. It is not the self-play choice because
MCTS cannot keep batches of that size full efficiently.

### Self-play scheduler

The final validation used 384 games × 32 simulations and three repeats.

| Configuration | Eval/s repeats | Median | Median batch fill |
| --- | --- | ---: | ---: |
| Former: B128 / P160 / 100 ms / 16 workers | 16,667; 16,686; 16,800 | 16,686 | 77.8% |
| B64 / P384 / 1 ms / 16 workers | 17,061; 17,325; 17,564 | 17,325 | 95.7% |
| **B128 / P384 / 5 ms / 16 workers** | **17,456; 17,604; 17,838** | **17,604** | **89.5%** |
| B128 / P384 / 5 ms / 8 physical workers | 17,346; 17,542; 17,582 | 17,542 | 89.3% |

The throughput finalist is 5.50% faster than the former baseline. Eight physical-only
workers were 0.35% slower than all 16 logical CPUs, so SMT stays enabled.

Timeout interacts with batch size and concurrency. At B128/P256, 100 ms could
form fuller batches and did well, but at the throughput-optimal P384 concurrency
it added latency and reduced throughput. At P384, B128/5 ms reached 17,462
eval/s in the interaction sweep versus 16,237 at 100 ms; B64 similarly favored
1 ms.
The B128/P256/100 ms point reached 16,819 eval/s in the one-factor sweep—only
0.8% above the controlled P160 median of 16,686. Production uses P160 because
that negligible gain does not justify 60% more concurrently retained MCTS trees.

### Optimizer batch size

The synthetic optimizer benchmark includes CPU-to-GPU copies, forward, backward,
Adam update, and loss synchronization. Values are medians of three repeats.

| Batch | Samples/s | Step time | Peak observed sweep memory |
| ---: | ---: | ---: | ---: |
| 128 | 15,508 | 8.254 ms | — |
| **256** | **28,109** | **9.107 ms** | — |
| 512 | 27,156 | 18.854 ms | — |
| 1,024 | 22,313 | 45.892 ms | — |
| 2,048 | 13,915 | 147.176 ms | — |
| 4,096 | 13,576 | 301.702 ms | — |
| 8,192 | 13,557 | 604.262 ms | 17,589 MiB |

Batch 256 is the throughput choice. Real replay training in the pilot reached
16,858 samples/s because state/policy augmentation and sample construction are
also included there; training remained a small fraction of total epoch time.

## GPU utilization and Nsight interpretation

Coarse 200 ms telemetry across the repeated finalist workload averaged 15.9%
GPU utilization, peaked at 23%, and used at most 915 MiB. The optimizer sweep
averaged 70.3%, reached 100%, and used at most 17,589 MiB. This separates a
self-play scheduling bottleneck from a GPU capacity problem.

The representative Nsight capture recorded 245,233 evaluations. Profiler
overhead reduced its throughput, so it is diagnostic only:

- Summed CUDA kernel time was about 2.07 s over 23.65 s wall time (~8.7%).
- Host-to-device copies totaled 31.8 ms and device-to-host copies 18.9 ms.
- CUDA API statistics recorded 371,328 regular kernel launches plus 36,637
  extended launches; launch API time was about 2.16 s in aggregate.
- The largest kernel groups were cuDNN Winograd convolution (23.7%) and TF32
  implicit GEMM convolution (22.7%).

The actionable conclusion is to supply enough independent games to form batches
and avoid long partial-batch waits. A much larger gain would require an
architectural change—such as concurrent leaf searches within a game or a fused/
compiled network—not another increase in L40S batch size.

## End-to-end pilot

One complete epoch used 128 games, 64 simulations, and the throughput-optimal settings:

| Phase | Result |
| --- | ---: |
| Self-play | 59.54 s; 13,561 eval/s |
| Replay training | 6.04 s; 101,800 augmented samples; 16,858 samples/s |
| Checkpoint | 0.48 s |
| Render 2 games | 0.03 s |
| Whole epoch | 66.09 s |

The snapshot restored successfully. Reissuing the command with the same
`--epochs 1` target exited without creating epoch 1. The first snapshot was
7.3 MiB (2.18 MB model, 4.43 MB optimizer, 0.95 MB compressed replay), so
50 retained snapshots should fit comfortably on the 201 GiB disk.

## Long-run configuration and expectations

`scripts/run_long_training.sh` launches 50 total epochs, 700 games/epoch, 2,000
simulations/move, deterministic seed 20260829, replay capacity 1,800 games, and
five rendered games/epoch. It records one-second GPU telemetry, 60-second
heartbeats, append-only epoch metrics, and restart-preserving run segments.

At roughly 100 moves/game, one target epoch requires about 140 million network
evaluations. The P160 median of 16.7k eval/s implies an optimistic 2.3 hours
of self-play per epoch, or about 4.8 days for 50 epochs. MCTS tree-selection cost
grows at 2,000 simulations, so 5–8 days is a more responsible initial range.
The first target epoch will replace that estimate with an observed rate.

### Launch status

The run root is `/home/arch/alpha-zero/runs/training-50e-20260829` and the tmux
session is `alpha-zero-train`.

The first segment started at 21:37:02 UTC with the throughput-optimal P384/5 ms
scheduler. It emitted eight uninterrupted heartbeats and kept about 14 CPU cores
and the GPU active. Target-depth tree allocation nevertheless continued past an
apparent plateau: by 8:47 it had reached 50.9 GiB RSS with only 10.0 GiB host
memory available and no swap. It was stopped cleanly at 21:47:00 with status 130,
before OOM and before producing any checkpoint. Its log and telemetry remain in
`segments/20260829T213702Z-74034`.

The second preflight segment started from epoch 0 at 21:49:16 UTC using
P256/100 ms:

- segment: `segments/20260829T214916Z-78932`
- training PID at launch: 79173
- exact scale: 50 epochs, 700 games/epoch, 2,000 simulations/move

Its five-minute checkpoint showed five uninterrupted heartbeats, about
1,360% process CPU, 13.1% average GPU utilization (17% maximum), at most 899 MiB
GPU memory, and 33.1 GiB process RSS. By 7:20, however, RSS had continued to
35.2 GiB. P256 was only 0.8% faster than the controlled P160 median while
retaining 60% more simultaneous trees, so it was stopped cleanly at 21:57:50
with status 130 and no checkpoint. Its artifacts remain in
`segments/20260829T214916Z-78932`.

The final production segment started from epoch 0 at 22:00:00 UTC using
B128/P160/100 ms:

- segment: `segments/20260829T220000Z-84338`
- training PID at launch: 84650
- exact scale: 50 epochs, 700 games/epoch, 2,000 simulations/move

Its five-minute health checkpoint showed five uninterrupted heartbeats, about
1,404% process CPU, 13.7% average GPU utilization (15% maximum), at most 899 MiB
GPU memory, and 22.5 GiB process RSS. Available host memory remained at least
38.5 GiB. Zero games during the initial watch is expected because the first wave
advances through many 2,000-simulation moves together. GPU and host telemetry
continue at one-second resolution inside the segment directory.

Status can be checked without attaching to tmux:

```bash
cd /home/arch/alpha-zero
scripts/check_training_status.sh runs/training-50e-20260829
```

## Production-depth B192/P500 trial (2026-08-30)

After the MCTS node-state removal reduced tree memory, training resumed from
checkpoint 39 for two full production-depth epochs with inference batch 192,
500 games of parallelism, and the existing 100 ms partial-batch timeout. The
timeout, model, optimizer, replay buffer, seed schedule, and all other training
settings were held constant. Both epochs completed and produced checkpoints 40
and 41 before the process was stopped.

| Metric | B128/P160 epochs 36–39 | B192/P500 epochs 40–41 | Change |
| --- | ---: | ---: | ---: |
| Evaluations/s | 30,213 | 30,003 | -0.7% |
| Moves/s | 19.132 | 19.031 | -0.5% |
| Actual batch size | 75.38 | 89.34 | +18.5% |
| Queue wait | 2.38 ms | 4.46 ms | +87.3% |
| Request latency | 3.63 ms | 8.74 ms | +140.7% |
| Network service time | 1.52 ms | 1.72 ms | +12.9% |

The two trial epochs individually reached 30,629 and 29,376 evaluations/s.
During their high-concurrency waves they sustained roughly 40–41k evaluations/s,
but the gain disappeared as the longest games became low-batch stragglers. Only
35.9% of network invocations used the full batch of 192, so increasing the
nominal maximum did not produce a correspondingly large average batch.

Memory is no longer the limiting factor for P500: maximum observed process RSS
was about 33.0 GiB, minimum available host memory was 27.9 GiB, and no swap was
used. It is still an expensive trade: the preceding P160 segment retained at
least 47.3 GiB of available host memory. Whole-segment average GPU utilization
also fell slightly from 30.7% to 29.4%, while average one-minute host load was
essentially unchanged (3.16 versus 3.29).

The production decision is therefore to reject B192/P500/100 ms as the default.
It is memory-safe, but the larger batches and task population increase latency
without improving normalized self-play throughput. The long-run runner now
keeps B128/P160/100 ms as its defaults while accepting `INFERENCE_BATCH_SIZE`,
`GAMES_PARALLELISM`, and `BATCH_TIMEOUT_US` environment overrides for explicit
future trials. Raw artifacts are in segment
`segments/20260830T155700Z-484086` under the existing run root.

## Validation and raw artifacts

- `cargo test`: 30 passed, 0 failed on the server.
- `cargo clippy --all-targets -- -D warnings`: passed.
- `cargo build --release`: passed with release debug symbols.
- Python CUDA tensor and convolution smoke tests: passed.
- Rust CUDA benchmark, end-to-end epoch, checkpoint restore, JSON/JSONL metrics,
  rendered games, and GPU monitor: passed.

Raw JSON, logs, telemetry, and the 32 MiB `.nsys-rep` were copied into the
ignored local `remote-results/` tree. The redundant 99 MiB Nsight SQLite export
remains at `/home/arch/alpha-zero/runs/exp-nsys-final-20260829-2/alpha-zero.sqlite`.
Sweep definitions live in `scripts/`; methodology is in
`docs/experiment-methodology.md`.
