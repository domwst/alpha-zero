# BATCH-001: active producers and fixed batch grids

**Decision:** resume production with active-producer dispatch and unpadded batches.
Keep the bucket grids optional. The new system improved this controlled MCTS
workload; neither grid improved its average throughput further or reduced peak
allocator reservation. This is not a claim of a 39% production self-play speedup.

## Production-precision MCTS results

Three repeats per arm, with the order rotated. Each repeat played 96 games using
the same checkpoint for both seats, S256/P100/B64, a 1 ms batch deadline, and
sampling temperature 0.7. Seeds were matched between arms within a repeat. Game
lengths and useful inference counts closely matched: about 238–255 thousand
evaluations per trial and 10.6–11.3 moves/game.

| Executor | Mean games/hour | Range across repeats | Mean useful evaluations/s |
|---|---:|---:|---:|
| Previous executable | 6,151 | 5,930–6,386 | 4,426 |
| New, unpadded | **8,573** | 8,117–9,074 | **6,181** |
| New, coarse grid | 8,550 | 8,119–8,849 | 6,146 |
| New, dense grid | 8,422 | 7,450–9,499 | 6,048 |

The unpadded new system improved average games/hour by **39.4%** and mean useful
inference throughput by **39.6%**. It was faster than the old executable in every
repeat. The old/new comparison includes the controller, executor, telemetry, and
archive implementation; it cannot attribute the whole improvement solely to the
activity guards.

The coarse and dense grids changed mean games/hour by −0.3% and −1.8% relative to
new unpadded execution. Their repeat ranges overlap substantially. The shortest
individual run used the dense grid, but that was not a repeatable overall win.
Three repeats are descriptive evidence, not a precise confidence interval.

## Strict-FP32 executor workloads

These isolate repeated inference requests and producer lifetime changes. They use
**strict FP32 with TF32 disabled**, unlike the production-precision MCTS runs.
Do not directly transfer their absolute timings or allocator results to production
precision. Heavy workloads used P400/B256; the demo workload used two producers
with the same B256 ceiling and dropped/reacquired submission guards. All used a
1 ms deadline, identical inputs, and three rotated repeats.

| Workload | Task-count adaptation | Active, unpadded | Coarse grid | Dense grid |
|---|---:|---:|---:|---:|
| Steady demand, evaluations/s | 34,382 | **38,546** | 37,489 | 33,396 |
| Declining concurrency, evaluations/s | **29,948** | 25,125 | 28,052 | 28,721 |
| Two-user workload, evaluations/s | 346 | **653** | 630 | 604 |
| Two-user p95 request latency, ms | 5.21 | **3.00** | 3.07 | 3.34 |

The active dispatcher helps steady demand and especially interactive demand. The
old task-count reduction did better on this synthetic declining-concurrency
workload. Padding recovered some tail throughput but did not produce a better
MCTS average. The admission target during a long tail remains a worthwhile tuning
question. No change to that policy was smuggled into these measurements.

The coarse grid was `1,2,4,8,16,32,64,128,256`; the dense grid added
`48,80,96,112,160,192,224`. MCTS trials used the same grids truncated at B64.
Padding accounted for 11.9% and 3.5% of executed rows in the coarse and dense tail
workloads. Padding never counted as useful work.

## Allocator profiles

Separate profiling processes used the P400/B256 declining-concurrency workload.
These counters are from PyTorch's **native CUDA allocator in strict FP32**; they
are not total device memory or host/MCTS memory.

| Grid | Peak allocated MiB | Peak reserved MiB | Peak inactive split MiB | Device allocations |
|---|---:|---:|---:|---:|
| Unpadded | 171.6 | **236** | 84.1 | **21** |
| Coarse | 203.5 | 268 | **77.2** | 25 |
| Dense | 181.6 | 272 | 99.5 | 24 |

Every profile recorded **zero allocation retries and zero OOMs**. The coarse grid
slightly reduced peak inactive split bytes, while increasing allocated/reserved
memory and device allocation count. The dense grid increased all three memory
peaks. There is no evidence here that grids solve the prior production pauses,
which involved host RAM. Production-precision allocator behavior would require a
separate profile before making a stronger claim.

## Numerical gate and amendment

The initial CUDA preflight against the trained checkpoint failed its single-row
reference check: maximum value error 0.004533 and policy error 0.001286. Controlled
diagnostics established:

| Diagnostic | Maximum value error | Maximum policy error |
|---|---:|---:|
| Default precision, unpadded | 0.004533 | 0.001286 |
| TF32 disabled, padded | 1.19e-7 | 2.09e-7 |
| TF32 disabled, unpadded | 1.19e-7 | 2.09e-7 |
| Default precision, single request | 0 | 0 |

The discrepancy was reproduced without padding and disappeared when TF32 was
disabled. [PyTorch documents the global TF32 override](https://docs.pytorch.org/docs/main/cuda_environment_variables.html).
The benchmark plan was explicitly amended: executor timing/correctness/allocator
trials use `NVIDIA_TF32_OVERRIDE=0`; closed-loop MCTS retains production precision.
The original tolerances—2e-4 for value, 2e-5 for policy—were not relaxed. All amended
numerical trials passed, including finite outputs and zero illegal-move probability.
The failed attempt, original plan, diagnostic outputs, and amendment are retained.

Training precision was not changed. Accuracy sensitivity to TF32 on real replay
positions is a separate possible investigation.

## Reproduction and limits

- RTX 3090, driver 580.126.20; Torch 2.13.0+cu130, CUDA 13.0, cuDNN 9.20.0.
- `tch-rs` revision `07f5604711b5bd9acd3ad501dbdf76fa758720a4`.
- Model: displayed epoch 41, native checkpoint 40, SHA-256
  `12b62bb89dd4b594ab676bd5a1f1abfef98b619b9fba70d06f9ed7a7caa2b282`.
- New executable SHA-256:
  `0f2fd775cd058a623f447fd077bb5f5b00c4f0063f746295e7791a70b59ee30a`.
- Old executable SHA-256:
  `6b66eb466dc8bf833dabbb509b36d9c9a852a61f202c5c5e538fbbb27075c567`.
- OMP/MKL threads: 1; Tokio worker threads: 16. Fresh processes per trial.
  Executor references and ten maximum-batch warmups are excluded from timed work.
- 52 logical jobs: one preflight, 36 executor timing trials, 12 MCTS comparisons,
  and three allocator profiles. The initial failed preflight is a preserved extra
  attempt. The isolated TF32 diagnostics are recorded separately.
- The environment query was collected during the first timing repeat. There was
  no concurrent training or replay conversion; no claim is made that these short
  measurements eliminate all shared-host or scheduling variation.

The service owns the trial dependencies, launch parameters, process identities,
logs, and results. Scripts live in `scripts/benchmarks/`. The pod retains
`/workspace/alz-job-service/experiments/BATCH-001-20260912/`, including `trials.json`,
`summary.json`, and environment/diagnostic records. The ignored local cutover
folder contains `BATCH-001-results.tar.gz` (149,411,885 bytes), SHA-256
`8ca1285bdac478fa45255fc0b63fb5260be6f8ea4f632230b40a779d5217096c`.
It contains raw results, attempts, comparison games, both executables, and the
pinned model.

Production continuation uses **S3000/P400/B256**, 1,000 games/epoch, top-p 0.95,
the existing replay-linked LR schedule, and no bucket grid. Its first complete
epoch serves as an operational check at the production simulation budget; a
single new epoch is not a controlled old/new speed comparison.
