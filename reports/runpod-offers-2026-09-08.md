# RunPod cost review — 2026-09-08

Read-only account and availability queries; no pod was created, stopped or
deleted. The API key was loaded from `.env` in memory and not included in
requests' URLs, command arguments or saved responses.

## Live shortlist

These are on-demand `uninterruptablePrice` quotes, not spot minimum bids.
The current pod reports an L40 at $0.69/hour. Storage is additional to the
compute comparisons below; this is not a reconstruction of the account invoice.

| GPU | Cloud | VRAM, GB | USD/hour | API-reported minimum RAM / vCPUs | Assessment |
| --- | --- | ---: | ---: | --- | --- |
| RTX A4500 | Community | 20 | 0.19 | 62 GB / 14 | Cheapest plausible single trainer; speed unmeasured |
| RTX 3090 | Community | 24 | 0.22 | 62 GB / 32 | First candidate for a cost benchmark |
| RTX 4090 | Community | 24 | 0.34 | 107 GB / 20 | Alternative with more host RAM; speed unmeasured |
| A40 | Secure | 48 | 0.49 | 50 GB / 9 | VRAM for two trainers, but host RAM/CPU need checking |
| Existing L40 | Existing pod | 48 | 0.69 | Not queried in this audit | Two trainers already benchmarked |

All four new-pod offers above pass `minCudaVersion: "13.0"`, at least
32 GB host RAM and 8 vCPUs. All returned `Low` stock. These are filtered
GPU-type quotes, not reserved machines; host CPU model, actual cgroup limits,
disk capacity and final availability must be checked at deployment.
The response fields `minMemory` and `minVcpu` describe the filtered offer pool,
not a benchmark or a selected physical machine.

A stricter 64 GB RAM filter excluded the 3090 and A4500 because the quoted
allocations are 62 GB. That arbitrary cutoff is not a reason to reject them
for a single trainer. The 4090 passes 64 GB RAM / 8 vCPUs unchanged.
None of these four new offers passed the conservative 128 GB / 16 vCPU
filter for overlapping large MCTS comparisons at query time. This does not
establish that 128 GB is necessary; actual peak RSS should determine that.

An advertised V100 at $0.23/hour is not a drop-in alternative for our CUDA 13
stack: CUDA 13 removed Volta compilation/library support. Avoid the 16 GB
offers for this initial comparison because the existing cached trainer has
little memory margin there.

## Cost per experiment, not per GPU-hour

Our [concurrent-training measurement](concurrent-training-smoke-2026-09-07.md)
found 1.997x aggregate training throughput from two L40 workers. GPU memory
peaked at 14,817 MiB with one worker and 29,630 MiB with two. With the existing
separate full GPU replay caches, budget for one worker on a 20–24 GB device.
Sharing/reducing caches could change this, but would introduce a separate
performance experiment.

Let one L40 worker's throughput be 1.0, and retain the measured dual-worker
scaling as an estimate. A new single-worker GPU beats the occupied L40's
training throughput per dollar only above these rates:

| Candidate | Minimum relative throughput |
| --- | ---: |
| A4500 | 55.0% |
| 3090 | 63.7% |
| 4090 | 98.4% |

Calculation: `candidate_price / 0.69 * 1.997`. These are break-even
thresholds, not predictions. Include loading, validation and checkpoint I/O
when measuring whole-run cost. A 4090 must be more than `0.34 / 0.22 = 1.545x`
as fast as a 3090 on the same complete job to offer lower compute cost.
Two A40 workers would need 71% of the L40's aggregate throughput; their
host-memory fit and concurrency scaling have not been measured.

The [older 3090 benchmark](runpod-rtx3090-benchmark-2026-09-01.md) supports
testing this card, but compared an earlier ReLU/self-play workload against
Nebius L40S. Its old price advantage and throughput ratios cannot be applied
directly to the current GELU/device-cache workload and L40 dual-worker setup.

As an illustrative budget, if a 20-pass baseline run takes two hours per
L40 worker, two concurrent runs cost about $1.38 in compute. For the same
pair run sequentially on a 3090, cost is `$0.88 / r`, where `r` is its
whole-run speed relative to one L40 worker: $1.47 at r=0.6, $1.26 at r=0.7,
or $0.88 at r=1.0. These are scenarios, not measured ETAs; transfers,
environment setup, storage and battles are excluded.

## Applying this to the requested experiments

The requested scope is findings (2), normalization diagnostics, and (3),
learning-rate scheduling, from the [unresolved-findings review](unresolved-findings-2026-09-08.md).

1. Diagnose existing frozen checkpoints: comparable train/validation losses,
   batch versus stored BN statistics on clones, per-layer scales and value
   saturation. Recalibrate running statistics with training data only and
   evaluate held-out data independently. No fresh 20-pass model is needed
   for these measurements. Keep original checkpoints intact.
2. Compare constant LR against a specified decay schedule with architecture,
   normalization, replay split, data exposure and checkpoint selection fixed.
   Lower starting LR, warmup and weight averaging are separate follow-ups,
   rather than changes silently bundled into the schedule comparison.
3. Replicate promising learning changes before spending the battle budget on
   many one-seed candidates. Diagnostics can narrow the candidates but do
   not replace playing-strength evaluation.

Recommendation: use the existing pod for short diagnostic measurements while
it remains allocated; ten minutes costs $0.115 at its reported hourly rate.
For recurring training, benchmark the current cached trainer on the $0.22
3090 first, measuring complete pass duration and peak GPU/host memory.
Measure representative MCTS separately before migrating battle work.
The L40 remains a reasonable option for pairs of trainers already running
concurrently. Leaving it allocated for an idle day costs $16.56 in compute,
which can exceed the savings from moving a small training batch.

## Sources and retained responses

- [RunPod GraphQL specification](https://graphql-spec.runpod.io/): pricing
  filters and response field definitions.
- [RunPod pricing](https://docs.runpod.io/pods/pricing): per-second compute
  billing, storage charges and no ingress/egress fees.
- [NVIDIA CUDA 13 release notes](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html): Volta support removal.
- Local raw responses: `runs/runpod-offers-20260908/current-pods.json`,
  `gpu-types.json`, `filtered-gpu-types.json`, and `constraint-gpu-types.json`.
  No credentials are contained in these responses.
