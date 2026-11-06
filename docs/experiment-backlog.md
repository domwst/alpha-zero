# Experiment backlog

This is the queue of proposed measurements, not the GPU execution queue. Entries
require their prerequisites and an explicit scheduling decision before consuming
pod resources. Completed experiments should link their report and record the
resulting decision here.

See [experiment methodology](experiment-methodology.md) for common controls and
[execution and analysis design](execution-and-analysis-design.md) for the agreed
interfaces.

## SAMPLING-001: Nucleus move sampling

- Status: archived / deprioritized. [Decision and results](../reports/nucleus-self-play-20260914/README.md).
- Routine default: top-p 1.0; lower cutoffs remain available for reproduction.
- The E17-versus-E54 comparison changes temperature schedule and training age,
  so it does not establish a causal benefit or failure of nucleus sampling.
- Reopening requires a matched training comparison and a new scheduling decision.

## BATCH-001: Active producers and fixed inference batch buckets

- Status: completed (52 jobs). [Results](batching-experiment-20260912.md).
  Decision: active-producer dispatch with unpadded production batches; grids remain
  optional. Full-epoch operational validation follows during continuation.
  The legacy RAM guard interrupted epoch 42 after collection but before saving its
  checkpoint; epoch 41 is the last complete state and is pinned for these trials.
- Added: 2026-09-12.
- Prerequisites: submission activity guards, configurable batch buckets/padding,
  and executor dispatch/allocation instrumentation. Activity guards and opt-in
  buckets and allocation instrumentation are deployed and measured.
- Question: do a bounded set of tensor batch shapes and active-producer-aware
  dispatch improve useful throughput, latency, or memory stability?

Changing allocation sizes can interfere with CUDA allocator reuse. This is a
hypothesis to measure for our workload, not an established explanation of its
memory behavior. The September 12 self-play pause crossed the **host RAM** guard;
a reduction in CUDA fragmentation would not by itself explain or fix that pause.
Background: [PyTorch CUDA memory management](https://docs.pytorch.org/docs/main/notes/cuda.html#memory-management).

### Comparisons

Separate the effect of activity tracking from tensor-shape bucketing:

1. Existing task-count batch adaptation with unpadded partial batches.
2. Active-producer-aware dispatch with unpadded partial batches.
3. The same activity policy with padded execution on a fixed bucket grid.

Within (3), compare a coarse grid (for example powers of two) with a denser grid
near the measured throughput knee. Keep maximum batch size and memory budget
matched. Treat grid choices as candidates, not defaults already selected.
Allocator configuration experiments, if needed, form a separate controlled arm;
do not change the allocator and padding policy together and attribute the result
to only one of them.

### Workloads and controls

- Pin model/checkpoint, device, precision, input positions, binary/library versions,
  timeout, and policy configuration. Record every effective batch-size decision.
- Use a repeatable inference request workload to isolate executor behavior, then
  validate finalists with closed-loop MCTS and a complete self-play epoch at the
  production simulation budget.
- Exercise steady high concurrency, a long tail of declining concurrency, and
  demo-like demand: idle sessions, two searches becoming one, cancellation, and
  dropping/reacquiring producer guards.
- Exercise comparisons with separate checkpoint executors; measure per-turn
  activity notifications before introducing optimizations to avoid them.
- Use fresh processes for cold-memory comparisons and long warm runs for allocator
  reuse. Repeat finalists at least three times and vary run order.
- Run allocation profiling separately from the primary throughput measurements.

### Measurements and acceptance

- Useful evaluations/second and games/hour; padded rows never count as work done.
- Request latency p50/p95/p99, queue wait, dispatch reasons, real/padded batch-size
  histograms, and the fraction of execution spent on padding.
- CUDA allocated/reserved/peak bytes, allocation retries and fragmentation-related
  statistics where supported, plus host RSS/cache measurements kept separate.
- GPU allocation activity and synchronization costs; producer lifecycle-notification
  overhead; MCTS/controller time versus neural inference time.
- Validate real outputs against unpadded evaluation within an explicit numerical
  tolerance. Verify padding never changes legal-action mapping or publishes dummy
  outputs, and guard cancellation/drop cannot stall pending work.
- Select settings using the latency/throughput/memory tradeoff. Do not select solely
  by GPU utilization or assume that fewer shapes guarantee less memory usage.

Deliverable: reproducible benchmark configuration, raw metrics, a short report,
and a decision on bucket grids, dispatch policy, and any needed allocator API.

### September 12 numerical gate amendment

The trained epoch-41 checkpoint exposed TF32-dependent differences between
single-row and batched inference: value error 0.00453 and policy error 0.00129,
identical with and without padding. Disabling TF32 reduced both errors below
3e-7. The original failed attempt and four diagnostic runs are retained.

Executor timing and allocator trials therefore explicitly use
`NVIDIA_TF32_OVERRIDE=0` (`disable-tf32` in the typed benchmark spec). All closed-loop
MCTS trials retain production-default precision, including the old-binary baseline.
The two sets of timings must not be treated as interchangeable. Allocator
conclusions from the executor profiles apply to strict FP32; production-precision
memory behavior still needs separate confirmation before attributing any OOM to
CUDA fragmentation. This is an explicit amendment, not a relaxed numerical tolerance.
