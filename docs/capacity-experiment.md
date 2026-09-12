# Trunk depth versus width — 2026-09-07

The user requested six additional **residual blocks** (twelve 3×3
convolutions), and chose to retain two global-pooling blocks to isolate depth.
The experiments keep GELU, original global pooling, the 64 → 64 → 64 → 1
value MLP, and the existing policy-head width.

| Model | Residual blocks | Trunk channels | Global-block indices | Value convolution | Architecture |
| --- | ---: | ---: | --- | --- | --- |
| Reused baseline | 10 | 32 | 3, 7 | 32 → 32 | `kata-gelu-value64x2-v1` |
| Deeper | 16 | 32 | 3, 7 | 32 → 32 | `kata-gelu-b16c32-value64x2-v1` |
| Wider | 10 | 48 | 3, 7 | 48 → 32 | `kata-gelu-b10c48-value64x2-v1` |

The deeper model appends six ordinary two-convolution residual blocks. The
wider model increases trunk and global-branch widths (global channels 16 → 24),
but retains 32 output channels in the value convolution, giving 64 pooled MLP
inputs, and 10 hidden policy channels. All existing architecture identities
keep their parameter layout and semantics; the production default remains
the baseline until these experiments provide evidence for a change.

## Training and comparisons

Both candidates train fresh for 20 passes over the same deduplicated replays
60–69 and the same held-out trajectory groups: 15,750 training games and 1,750
validation games. They use all eight symmetries, seed 20260906, batch 256,
standard Adam, LR 0.001, and weight decay 0.0001. The validated device replay
cache is requested. Cache-check failure falls back to uncached standard Adam;
architecture CUDA-check failure stops training. No fused-Adam change is mixed
into the experiment.

Each model is selected by minimum combined validation policy loss + value
MSE over its 20 passes, with ties favoring earlier passes. The baseline reuses
pass 18 (`00000017`, SHA-256
`6c9d758cfba6d1ed86a06d176efeb89f6584fb76425898f3f6a05a941dde2e8f`).
This keeps selection consistent across the candidates and baseline without
training a third copy. Validation loss remains a selection proxy, not proof
of stronger play.

After both training jobs, three matches run:

1. Baseline vs deeper; seed 20260926.
2. Baseline vs wider; seed 20260927.
3. Deeper vs wider; seed 20260928.

Each match uses 1,000 games, 500 per seat, 4,000 simulations per move,
temperature 0.7, concurrency 300, inference batch size 64, and 1 ms batch
timeout. Inference/training preflights measure all three architectures with
the same protocol. Larger models have different compute costs: this compares
equal replay exposure and equal search visits, not equal training or search
wall-clock budgets. Results remain conditional on one training seed.

### Overlapping comparison rounds

At the user's request, the next match starts when its immediate predecessor
has completed **600/1,000 games (60%)**, with a maximum of **two active matches**.
Thus match 2 can overlap match 1; match 3 needs match 2 to reach 60% and a free
slot. A successfully completed predecessor also satisfies the threshold.
Progress is checked every five seconds from native per-game progress logs.
Each match retains its own 1,000-game budget, seeds, seat balance, 300-game
parallelism, 4,000 simulations, and temperature. Both trainers must finish
and their selected checkpoints must validate before any match starts.

`scripts/archive/run_capacity_battles_overlap.py` runs in tmux session
`alz-capacity-battles-overlap`. Its fixed scheduling plan, control-transfer
audit, status, per-match states, and GPU/host telemetry live under
`capacity/overlap-battles/`. The dashboard shows simultaneous match states and
the overlap rule in comparison details. The watcher chains its queue gate
after the width worker's existing gate; running trainers continue unchanged.
Once both trainers have exited, it verifies that the original serial queue
worker is paused and has no child process, retires that idle worker, acquires
the existing queue locks, and runs the comparisons. This avoids duplicate
matches from the old serial schedule.

Only complete validated reports are published. Restarted unfinished matches
start afresh, and progress parsing begins at the new attempt's log offset so
old progress cannot trigger early overlap. A failed match blocks new launches;
already active matches may finish and retain their results. The primary
comparison score still uses all 1,000 games, including long games. Overlap
changes resource contention and timing, so concurrent battle durations are
not isolated throughput benchmarks. The 60% threshold does not guarantee a
low-load tail: up to 300 games may still be active in each match.

The read-only dashboard shows both training jobs and all three matches.
The pod checkout is `/workspace/alpha-zero-capacity-20260907`; output is
`/workspace/alpha-zero-followups/runs/value-heads-20260906/capacity`.
The tmux session is `alz-capacity-20260907`.
After the concurrent-training smoke test showed 1.997× aggregate throughput,
the user requested that width training begin alongside the active depth run.
`scripts/archive/run_capacity_width_parallel.py` now runs the width model in tmux
session `alz-capacity-width48-parallel`, using the same executable, output
directory, replay configuration, and 20-pass budget. Its independent status,
plan, validation selection, and deployment receipts are under
`capacity/parallel-width48/`; the dashboard displays both active trainers.

The parallel worker owns a durable pause of **future parent queue commands**;
the depth training process continues uninterrupted. After width training exits
successfully and all 20 checkpoints validate, the worker transfers control to
the installed comparison scheduler's gate. Both trainers must exit before the
comparison scheduler takes over the queue, as described above.
Failure leaves later commands paused and is reported in the width job. Queue
control changes by another controller are preserved rather than overwritten.
The width preflight was run concurrently, so its timings are correctness
preflight measurements and must not be treated as isolated throughput results.

`scripts/archive/launch_capacity_runpod.sh` builds and tests the isolated checkout,
then invokes `scripts/archive/run_capacity_experiments.py`. Immutable plans record
source metrics, replay/model hashes, code, binary, and training settings.
Completed stages are validated on restart; predecessor locks exclude
competing queue workers. Local deployment records live in
`runs/capacity-20260907/`.

## Cancelled checkpoint-only comparisons

At the user's request, the pass-18-versus-pass-20 match was stopped and the
KataGo-style selected-versus-final match cancelled before starting. Logs and
the completed original-versus-KataGo match are retained.

The stopped match completed 465/1,000 games:

| Model | First-seat W–L | Second-seat W–L | Total W–L |
| --- | --- | --- | --- |
| Original pooling pass 18 | 227–11 (238 games) | 10–217 (227 games) | 237–228 |
| Original pooling pass 20 | 217–10 (227 games) | 11–227 (238 games) | 228–237 |

The raw pass-18 score is 50.97%; equally weighting the two seats gives
49.89%. There is no apparent separation in this partial sample. Unfinished
long games are excluded, so neither number is an unbiased completed-match
strength estimate. No full-result confidence interval or promotion decision
is inferred from these outcomes. The pod retains `cancellation.json`, the
log (SHA-256 `1f14ecb7bd68b29dfedb4496d976e2b992f552b192ed05e0041f335c65c3ed39`),
and the original immutable match plan under `pooling-checkpoints/`.

## Validation detail

Local tests cover new tensor shapes, two global blocks, fixed head widths,
finite gradients, serialization, and optimizer/checkpoint continuation.
A single-empty-board training fixture caused non-finite parameters in the
16-block model before checkpoint saving. The serialization test now uses a
fixed, nondegenerate binary-board batch and explicitly rejects non-finite
parameters before saving; uninterrupted and restored training agree. This
does not establish stability for degenerate training batches. The experiment
uses mixed replay batches of 256, and actual replay optimizer steps plus
CPU/CUDA output parity and finite gradients must pass before full training.

The first CUDA parity attempt stopped the queue before training. A seeded
sweep reproduced a policy-logit error of 0.00239 with CUDA's default TF32
convolutions. Disabling TF32 for the parity process made all 33 seeded
initializations of both architectures pass at the original 0.002 absolute
and relative tolerances (264 head-output checks; maximum absolute error
0.00000954). The wrapper therefore compares CPU and CUDA at
matching FP32 precision; this override does not affect training or battle
processes. Real-replay preflights use the production CUDA settings. The
failed attempt and its original plan are retained under
`capacity/deployment/failed-cuda-attempt-1/`.
