# Self-play training protocol

New self-play uses normalized MCTS root visit counts as the policy training
target. Move selection applies temperature to a separate copy of that distribution;
diagnostics retain the actual sampling distribution. Optional `--top-p` filters
that temperature-adjusted sampling distribution, then renormalizes it. The
shortest descending-probability prefix reaching the threshold is extended through
all exact ties at its boundary; zero-probability moves stay zero. The default 1.0
preserves the previous sampling behavior. Outcome-dependent loss weighting is
not enabled.

Temperature uses consecutive pairs of moves to avoid giving the two seats
different temperatures within a turn pair:

- Moves 1–6: 1.0.
- Pairs 7–8 through 19–20: decrease linearly in seven steps to 0.7.
- Later moves: 0.7.

This removes a scheduling asymmetry, not Gomoku's intrinsic first-player advantage.

## Replay capacity

Capacity counts stored positions before the eight symmetry augmentations. At
600 new games per cycle and an initial estimate of 25 positions per game, the
default capacity for one-based training cycle `e` is:

```text
37,500 + 2,250 * max(0, e - 15)
```

This corresponds to 2.5 nominal epochs initially, growing by 0.15 nominal epochs
per subsequent cycle. These are fixed position counts, not measurements of game
length: changing games per epoch does not silently rescale them. Override with
`--replay-positions`, `--replay-position-growth` (zero disables growth), and
`--replay-growth-start-epoch`. On resume, growth uses the absolute snapshot epoch;
snapshot IDs are zero-based, so snapshot 15 is the first increased capacity.

Games within each incoming epoch are shuffled before FIFO eviction so completion
order does not select for slow games. Eviction preserves whole trajectories, so
occupancy can be slightly below capacity. One game is retained even if that game
alone exceeds capacity. Explicit `--replay-games N` retains the legacy fixed
game-count limit and cannot be combined with position capacity options.

The trainer still makes one pass over every retained position in all eight
symmetries. Growing the buffer therefore also increases optimizer updates per
self-play cycle. There is no separate optimizer-step budget. With `--replay-lr-exponent X`, the
learning rate becomes `initial_lr * (initial_capacity / current_capacity)^X`.
This requires an explicit `--learning-rate` and position-based capacity. The
rate is derived from the absolute epoch on resume, not compounded from the last
saved rate. Omitting the exponent preserves the existing constant/restored LR.

For the September 9 fresh 100-epoch run: 1,000 games/epoch, initial capacity
62,500 positions, growth 3,750 positions/epoch after epoch 15, LR 0.001 and
exponent 1.1. Epoch 100 has capacity 381,250 and LR about 0.000137. CPU replay
caching plus two prefetched batches avoids full GPU replay caches competing
with concurrent workers. `--bn-gamma-one` sets only fresh BatchNorm scales to
one; resumed model parameters are preserved.

## Existing data and reproducibility

Existing snapshots remain loadable and their stored policy targets are preserved.
Old, temperature-adjusted targets in a resumed replay buffer are gradually evicted;
they are not reconstructed or silently relabeled. A clean comparison of collection
protocols requires newly collected replay data.

Invocation and epoch statistics use schema version 7 and record the policy target
and temperature protocol for **new games**. Epoch statistics also record the actual
position count and effective capacity (`null` for legacy game-count mode). Preserve
these statistics alongside checkpoints when archiving a run.

Epoch statistics also record the scheduled LR and sampling totals: positions,
positive support before/after filtering, deterministic positions, removed
probability mass, and entropy. Divide the latter two sums by positions to obtain
per-position means. Training stops on non-finite losses before applying the
corresponding optimizer update.

## Ordered replay-history reconstruction

Fresh training now defaults to `kata-gelu-boardmask-value64x2-v1`. Existing
checkpoints keep their recorded architecture; the default does not migrate them.

`train --replay-history-dir OLD_RUN --replay-history-epochs 20` uses
`OLD_RUN/checkpoints/00000000` through `00000019` for the first twenty optimizer
passes. Each buffer is loaded exactly in saved order, including its policy/value
targets and retained trajectory set. The trainer does not pool or deduplicate
buffers, re-split data, regenerate policies, reshuffle trajectories, or reset Adam
between passes. Batch augmentation/shuffling, per-epoch seeds, BatchNorm behavior,
and the replay-linked LR use the same code as native self-play training.

Source settings, per-epoch sample counts and LR are verified, and a manifest pins
source replay, metadata and stats hashes. Each reconstructed epoch saves a new
model, optimizer and the matching replay buffer under its original epoch number.
Resuming loads this full snapshot and skips completed reconstruction epochs.
Once the requested history ends, the same training loop generates fresh games
with the new model (epoch 21 onward in this run), retaining Adam and replay state.

Reconstructed stats carry `history_source` and source checksums. Game counts and
outcomes describe the original model's games; `self_play_seconds` is null because
no new games were generated. The dashboard labels these epochs explicitly.
`scripts/import_self_play_durations.py --run-dir NEW_RUN` copies the original
game-generation timings into `inherited-self-play.json`, checking their source
hashes against the reconstruction manifest. The dashboard uses these for the
reconstructed epochs' self-play duration plot and table, with an inherited-timing
note. Actual reconstruction statistics, training timings, and job elapsed time
remain unchanged; newly generated epochs use their own recorded durations.
This recreates training on the old trajectories, not the trajectories the new
architecture would actually have generated itself.

The September 10 board-mask run uses seed 20260909, BN gamma=1, standard Adam,
CPU replay caching/prefetch 2, B256, and the original twenty self-play buffers.
After reconstruction, `scripts/run_board_mask_selfplay.py` waits for the existing
comparison queue to complete, then continues toward 100 total epochs at
S3000/P500/B256, top-p 0.95 and the original temperature/replay/LR schedules.
It has a separate output directory `runs/selfplay-nucleus/boardmask-p095` and
pinned executable. Old self-play checkpoints are retained and remain paused.

CPU and CUDA integration tests reconstruct a small original run with the same
architecture, compare all saved tensors, verify interrupted reconstruction, and
compare the next self-play epoch after restoring the reconstructed state.
They exercise the board-mask model with CPU replay caching and prefetching.
Run the GPU check with `bash scripts/check_replay_history_runpod.sh`: it enables
deterministic cuDNN/cuBLAS and disables TF32 only in the test process. Ordinary
CUDA reductions produced different weights on repeat runs; strict tensor parity
passed with deterministic algorithms. Production retains its existing CUDA settings.
