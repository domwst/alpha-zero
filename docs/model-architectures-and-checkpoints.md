# Model architectures, checkpoints, and strength comparisons

The binary has one persistent architecture description, `ModelSpec`, and one
runtime-dispatch enum, `GomokuModel`. A new architecture must be added to both.
Each `ModelSpec` variant is an immutable checkpoint ABI: changing parameter
names, shapes, registration, or forward semantics requires a new versioned
variant rather than changing an existing one.

The default for **new training runs and benchmarks** is
`kata-gelu-boardmask-value64x2-v1`: a constant board-mask input, exact GELU,
a 64 → 64 → 64 → 1 value MLP,
the original mean/max global pooling, and the existing 10-block/32-channel
trunk. See the [consolidated experiment findings](../reports/architecture-experiments-2026-09-07.md)
for the earlier decisions and their uncertainty, and the
[board-mask experiment](board-mask-experiment.md) for the September 10 addition.
ReLU, smaller heads, legacy ResNet,
and KataGo-style pooling remain available for explicit experiments and old
checkpoints. `ModelSpec::default()` retains the legacy identity; fresh-run
selection uses the CLI's `ArchitectureChoice::default()` instead.

`kata-gelu-v1` is Kata v1 with exact GELU replacing every hidden ReLU. It has a
separate architecture identity despite sharing the tensor layout. For a
controlled comparison using saved games, see [the activation experiment](activation-experiment.md)
and the `train-replay` command.

`kata-value64-v1` and `kata-value64x2-v1` replace the value MLP with
64 → 64 → 1 and 64 → 64 → 64 → 1, respectively. Their GELU counterparts are
`kata-gelu-value64-v1` and `kata-gelu-value64x2-v1`. See the
[replay follow-up experiments](replay-followup-experiments.md) for the queued
training and strength comparison protocol.

The `kata-pool-*` and `kata-gelu-pool-*` variants use KataGo-style trunk global
pooling blocks with the current, wide or deep value head. See the
[pooling experiment](pooling-experiment.md) for exact topology and scope.

`kata-gelu-b16c32-value64x2-v1` and `kata-gelu-b10c48-value64x2-v1`
test trunk depth and width while keeping the selected heads and two original
global blocks. These are experimental identities; see the
[capacity experiment](capacity-experiment.md).

`kata-gelu-b16c32g3-value64x2-v1` is the 16-block/32-channel experiment
with three original global-pooling blocks at zero-based indices 3, 7 and 11.
See [third-global-experiment.md](third-global-experiment.md).

## Starting and resuming runs

New self-play runs use the selected default when `--architecture` is omitted.
For reproducible commands, it can also be specified explicitly:

```bash
./run.sh cargo run --release -- train \
  --architecture kata-gelu-boardmask-value64x2-v1 \
  --checkpoint-dir runs/example/checkpoints
```

Resume infers the architecture from the latest complete snapshot. Passing
`--architecture` while resuming acts as an assertion and fails if it disagrees
with the checkpoint. A run directory may not mix architectures or tensor
schemas.

`train-replay` also defaults to `kata-gelu-boardmask-value64x2-v1`. Replaying or resuming
an experiment using another architecture requires its explicit
`--architecture`; the saved replay configuration must still match exactly.

Inference, training, and self-play benchmarks also accept `--architecture`, so
architecture throughput and memory measurements can use the same binary and
benchmark protocol.

## Checkpoint format

Version-2 metadata records:

- the versioned `ModelSpec`;
- a SHA-256 digest of the exact model file;
- a SHA-256 fingerprint of sorted tensor names, shapes, and data types;
- game, position, action, value, replay, and optimizer schema identifiers;
- epoch and replay-size information.

Loading verifies the model digest, the stored tensor layout, and the layout of
the model constructed from `ModelSpec` before loading weights. Play and battle
need only `model.safetensors` plus `metadata.json`; training resume additionally
requires optimizer and replay files.

## Comparing playing strength

`battle` runs a concurrent match series, alternates checkpoint identities
between first and second seat, and keeps per-checkpoint temperatures attached
to checkpoint identity across those seat swaps. For example:

```bash
./run.sh cargo run --release -- battle \
  --first-checkpoint-dir runs/a/checkpoints/00000049 \
  --second-checkpoint-dir runs/b/checkpoints/00000049 \
  --games 200 \
  --simulations 2000 \
  --first-temperature 0.0 \
  --second-temperature 0.0 \
  --games-parallelism 32 \
  --inference-batch-size 16 \
  --batch-timeout-us 1000 \
  --seed 20260831 \
  --heartbeat-seconds 60 \
  --output reports/a-vs-b.json
```

Use `--temperature` for a shared value. The report contains exact checkpoint
and model identities, every move, per-game outcomes, W/L/D and score rates,
an approximate 95% Wilson interval, an Elo point estimate when finite,
first-seat/second-seat splits, evaluations per second, games per second, and
network batching statistics. `--no-move-logs` suppresses move-by-move stdout
without removing moves from the JSON report.

For architecture decisions, use the same simulation count and selection
temperature for both competitors. Run enough games for the confidence interval
to become decision-useful; a small series is a smoke test, not evidence of an
improvement.

## Checkpoint format support

All retained checkpoints have been migrated to metadata format v2. The binary
now supports v2 directly and rejects older metadata versions; the completed
one-shot migration command is intentionally no longer part of the application.

### Board-mask ablation (10 September 2026)

`kata_gelu_boardmask_value64x2_v1` / CLI `kata-gelu-boardmask-value64x2-v1`
extends the current GELU, 10-block, 32-channel, original-pooling, deep-value-head
architecture with an implicit all-ones board plane. Its zero-padded 3×3
convolution is added to the existing input convolution before the trunk. This is
algebraically a third input channel, stored as `board_mask_conv.weight` (32×1×3×3)
and evaluated once per batch with broadcasting. Replay encoding and caches stay
two-channel. The additional kernel is registered after all baseline tensors, so
all shared initial tensors are identical under the same seed. Existing model
variants retain their parameter names and forward behavior.
