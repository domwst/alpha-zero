# Model architectures, checkpoints, and strength comparisons

The binary has one persistent architecture description, `ModelSpec`, and one
runtime-dispatch enum, `GomokuModel`. A new architecture must be added to both.
Each `ModelSpec` variant is an immutable checkpoint ABI: changing parameter
names, shapes, registration, or forward semantics requires a new versioned
variant rather than changing an existing one.

## Starting and resuming runs

New runs select an architecture explicitly when more than one is available:

```bash
./run.sh cargo run --release -- train \
  --architecture legacy-resnet-v1 \
  --checkpoint-dir runs/example/checkpoints
```

Resume infers the architecture from the latest complete snapshot. Passing
`--architecture` while resuming acts as an assertion and fails if it disagrees
with the checkpoint. A run directory may not mix architectures or tensor
schemas.

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
