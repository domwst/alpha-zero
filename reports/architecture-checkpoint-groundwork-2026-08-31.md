# Architecture and checkpoint groundwork — 2026-08-31

## Decision

Use a serialized, versioned `ModelSpec` as the source of truth for architecture
identity and a concrete `GomokuModel` enum for runtime dispatch. This lets one
binary train, benchmark, play, and directly match different architectures while
keeping the existing generic inference engine and avoiding trait-object changes
throughout MCTS.

The existing network is named `legacy_resnet_v1`. The name is intentionally
permanent: the planned global-pooling architecture will receive a separate
variant and cannot silently reinterpret an old checkpoint.

## Changes and rationale

- Checkpoint metadata is versioned to v2 and now records model specification,
  exact model SHA-256, tensor-layout SHA-256, and semantic data schemas. This
  prevents a same-shaped but semantically incompatible model from loading and
  gives match reports a stable identity even after checkpoints are copied.
- Loading constructs the network from metadata before touching weights, then
  checks the constructed layout, stored layout, and file digest. Run-directory
  discovery rejects mixed model schemas instead of selecting one by accident.
- Model-only evaluation is separated from full training restoration. A battle
  does not fail merely because optimizer or replay artifacts were intentionally
  omitted from an evaluation copy.
- Training, play, and every benchmark use the same model factory. Existing runs
  infer their architecture; an explicit CLI architecture is treated as an
  assertion. New benchmark records and epoch metrics include the model spec.
- `battle` is now a deterministic, concurrent series instead of one game. It
  alternates seats, supports shared or per-checkpoint temperatures, retains full
  move records, and emits aggregate W/L/D, score, Wilson interval, Elo estimate,
  seat splits, and inference/batching throughput. Per-game summaries and
  evaluation-rate heartbeats make long series observable while they run.
- A temporary v1-to-v2 command validated and migrated the retained checkpoint
  copies before being removed from the application.

## Interpretation limits

The Elo value is a point estimate derived from match score and is omitted for a
complete sweep because the unregularized estimate is infinite. The Wilson
interval treats the draw-adjusted score rate as a binomial proportion; it is a
useful compact uncertainty indicator, not a full paired-color or sequential
rating model. Architecture promotion should therefore use many games and also
inspect the first-seat/second-seat split.

## Migration status

On 2026-09-01, all 119 retained checkpoints (118 from the stopped training run
and one pilot checkpoint) were migrated to metadata format v2 and revalidated.
No version-1 metadata remains in the local archive. The one-shot migration code
was then removed, leaving strict v2 loading as the only supported path.
