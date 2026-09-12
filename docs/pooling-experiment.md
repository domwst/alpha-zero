# Global pooling follow-up

## Validation-selected checkpoint comparisons (2026-09-07)

After the original final-checkpoint comparison completed (current pooling
532 wins, KataGo-style pooling 468), a separate queue compares checkpoints
selected by minimum **validation policy loss + value MSE** over the same
20-pass training budget. Exact ties prefer the earlier pass. This proxy does
not establish playing strength; the following matches test that relationship.

| Order | First checkpoint | Second checkpoint | Match seed |
| --- | --- | --- | --- |
| 1 | Current pooling, pass 18 (`00000017`) | KataGo pooling, pass 15 (`00000014`) | 20260912 |
| 2 | Current pooling, pass 18 (`00000017`) | Current pooling, final pass 20 (`00000019`) | 20260913 |
| 3 | KataGo pooling, pass 15 (`00000014`) | KataGo pooling, final pass 20 (`00000019`) | 20260914 |

Each match has 1,000 games, 500 per seat, 4,000 simulations per move,
temperature 0.7, concurrency 300, and inference batch size 64. All models
use GELU and the `64 → 64 → 64 → 1` value head. No additional training is
performed. Selected combined validation losses are 2.0858653859787033 for
current pooling and 2.0856278091070735 for KataGo-style pooling.

`scripts/archive/run_checkpoint_comparisons.py` pins source metrics, training settings,
model checksums, match seeds, code, and the already validated pooling binary
in `plan.json`. It rejects changed inputs on restart, reuses validated completed
matches, and holds the earlier queue locks to exclude concurrent GPU work.
`scripts/archive/launch_checkpoint_comparisons_runpod.sh` runs it in the existing
pooling checkout and records GPU/host telemetry and an exit receipt.

The pod output directory is
`/workspace/alpha-zero-followups/runs/value-heads-20260906/pooling-checkpoints`.
The tmux session is `alz-pooling-checkpoints-20260907`. The main comparison
started at 2026-09-07 12:34 UTC. All three jobs appear in the existing read-only
dashboard, with labels distinguishing selected and final checkpoints.
Local deployment records are under `runs/pooling-checkpoints-20260907/`.

## Original final-checkpoint experiment

This job runs after the three value-head round-robin matches. It ranks the
current, wide and deep heads by mean score rate against their two opponents,
giving each opponent equal weight. Draws count half. The current-head pairings
retain 600 games each; the upcoming wide/deep pairing uses 1,000 games. Extra
games therefore improve precision without giving that opponent more weight.
Exact ties prefer current, then wide,
then deep. The chosen activation is inherited. Selection records all scores,
checkpoint identities and report digests; it does not establish significance.

The current-pooling baseline reuses the selected final checkpoint from the
value-head round robin. Only the KataGo-style pooling model trains afresh.
The runner verifies the baseline's model hash, training configuration, dataset
and completed budget before reuse. Both use checkpoints 60–69, deduplication, holdout, all eight
symmetries, seed 20260906, 20 passes, batch 256, learning rate 0.001 and weight
decay 0.0001. Adam must match the reused checkpoint (currently standard Adam).
The new model requests a device replay cache, which preserves samples, targets
and ordering and is independently checked on CUDA. Fused Adam is not enabled
only for the new arm: its rounding differences would add an optimizer change
to this architecture comparison. Reuse saves one roughly two-hour training run.
Different tensor shapes consume different initialization randomness; common
layers are not explicitly paired. This remains a single-seed experiment.

The final checkpoints play **1,000 games, 4,000 simulations per move,
temperature 0.7**, 500 games per seat, seed 20260911, up to 300 concurrent games,
inference batch size 64. The runner inherits these budgets from the predecessor
(small CPU smoke fixtures therefore use their own reduced budgets). The
parent's audited `match-plan.json` supplies future overrides while historical
comparisons retain their original budgets. Head selection validates all three
round-robin reports against their individual pinned game counts. Score-rate
ranking uses exact fractions to keep ties deterministic.

## Architecture change

The reference is the ordinary `ResBlock`, `KataConvAndGPool` and `KataGPool` in
[KataGo v1.18.2](https://github.com/lightvector/KataGo/blob/v1.18.2/python/katago/train/model_pytorch.py).
Only trunk global blocks 3 and 7 change. The regular residual blocks, trunk
width, policy head and chosen value head retain their existing definitions.

The new block applies BN/activation, parallel 3×3 local and global convolutions,
global BN/activation, then pools mean, mean × (sqrt(board area) − 14)/10, and
maximum. A bias-free linear layer projects these statistics into local-channel
biases. After addition to the local path, BN/activation and a second 3×3
convolution produce the residual. The intermediate width of 32 is split into
16 local and 16 global channels, as in KataGo's width partition. Each new block
has 14,720 trainable parameters versus 14,976 in the current block.

This adopts KataGo's pooling statistics and branch topology, using our ordinary
BatchNorm and existing initialization conventions. It is not a full port of
KataGo's normalization, initialization, auxiliary heads or training system.
All tensor cells represent actual board locations, so the upstream on-board
mask is identically one; occupied cells still participate in pooling. On 19×19,
the scaled mean equals 0.5 × mean and supplies no independent statistic. The
experiment tests the complete block change, not the third statistic alone.
Maximum pooling follows the upstream single-argmax gradient when values tie.

New checkpoint identities insert `-pool` after `kata` or `kata-gelu`, including
all three head sizes, e.g. `kata-gelu-pool-value64x2-v1`. Existing architecture
identities and checkpoint semantics are unchanged.

## Queue and recovery

`scripts/archive/run_pooling_followup.py` uses a separate executable, repository and
output directory. It waits for predecessor `status.json` to say `complete`,
its launcher to exit successfully, and its writer lock to become available.
It then holds that lock to prevent accidental overlapping restarts, validates
all three reports and training budgets, and selects the head. Failed
predecessors stop the new queue without starting GPU work.

CUDA architecture parity/gradient checks run before training and must pass.
The test launcher compiles with Cargo, then executes the test binaries through
`run.sh` so the CUDA libraries are loaded in the same way as the trainer.
An invalid new architecture cannot be handled by the old binary, which does
not implement it. Cache checks have a five-minute timeout; failure selects
standard Adam and uncached batching, preserving the baseline's optimizer.
Backend selection is persisted before training starts and cannot change on
resume. If a future baseline used fused Adam, its optimizer must pass the full
backend checks; failure stops instead of silently changing the optimizer.
Inference and training preflights run for the new architecture.

The deployed output is
`/workspace/alpha-zero-followups/runs/value-heads-20260906/pooling`.
The parent `pooling-followup.json` registers two extra read-only dashboard
rows under the reuse policy: train KataGo pooling and compare it with the
existing winner. Older paired-training queues retain their three rows. Progress,
curves, results and known logs remain available through GET/HEAD only.

The worker pins its executable and supporting code in `queue-config.json`.
`selection.json`, `reused-baseline.json`, `backend.json`, `experiment.json` and `summary.json` record
decisions and results. Rerunning the exact command resumes completed epochs
and reuses validated complete matches. Interrupted matches restart in full.
The pod remains rented when the queue finishes.

```bash
cd /workspace/alpha-zero-pooling-20260906
source scripts/runpod_env.sh
python3 scripts/archive/run_pooling_followup.py \
  --predecessor /workspace/alpha-zero-followups/runs/value-heads-20260906 \
  --output-dir /workspace/alpha-zero-followups/runs/value-heads-20260906/pooling \
  --validation-plan pooling-validation-plan.json
```

Use tmux for the deployed worker. The launcher records exit status and retains
GPU/host telemetry. It runs without the dashboard, laptop or Codex session.
