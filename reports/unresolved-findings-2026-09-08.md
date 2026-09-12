# Unresolved network and training findings — 2026-09-08

Audit of the experiment dashboard, existing reports, current Rust implementation,
and the exact pinned tch revision `7e7d436ac73523afedf1a192892dfc9346568ba6`.
This is a review; it does not change architectures, checkpoints or the experiment queue.
The result snapshot and selected/final training metrics are retained in
[unresolved-findings-2026-09-08.json](unresolved-findings-2026-09-08.json).

## What the capacity experiments establish

Relative to the selected current 10-block/32-channel checkpoint:

| Candidate | Candidate wins / games | Interpretation |
| --- | ---: | --- |
| 16 blocks, two global blocks | 437 / 1,000 | Lost this comparison. |
| 10 blocks, 48 channels | 479 / 1,000 | No demonstrated advantage; result inconclusive. |
| 16 blocks, three global blocks | 500 / 1,000 | Tied exactly; pooled 95% Wilson interval about 46.9–53.1%. |

The third-global model was selected at pass 9. At this audit the comparison
against the previous 16-block/two-global model was still running at 999/1,000.
The original pooling implementation should remain the baseline. These are
single-training-seed results on one frozen replay pool and one optimization
recipe, not evidence that additional capacity cannot help with other data or training.

## Unresolved findings from the previous review

### 1. BatchNorm affine initialization

Every Kata BatchNorm call currently uses `Default::default()`. The pinned
binding's `src/nn/batch_norm.rs` initializes gamma with `Uniform(0, 1)`, beta
with zero, running mean with zero, and running variance with one. Epsilon is
1e-5 and momentum is 0.1. The unusual part is gamma initialization;
PyTorch's documented gamma initialization is one. Epsilon and momentum match
its usual defaults. Gamma is learned, so its initial distribution does not
imply that trained gamma remains in [0, 1].

The random initial scales attenuate channels unequally and alter residual-branch
scales. Their effect on learning and depth is a hypothesis; we have not measured
it. Test explicit gamma=1 on the 10-block baseline before adding more capacity.
Use paired initial values for all other tensors: merely keeping the numerical
seed is insufficient because constant initialization consumes different random
numbers from uniform initialization. Preserve existing architecture identities.
Do not overwrite gamma in trained checkpoints as an alleged correction.

Sources: `src/gomoku/kata_nn.rs`, pinned tch `src/nn/batch_norm.rs`, and
[PyTorch BatchNorm2d](https://docs.pytorch.org/docs/2.14/generated/torch.nn.BatchNorm2d.html).

### 2. BatchNorm running statistics and comparable loss measurements

`train_epoch` averages losses measured before each update, across changing
weights, using training-mode BatchNorm. `validate` uses the end-of-pass model
with inference-mode BatchNorm. Their difference combines dataset generalization,
within-pass model changes and normalization mode. The plotted gap is therefore
not a clean estimate of overfitting.

First evaluate fixed training and validation subsets using the same frozen
checkpoint in inference mode. On cloned models, also compare batch statistics
against stored running statistics, and measure prediction changes and per-layer
activation/gradient scales. Inspect gamma/beta and running-variance distributions.
If indicated, recalibrate running statistics using representative shuffled
**training data only**, with no optimizer updates, then reevaluate validation.
Do not use held-out validation data to estimate normalization statistics.
Keep raw snapshots intact. Normalization changes and alternative BN momentum
values should follow measured evidence; 0.1 itself is not a bug.

### 3. Learning-rate schedule and optimization budget

The replay runs use constant Adam LR 0.001 for 20 passes. There is no scheduler
in `train-replay` or `train`; self-play only supports an explicit LR override.
Selected passes differ markedly across variants (8, 9, 11, 15, 16, 18), and
validation often deteriorates later. For the three-global model, validation
value MSE is 0.42775 at selected pass 9 versus 0.51182 at pass 20; combined loss
rises from 2.08084 to 2.22100. This supports investigating the training recipe,
but does not distinguish overfitting from running-statistics/optimization effects.

Compare constant LR with a specified decay schedule at equal data exposure and
update count, using the same architecture and checkpoint-selection rule. A lower
starting LR and warmup are separate knobs. More passes at unchanged LR are not
a substitute for this experiment. Weight averaging is an additional untested
candidate if checkpoint variability remains after diagnosing BatchNorm.

### 4. Replay horizon and training exposure in continued self-play

The replay-trained ReLU beat checkpoint 69 by 169–131. This is evidence for that
checkpoint comparison, not proof that increasing replay lookback caused the gain.
The fresh initialization, pooled data, repeated exposure and optimizer trajectory
all differ from continued self-play.

Pooling checkpoints 60–69 was implemented; tuning the self-play replay horizon
was not. `train` still makes one training pass over its retained buffer after
each self-play epoch. A larger buffer therefore also increases gradient updates
per epoch. Separate replay-window size, updates per new game, data age and learning
rate when testing this. The CLI defaults (1,800 retained games / 600 new games)
are not a description of every historical invocation; historical runs used
explicit overrides. Evaluation must eventually include continued self-play with
fresh search targets, not only fitting the same 17,500 saved trajectories.

### 5. Checkpoint selection, promotion and replication

Minimum combined validation loss now selects checkpoints for the pooling
follow-up, capacity and third-global experiments. That implementation is complete;
its ability to select the strongest player remains unproven. The selected
KataGo-style model had slightly lower combined validation loss yet lost 430–570.
The selected-versus-final intra-architecture comparisons were deliberately
cancelled. The partial original-pooling pass-18/pass-20 match is not a full test.

Earlier results also showed non-monotonic checkpoint strength. A fixed champion
plus a small opponent panel remains a useful promotion test. Self-play currently
continues with the latest training checkpoint without an arena promotion gate.
Repeated training seeds are still missing: more battle games reduce uncertainty
for a fixed pair but do not establish that an architectural change trains better
reliably. Use at least a small replication set before treating modest gains as
settled. Equal seat counts already address the large first-player advantage;
equal absolute margins in both seats are not independent corroboration.

GELU's 52.0% versus ReLU and the deep value head's 51.1% versus the wide head
remain weak evidence despite their adoption as practical defaults. Neither
larger-head versus original-head result individually excluded 50% at 95% confidence.

## Additional initialization and objective items confirmed by this audit

These are candidates for measurement, not newly proven causes of weak play:

- Convolutions and linear layers inherit tch's ReLU-gain Kaiming initialization,
  including the final scalar layer before tanh. Initial value saturation and
  residual-branch scale versus skip-path scale deserve measurement, especially
  across depths. GELU does not have a universal drop-in gain supplied by this
  API. Test smaller output-layer initialization or residual-branch initialization
  separately if diagnostics justify them. See
  [PyTorch initialization reference](https://docs.pytorch.org/docs/2.14/nn.init.html).
- All trainable parameters use the same coupled Adam weight decay, including
  BatchNorm affine parameters and biases. No norm/bias exemption or AdamW
  comparison was tested. This is a regularization choice, not a proven bug.
  Changing it should be a separate experiment; see `src/commands/common.rs` and
  [PyTorch Adam](https://docs.pytorch.org/docs/2.14/generated/torch.optim.Adam.html).
- Loss is an unweighted sum of value MSE and policy cross-entropy. The policy
  head retains a 10-channel bottleneck even in the wider trunk. Neither relative
  gradient influence, value calibration by game phase, policy-head width, nor
  alternative/auxiliary targets was independently tested. Lower priority than
  the normalization and schedule issues above.
- Softplus remains untested. There is currently no result suggesting activation
  swaps should take precedence over the diagnostic work.

## Performance work: completed versus outstanding

Completed: pre-encoding/cache and CPU prefetch implementations; CUDA device-cache
correctness; cached batches in later experiments; two simultaneous training runs
with about 1.997× aggregate steady-state throughput in the short smoke test;
overlapping comparisons at 60%; larger battle budgets and concurrency. Earlier
FIFO retention biased by completion order was corrected by shuffling each new
epoch before truncating the replay buffer.

Still open in the saved performance audit: full fused-Adam CUDA validation and
an isolated, representative throughput comparison. The implementation exists,
but current architecture experiments deliberately use standard Adam. Earlier
value-head runs used the authorized old-trainer fallback, so they did not
measure the new speedups. Cached batching alone does not remove per-tensor Adam
launch overhead. CUDA graphs/further operator fusion, AMP and larger training
batches remain unmeasured possibilities. Scalar loss readbacks were not the
primary bottleneck in the captured trace.

No speedup or playing-strength benefit should be attributed to an unmeasured
backend. Cache construction and validation/checkpoint writing must be included
when estimating whole-run time.

## Recommended order

1. Diagnose fixed-checkpoint train/eval losses, BatchNorm statistics, value
   saturation and per-layer scales using existing checkpoints and training data.
2. Run a controlled baseline versus explicit gamma=1 comparison, preserving
   initial values of every other tensor and the batch order; replicate if useful.
3. Test a learning-rate decay schedule with architecture and normalization fixed.
4. Tune self-play replay horizon and update budget independently, then verify
   improvements with fresh self-play and an opponent panel.
5. Revisit depth only after these measurements provide a reason to expect it to help.

Relevant prior audits: [architecture](architecture-experiments-2026-09-07.md),
[training performance](../docs/training-performance.md),
[profiling](../docs/training-profile-20260906.md),
[concurrent training](concurrent-training-smoke-2026-09-07.md), and
[checkpoint strength](checkpoint-strength-2026-09-02.md).
