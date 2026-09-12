# Gomoku architecture decision — 2026-09-07

The working architecture for new training is **`kata-gelu-value64x2-v1`**:
exact GELU, a two-hidden-layer value MLP, and the original global-pooling
blocks. The trunk remains 10 blocks with 32 channels. Existing architecture
identities retain their original meaning and all experimental checkpoints
remain loadable.

This is a practical selection from a single-seed experiment series. The
evidence for retaining original pooling is stronger than the evidence for
GELU or for the second value-head hidden layer.

## Decisions

| Component | Adopted configuration | Evidence and interpretation |
| --- | --- | --- |
| Activation | Exact GELU (`gelu("none")`) | 52.0% against ReLU; retain the selected candidate, with no established repeatable advantage. |
| Value MLP | 64 pooled inputs → 64 → 64 → 1, scalar tanh output | Won the equally weighted value-head round robin. Its 51.1% against the wide head is inconclusive; the additional layer is a pragmatic choice, not a proven improvement. |
| Global pooling | Original mean/max global blocks | Won at final checkpoints and again at validation-selected checkpoints. Do not adopt the tested KataGo-style block. |
| Trunk | 10 residual blocks, 32 channels; global blocks at indices 3 and 7 | Depth and width changes were not tested. |
| Policy head | Existing 32 → 10 → 1 convolutions, with GELU | No independent policy-head experiment was conducted. |
| Normalization | Existing BatchNorm | No normalization change was tested; diagnostics remain separate work. |

The value head keeps its existing 1×1 convolution and mean/max pooling to
produce the 64 MLP inputs. No parameter names, tensor shapes, registration
order, or forward semantics change in this consolidation: the adopted
architecture already exists. It becomes the CLI default for fresh self-play
training, replay training, and benchmarks.
Resuming self-play training continues to infer the stored architecture;
an explicit conflicting architecture is rejected.

## Completed comparisons

Scores below are for the first named network. No games were draws. Intervals
are the dashboard's original pooled 95% Wilson intervals, retained for
consistency with the machine reports.

| Comparison | Wins–losses | Score | 95% interval |
| --- | ---: | ---: | ---: |
| GELU vs ReLU, current head | 156–144 | 52.0% | 46.4–57.6% |
| Replay ReLU vs self-play checkpoint 69 | 169–131 | 56.3% | 50.7–61.8% |
| Wide value head vs current head | 320–280 | 53.3% | 49.3–57.3% |
| Deep value head vs current head | 313–287 | 52.2% | 48.2–56.1% |
| Deep vs wide value head | 511–489 | 51.1% | 48.0–54.2% |
| Original vs KataGo-style pooling, final checkpoints | 532–468 | 53.2% | 50.1–56.3% |
| Original pass 18 vs KataGo-style pass 15 | 570–430 | 57.0% | 53.9–60.0% |

In the selected-checkpoint pooling match, original pooling won 461/500 games
as first player and 109/500 as second. KataGo-style pooling won 391/500 and
39/500 respectively. The difference is 14 percentage points in either seat.
With balanced seats and no draws, these equal absolute margins follow
arithmetically; they are not independent confirmations of an improvement.

The two pooling comparisons involve checkpoints from the same two training
runs. They test sensitivity to checkpoint choice, not independent training
replication. Their results are intentionally not pooled into a single
architecture confidence interval. Match intervals also exclude uncertainty
from training seeds and architecture selection across several comparisons.

## Checkpoint-selection sensitivity

All initial matches used the final checkpoint after 20 training passes.
The follow-up selected each pooling model by the same rule: minimum validation
policy loss + value MSE among its 20 checkpoints, with ties favoring the
earlier pass. Pass numbers are one-based; checkpoint directory numbers are
zero-based.

| Pooling model | Selected pass / checkpoint | Selected combined loss | Final combined loss |
| --- | --- | ---: | ---: |
| Original, deep value head | 18 / `00000017` | 2.085865 | 2.161591 |
| KataGo-style, deep value head | 15 / `00000014` | 2.085628 | 2.136121 |

KataGo-style pooling lost despite slightly lower selected combined validation
loss. At the final checkpoints it also had lower validation value and policy
losses than its opponent. Validation loss is useful for choosing candidates,
but it does not reliably rank these models' playing strength with MCTS.
This experiment does not establish minimum-loss selection as the new training
or promotion policy.

At the result snapshot (2026-09-07 15:33 UTC), **original pooling pass 18 vs
pass 20 was running**, and **KataGo-style pooling pass 15 vs pass 20 was
queued**. These remain authorized and can refine checkpoint selection without
blocking the architecture decision. No checkpoint is promoted here as the
strongest within its architecture.

## Protocol and scope

- Replays: deduplicated checkpoints 60–69, 17,500 unique games; 15,750 training
  and 1,750 validation games, split by trajectory group.
- Training: 352,113 base training positions, all eight symmetries, 20 passes,
  batch 256, standard Adam, constant LR 0.001, weight decay 0.0001, seed
  20260906. Validation has 39,099 base positions.
- Dataset SHA-256:
  `2a391f1f93a588877e06dc550d2eaa8f953d1c198f8d6988053594a32a46f5bc`.
- Matches: 4,000 simulations per move, temperature 0.7, equal games in each
  seat, inference batch size 64. The first two comparisons used 300 games;
  current-vs-larger-head matches used 600; subsequent comparisons used 1,000.
  The later comparisons use concurrency 300.
- The KataGo-style arm used the validated device replay cache while retaining
  standard Adam and the same samples and ordering. No fused-optimizer change
  was introduced into that comparison.
- The tested pooling replacement changes branch topology and uses 16 local
  and 16 global intermediate channels, with our BatchNorm and initialization.
  On fixed 19×19 boards its extra scaled-mean statistic is redundant. The
  result concerns this implementation, not all KataGo pooling architectures.
- Different head/pooling shapes consume initialization randomness differently.
  One common numerical seed does not pair every shared tensor across variants.

The replay-trained ReLU result shows that this protocol can extract a stronger
checkpoint from existing data. It does not isolate learning rate, replay
lookback, initialization, or training exposure as the cause, and it does not
replace testing continued self-play. LR scheduling, replay-window tuning,
inference-mode training-loss measurements, BatchNorm diagnostics, and repeated
training seeds remain distinct future experiments.

Exact checkpoint identities, scores, match settings, and validation summaries
are in [the machine-readable results](architecture-experiments-2026-09-07.json).
Experiment protocols are documented in [activation](../docs/activation-experiment.md),
[value heads](../docs/replay-followup-experiments.md), and
[pooling](../docs/pooling-experiment.md).

## Implementation validation

The consolidation passed 19 CLI/command tests and 55 library tests, covering
fresh-run defaults, explicit overrides, inference of every retained
architecture on resume, mismatch rejection, model serialization, output
contracts, finite gradients, and model/optimizer/replay checkpoint round trips.
Two CUDA-only library tests were skipped locally. Existing forward
implementations and the running pod executable were not changed.
