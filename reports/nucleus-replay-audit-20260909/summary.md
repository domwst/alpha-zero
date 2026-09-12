# Nucleus sampling replay audit

Stored, already temperature-adjusted policies; whole games deduplicated across checkpoints.

| Top-p | Mean mass removed | Mean supported moves before → after | Recorded actions excluded | Games with an excluded recorded action |
|---|---:|---:|---:|---:|
| 0.9 | 2.82% | 173.8 → 28.2 | 10,915/373,712 (2.92%) | 8,630/17,500 (49.31%) |
| 0.95 | 1.00% | 173.8 → 61.1 | 3,925/373,712 (1.05%) | 3,414/17,500 (19.51%) |
| 0.98 | 0.41% | 173.8 → 69.1 | 1,578/373,712 (0.42%) | 1,488/17,500 (8.50%) |
| 0.99 | 0.22% | 173.8 → 72.8 | 854/373,712 (0.23%) | 827/17,500 (4.73%) |
| 1.0 | 0.00% | 173.8 → 173.8 | 0/373,712 (0.00%) | 0/17,500 (0.00%) |

## Top-p 0.95 by move stage

| Moves | Positions | Mean mass removed | Recorded actions excluded | Median retained moves |
|---|---:|---:|---:|---:|
| 01-06 | 105,000 | 0.52% | 0.56% | 28.5 |
| 07-12 | 85,315 | 1.53% | 1.69% | 2 |
| 13-20 | 71,347 | 1.00% | 1.01% | 1 |
| 21+ | 129,550 | 1.03% | 1.09% | 1 |

## Top-p 0.95 by seat

| Seat | Recorded actions excluded | Exclusion rate |
|---|---:|---:|
| first | 1,595/187,630 | 0.85% |
| second | 2,330/186,082 | 1.25% |

## Interpretation

- No counterfactual outcomes or strength estimates; positions follow original trajectories.
- Actual-action counts exclude each game's final action.
- Archived policies already include sampling temperature; no new temperature applied.
- Positive support counts exclude legal actions that had zero visits.
- Examples are the first 12 excluded actions in input order, not a quality-ranked sample.

Excluded actions are not established blunders. This measures which historical choices the filter would forbid, not which alternative would be sampled or how the game would end.

The JSON report includes seat/outcome breakdowns, entropy and support quantiles, and 12 inspectable examples (row-major action IDs; stones 1=current actor, 2=opponent).
