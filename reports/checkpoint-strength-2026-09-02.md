# Checkpoint strength tournament — 2026-09-02

## Summary

Kata-18 and Kata-19 were the top two checkpoints in the four-model Kata
round-robin. Kata-18 won the direct match against Kata-19 by 114–86, while
Kata-19 had the better aggregate result against the seven legacy checkpoints:
661–739 (47.2%) versus Kata-18's 596–804 (42.6%). On this matched legacy
schedule, epoch 19 improved over epoch 18 by 65 wins, or 4.6 percentage points.

The mature legacy family still led overall. Legacy-80 had the strongest
combined result against the two Kata finalists at 246–154 (61.5%), followed by
Legacy-90 at 233–167 (58.2%) and Legacy-129 at 226–174 (56.5%). Checkpoint
number did not monotonically predict legacy strength, so individual pairwise
scores are more useful than interpolating between epochs.

All 20 series completed successfully: 4,000 games, 0 draws, and 133 persisted
artifact files totaling 67.3 MB. Every series used 200 games, temperature 0.7,
2,000 simulations per move, alternating seats, inference B128/P200, and a 1 ms
batch timeout. Three series ran concurrently; observed memory peaked around
59.1 GiB of the pod's 116.4 GiB limit.

## Kata selection round

The aggregate round-robin ranking was:

| Rank | Checkpoint | W-D-L | Score rate | 95% CI |
| ---: | --- | ---: | ---: | ---: |
| 1 | Kata-18 | 359-0-241 | 59.8% | 55.9–63.7% |
| 2 | Kata-19 | 330-0-270 | 55.0% | 51.0–58.9% |
| 3 | Kata-16 | 313-0-287 | 52.2% | 48.2–56.1% |
| 4 | Kata-12 | 198-0-402 | 33.0% | 29.4–36.9% |

Pairwise scores are from the first checkpoint's perspective:

| First | Second | W-D-L | Score | 95% CI | Elo difference | Mean / median / max plies |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Kata-12 | Kata-16 | 70-0-130 | 35.0% | 28.7–41.8% | -107.5 | 35.2 / 28.5 / 116 |
| Kata-12 | Kata-18 | 65-0-135 | 32.5% | 26.4–39.3% | -127.0 | 32.7 / 27.0 / 152 |
| Kata-12 | Kata-19 | 63-0-137 | 31.5% | 25.5–38.2% | -135.0 | 36.8 / 30.0 / 121 |
| Kata-16 | Kata-18 | 90-0-110 | 45.0% | 38.3–51.9% | -34.9 | 34.4 / 29.0 / 165 |
| Kata-16 | Kata-19 | 93-0-107 | 46.5% | 39.7–53.4% | -24.4 | 36.7 / 29.5 / 146 |
| Kata-18 | Kata-19 | 114-0-86 | 57.0% | 50.1–63.7% | +49.0 | 31.6 / 27.0 / 102 |

Kata-18 and Kata-19 therefore advanced to the legacy comparison. Their
ordering should not be overinterpreted: Kata-18 won their single direct series,
but Kata-19 performed better on the broader, matched legacy schedule.

## Kata finalists versus legacy

Scores and confidence intervals are from the Kata checkpoint's perspective:

| Kata | Legacy | W-D-L | Score | 95% CI | Elo difference | Mean / median / max plies |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Kata-18 | Legacy-129 | 79-0-121 | 39.5% | 33.0–46.4% | -74.1 | 35.1 / 27.5 / 152 |
| Kata-18 | Legacy-125 | 95-0-105 | 47.5% | 40.7–54.4% | -17.4 | 30.7 / 25.0 / 107 |
| Kata-18 | Legacy-115 | 88-0-112 | 44.0% | 37.3–50.9% | -41.9 | 32.7 / 27.0 / 137 |
| Kata-18 | Legacy-105 | 85-0-115 | 42.5% | 35.9–49.4% | -52.5 | 32.8 / 27.0 / 134 |
| Kata-18 | Legacy-90 | 72-0-128 | 36.0% | 29.7–42.9% | -100.0 | 39.9 / 30.0 / 166 |
| Kata-18 | Legacy-80 | 82-0-118 | 41.0% | 34.4–47.9% | -63.2 | 31.3 / 27.0 / 112 |
| Kata-18 | Legacy-70 | 95-0-105 | 47.5% | 40.7–54.4% | -17.4 | 31.4 / 27.0 / 143 |
| Kata-19 | Legacy-129 | 95-0-105 | 47.5% | 40.7–54.4% | -17.4 | 32.7 / 29.0 / 129 |
| Kata-19 | Legacy-125 | 94-0-106 | 47.0% | 40.2–53.9% | -20.9 | 29.0 / 25.0 / 133 |
| Kata-19 | Legacy-115 | 97-0-103 | 48.5% | 41.7–55.4% | -10.4 | 28.6 / 25.0 / 96 |
| Kata-19 | Legacy-105 | 110-0-90 | 55.0% | 48.1–61.7% | +34.9 | 27.0 / 23.0 / 142 |
| Kata-19 | Legacy-90 | 95-0-105 | 47.5% | 40.7–54.4% | -17.4 | 33.6 / 29.0 / 150 |
| Kata-19 | Legacy-80 | 72-0-128 | 36.0% | 29.7–42.9% | -100.0 | 38.7 / 29.0 / 190 |
| Kata-19 | Legacy-70 | 98-0-102 | 49.0% | 42.2–55.9% | -6.9 | 34.6 / 29.0 / 179 |

Aggregate results on the identical seven-opponent schedule:

| Kata checkpoint | W-D-L | Score rate | 95% CI | Aggregate Elo difference |
| --- | ---: | ---: | ---: | ---: |
| Kata-18 | 596-0-804 | 42.6% | 40.0–45.2% | -52.0 |
| Kata-19 | 661-0-739 | 47.2% | 44.6–49.8% | -19.4 |

Kata-19's aggregate confidence interval still ends just below 50%, so the
seven-series result supports “slightly weaker than the legacy pool” rather than
parity. Most individual 200-game intervals include 50%; Legacy-80 is the clear
exception against both Kata checkpoints, and Legacy-90 is a clear exception
against Kata-18.

## Replay statistics

| Set | Series | Games | Mean / median / p90 | Range | First-seat W | Second-seat W | Unique 4-ply / 8-ply openings |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Kata round-robin | 6 | 1,200 | 34.6 / 29 / 61 | 9–165 | 897 | 303 | 742 / 1,021 |
| Kata vs legacy | 14 | 2,800 | 32.7 / 27 / 56 | 9–190 | 2,233 | 567 | 1,578 / 2,134 |
| All games | 20 | 4,000 | 33.3 / 27 / 58 | 9–190 | 3,130 | 870 | 2,232 / 3,114 |

First seat won 78.25% of all games, demonstrating a very large Gomoku
first-player advantage. This does not bias pair scores because every series
gave each checkpoint exactly 100 games in each seat, but it means a comparison
without seat alternation would be unusable. No game was drawn. The long tail is
material: the maximum game lasted 190 plies, versus a median of 27, and those
few games dominated each series' low-batch completion time.

Opening diversity remained high under temperature 0.7: 2,232 distinct
four-ply sequences and 3,114 distinct eight-ply sequences across 4,000 games.
Every persisted `result.json` contains each move, seat/checkpoint identity,
winner, ply count, and the acting network's value estimate.

## Interpretation

- Kata training is making useful progress: epoch 19 improved its matched
  legacy-pool score by 4.6 percentage points over epoch 18.
- The new architecture has not yet surpassed the mature legacy population at
  epoch 19. Continuing Kata training is justified; replacing the legacy best
  checkpoint is not yet justified.
- Legacy-80's unusually strong result and the non-monotonic legacy ordering
  suggest checkpoint regression/variance. Future promotion tests should use a
  fixed champion plus a small opponent panel, not epoch number alone.
- The battle scheduler should continue overlapping multiple series. Three
  simultaneous B128/P200 series used at most about 59.1 GiB RAM and kept the
  GPU busy during otherwise inefficient one-game tails.

## Artifacts

- Remote persistent tournament root:
  `/workspace/alpha-zero/runs/checkpoint-tournament-20260901`
- Local replay mirror:
  `remote-results/runpod-rtx3090-20260901/tournament`
- Independent local summary:
  `remote-results/runpod-rtx3090-20260901/tournament/local-analysis-summary.json`
- Independent local generated report:
  `remote-results/runpod-rtx3090-20260901/tournament/local-analysis-report.md`
- RunPod performance benchmark:
  `reports/runpod-rtx3090-benchmark-2026-09-01.md`

