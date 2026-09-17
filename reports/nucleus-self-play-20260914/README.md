# Nucleus sampling: research disposition, 2026-09-14

Decision: use `top_p = 1.0` (no nucleus filtering) for new routine self-play.
Retain lower cutoffs as an advanced experimental option for reproduction. Further
nucleus experiments are deprioritized; this is a project decision, not evidence
that nucleus sampling is universally harmful.

The native trainer and self-play benchmark already defaulted to 1.0. The dashboard
inherits that default. Existing explicit 0.95 job specifications, saved games,
checkpoint metadata, and historical recipes retain their original meaning.

## Observed comparison

[Sharp Temp E17 vs. Nucleus sampling E54](https://dashboard.oleja.dev/experiments?job=bedf611b-9298-4b76-b51d-15d39dee407c)
completed 1,000 games at S4000/P400/B128, with evaluation temperature 0.7 for both
models and 500 games in each seat. Epoch labels here are one-based.

| Checkpoint | Wins playing first | Wins playing second | Total wins | Losses | Draws |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sharp Temp, top-p 1.0, E17 | 251 / 500 | 76 / 500 | 327 | 673 | 0 |
| Paired temperature, top-p 0.95, E54 | 424 / 500 | 249 / 500 | 673 | 327 | 0 |

The older nucleus model won 67.3% of these games. This comparison **does not
establish that nucleus sampling failed**, nor does it isolate a benefit from it.
The checkpoints differ in training age, initialization/seed, and replay history,
as well as the sampling and temperature settings. The board-mask nucleus series
includes reconstruction from earlier replay buffers; the sharp run started fresh.

The self-play temperature schedules were:

- **Paired, top-p 0.95:** temperature 1.0 on moves 1–6; seven equal steps on
  successive move pairs through moves 19–20, reaching 0.7; then 0.7. The legacy
  job specification omits the schedule field and uses this original default.
- **Sharp, top-p 1.0:** temperature 1.0 on moves 1–5, 0.7 on move 6, 0.6 on move 7,
  then 0.5. This deliberately gives the second player lower temperatures during
  the transition.

A causal conclusion would require matched seeds, architecture, training budget,
temperature schedule, and checkpoint-selection criteria, changing only top-p.
No such conclusion is claimed here.

## Preserved evidence and reproduction

[results.json](results.json) archives the comparison's aggregate outcome, seat,
game-length, duration and completion distributions, job specifications, and exact
checkpoint model hashes. Source job specifications were captured after completion
of the comparison; they describe the series and do not replace the compared
checkpoint's own identity. Metadata epochs in JSON are zero-based.

The earlier [offline replay audit](../nucleus-replay-audit-20260909/README.md)
contains its source hashes, distributions, and instructions to regenerate the
interactive HTML report. It measures how filtering changes historical policies;
it is not a playing-strength experiment. Its generator and source assets remain
because they are required to reproduce that report. Generated HTML, downloaded
libraries and deployment scratch files stay in ignored `runs/`.

To reproduce the match, restore the two checkpoints identified in `results.json`,
register them in the job service, and use the archived comparison options and
input mapping. Retain `--top-p 0.95` explicitly when reproducing the nucleus
training recipe. Filtering remains after temperature, with boundary ties kept;
the policy training target is the unfiltered search visit distribution.
