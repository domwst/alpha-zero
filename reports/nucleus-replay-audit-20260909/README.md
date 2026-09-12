# Offline nucleus audit, 2026-09-09

Inputs: archived self-play replay checkpoints 60–69. No network inference,
training, or modified self-play was performed. The archived collector saved
temperature-adjusted visit distributions as training targets; this audit applies
top-p directly to those stored distributions, without another temperature transform.
It does not project results for the newly adopted paired 0.7 schedule.

The exporter validates replay metadata and policies using the repository loader,
deduplicates identical complete games, and reconstructs played actions from
consecutive states. Every recovered action is checked by applying the game's
transition function, including the canonical player flip. Final actions are
unavailable because terminal boards are not in the training replay.

The filter retains the smallest descending-probability prefix with mass at least
the threshold, including the entire exact-tie group at the boundary. It never
removes an arbitrary subset of equal-weight moves. The mean removed mass therefore
need not equal `1 - top_p`.

Results: [summary.md](summary.md), [summary.json](summary.json).
[Export metadata](sources.json) contains input replay hashes and exact deduplication counts:
17,500 unique games, 391,212 positions, 373,712 reconstructed actions, and 22,500
duplicate game entries removed. Aggregate counts were checked against this metadata;
top-p 1.0 preserves every observed action. Four Python audit tests and one Rust
transition-reconstruction test passed.
The large intermediate JSONL is under `runs/nucleus-replay-audit-20260909/`.

Reproduce from the repository root (Bash):

```bash
mkdir -p runs/nucleus-replay-audit-20260909
audit_replay_root=runs/pod-backup-rpii1ijfvmno97-20260908/restored/workspace/alpha-zero/replays
./run.sh cargo run --locked --example export_replay_policies -- \
  runs/nucleus-replay-audit-20260909/policies.jsonl \
  "$audit_replay_root"/0000006{0,1,2,3,4,5,6,7,8,9}
python3 scripts/analyze_nucleus_replays.py \
  runs/nucleus-replay-audit-20260909/policies.jsonl \
  reports/nucleus-replay-audit-20260909/summary.json
python3 scripts/nucleus_replay_distributions.py \
  runs/nucleus-replay-audit-20260909/policies.jsonl \
  reports/nucleus-replay-audit-20260909/distributions.json \
  --summary reports/nucleus-replay-audit-20260909/summary.json
curl --fail --location https://cdn.plot.ly/plotly-4.0.0.min.js \
  --output runs/nucleus-replay-audit-20260909/plotly-4.0.0.min.js
python3 scripts/build_nucleus_report.py reports/nucleus-replay-audit-20260909 \
  --plotly runs/nucleus-replay-audit-20260909/plotly-4.0.0.min.js \
  --output runs/nucleus-replay-audit-20260909/report.html
```

The HTML report embeds Plotly 4.0.0, its styles, and aggregate data for offline use.
Its two-dimensional heatmap counts original versus retained nonzero moves, with
selectable top-p and bin resolution. Conditional-mean curves compare every top-p
on the same axes. All six charts support hover, zoom, pan, and image export.
Stage and seat filters apply to all charts. The histogram/cumulative selector
applies to the four one-dimensional plots.

`distributions.json` schema 2 includes sparse `support_joint` triples
`[original nonzero count, retained count, number of positions]` in each disjoint
stage/seat group. These are measured from replay rows, not reconstructed from
marginal histograms. The generator verifies both marginals and removed-count
histograms against the joint counts, and checks that top-p 1.0 stays on the
no-change diagonal. The embedded Plotly library's source URL and checksum are in
[`scripts/report_assets/nucleus/plotly-4.0.0.min.source.json`](../../scripts/report_assets/nucleus/plotly-4.0.0.min.source.json).
The builder verifies this checksum. Templates, chart code, and design tokens live
in that same source directory; generated HTML and upload receipts belong under
ignored `runs/`, alongside the downloaded plotting library.

These are conditional, historical-position measurements. In an actual rerun,
subsequent states would change after selecting a different action. Neither an
excluded action nor the eventual game outcome establishes whether that action was
a blunder. Seat/outcome breakdowns are descriptive and are not causal comparisons.
