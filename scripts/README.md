# Experiment tooling

Run Python tools from the repository root with `python3 scripts/<tool>.py`.
Use `--help` for arguments. Data, logs, binaries, connection files, and generated
reports belong in ignored `runs/` (or `validated/` for pinned trainer binaries).

- `experiment_dashboard.py`, `experiment_agent.py`: local read-only HTTP dashboard
  and remote SSH reader. See [dashboard setup](../docs/experiment-dashboard.md).
- `experiment_io.py`, `experiment_statistics.py`, `experiment_metadata.py`:
  shared record validation, statistics, and display identities.
- `experiment_control.py`, `match_plan.py`: queue control and validated match plans.
- `run_self_play_experiments.py`, `run_board_mask_selfplay.py`,
  `self_play_runtime.py`: self-play supervision, ordered replay reconstruction,
  and shared process/resource helpers. The two-run launcher retains the documented
  nucleus/control recipe; it is not a general experiment scheduler.
- `self_play_dashboard.py`, `board_mask_dashboard.py`, `recipe_battle_agent.py`:
  readers for the existing experiment record layouts.
- `collect_self_play_metrics.py`, `collect_self_play_samples.py`,
  `import_self_play_durations.py`: checkpoint-derived dashboard data.
- `benchmark_training_backends.py`, `concurrent_training_smoke.py`,
  `self_play_capacity_probe.py`, `summarize_training_profile.py`: bounded profiling
  and capacity checks. Resource guards currently target single-GPU RunPod hosts
  with cgroup v1 memory accounting.
- `*_runpod_archive.py`, `archive_runpod_workspace.py`: archive, download, and
  verification tools. Preserve the manifests and checksums with the archives.
- `analyze_nucleus_replays.py`, `nucleus_replay_distributions.py`,
  `build_nucleus_report.py`: replay analysis and standalone report generation.
  Source assets are in `report_assets/`; reproduction instructions accompany
  [the curated report](../reports/nucleus-replay-audit-20260909/README.md).
- `migrate_experiment_metadata.py`: idempotent display-metadata backfill;
  see [the format and migration guide](../docs/experiment-metadata.md).
- `archive/`: historical experiment recipes and deployment/handoff scripts.
  Keep these for reproducibility, not as the default deployment workflow.

The `experiment-dashboard.service` template expects a local connection file at
`runs/dashboard-connection.json`; customize it before installation. Connection
files and SSH identities must not be committed. Changing this template does not
change an already installed service.

## Resume an ordered reconstruction

The supervisor resumes its frozen `reconstruction-plan.json`, verifies the
trainer checksum and CUDA receipt, and preserves its exact command and history:

```bash
python3 scripts/run_board_mask_selfplay.py \
  --run-dir runs/selfplay-nucleus --job-id boardmask-p095 \
  --binary validated/alz-boardmask-history-20260910
```

For a new reconstruction, also supply `--source-run`, `--history-epochs`, and
optionally `--epochs`, `--architecture`, `--parallelism`, and `--wait-for-queue`.
The source epoch's recorded training configuration supplies the replay schedule,
LR, seed, batch sizes, and training backend. Prepare the parent run's plan/status
and a matching `cuda-ready.json` first, as for other supervised self-play jobs.
Existing reconstruction plans reject conflicting overrides. For an execution-only
concurrency adjustment after reconstruction, use `--continuation-parallelism 400`;
it persists in `continuation-settings.json`, leaving the frozen reconstruction
recipe intact. Subsequent resumes reuse that setting. Changing other saved
experiment settings requires a separate run directory. No cleanup operation restarts the active trainer.

## Checks

```bash
PYTHONPATH=scripts python3 -m unittest discover -s scripts -p 'test_*.py'
(cd web && npm test && npm run build)
cargo fmt --all -- --check
```

Python HTTP tests need permission to bind loopback sockets. CUDA checks are
explicit opt-in deployment tasks; do not run them against an occupied GPU merely
to validate dashboard changes.
