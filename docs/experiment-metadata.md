# Experiment display metadata

Display names are experiment data. The dashboard no longer contains a map from
historical job IDs to checkpoint labels. Its `participants.first` and
`participants.second` describe the two **checkpoint arguments**, independent of
which player takes the first move in each game.

The local dashboard loads `<connection-stem>.metadata.json` beside its SSH
connection file, or an explicit `--metadata PATH`. This supports older remote
readers without a pod deployment. The SSH reader also supports
`experiment-metadata.json` in its queue directory. Both use this schema:

```json
{
  "schema_version": 1,
  "experiments": {
    "my-comparison": {
      "title": "Baseline versus candidate",
      "kind": "battle",
      "participants": {
        "first": {"label": "Baseline · selected pass 11"},
        "second": {"label": "Candidate · selected pass 8"}
      }
    }
  }
}
```

Each participant can additionally contain `model_sha256`, copied from the result's
checkpoint descriptor. Pin this for selected or completed checkpoints. A mismatch
suppresses that saved label and shows the actual checkpoint's architecture and
epoch with a diagnostic note. An unknown job still displays valid checkpoint
identities without a frontend deployment; before descriptors are available its
fallback labels are Checkpoint A/B. Labels are text rendered by Preact, never HTML.

A comparison may also carry one entry (the object inside `experiments`) in a
sibling `<result-stem>.metadata.json`, or embed it as `experiment_metadata` in its
result. The queue registry takes precedence over result metadata; a local dashboard
registry takes precedence over the queue registry. Use a sidecar for existing immutable
results. For offline dashboard archives, use `<archive-stem>.metadata.json` with
the complete schema above. The dashboard HTTP and SSH APIs remain read-only;
metadata is managed through files by the experiment operator. Local registries
are reloaded on each dashboard refresh.

## Backfill existing experiments

```bash
python3 scripts/migrate_experiment_metadata.py \
  --snapshot runs/snapshot.json \
  --output runs/my-queue/experiment-metadata.json
python3 scripts/migrate_experiment_metadata.py \
  --snapshot runs/dashboard-archive.json \
  --output runs/dashboard-archive.metadata.json
```

Inputs can be a raw snapshot or the dashboard/archive wrapper containing
`snapshot`. The one-time historical name catalog lives in
`scripts/migrations/legacy-experiment-labels.json`; runtime code never reads it.
The migration records the source snapshot checksum, backfills checkpoint hashes
when available, preserves previously edited entries, and refuses a conflicting
checkpoint hash. Repeating it on the same inputs does not rewrite the output.
Original results, snapshots, replay buffers, checkpoint metadata, and archive
manifests are never modified. Copy sidecars along with their parent data when
backing up or deploying a dashboard.
