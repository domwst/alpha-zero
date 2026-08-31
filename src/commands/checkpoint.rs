use anyhow::Result;

use crate::{
    cli::{CheckpointArgs, CheckpointMode},
    training_snapshot::migrate_v1_snapshots,
};

pub fn run(args: CheckpointArgs) -> Result<()> {
    match args.mode {
        CheckpointMode::MigrateV1(args) => {
            let summary = migrate_v1_snapshots(&args.checkpoint_dir, args.apply)?;
            tracing::info!(
                checkpoint_dir = %summary.root.display(),
                apply = summary.apply,
                snapshots_found = summary.snapshots_found,
                legacy_snapshots = summary.legacy_snapshots,
                already_current = summary.already_current,
                migrated = summary.migrated,
                "checkpoint metadata migration complete"
            );
            tracing::info!(
                result = %serde_json::to_string(&summary)?,
                "CHECKPOINT_MIGRATION_RESULT"
            );
            Ok(())
        }
    }
}
