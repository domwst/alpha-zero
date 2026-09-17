mod analyze;
mod battle;
mod benchmark;
pub mod benchmark_executor;
mod common;
mod play;
mod train;
mod train_replay;

use anyhow::Result;

use crate::cli::{Command, PlayMode};

pub async fn run(command: Command, batch_grid: &[usize]) -> Result<()> {
    let config = match &command {
        Command::Train(args) => Some(serde_json::to_value(args)?),
        Command::TrainReplay(args) => Some(serde_json::to_value(args)?),
        Command::Battle(args) => Some(serde_json::to_value(args)?),
        _ => None,
    };
    if let Some(config) = config {
        alz::engine::telemetry::event("configured", config)?;
    }
    match command {
        Command::Analyze(args) => analyze::run(args).await,
        Command::Train(args) => train::run(args, batch_grid).await,
        Command::TrainReplay(args) => train_replay::run(args),
        Command::Play(args) => match args.mode {
            PlayMode::Human(args) => play::run_human(args, batch_grid).await,
            PlayMode::Policy(args) => play::run_policy(args, batch_grid).await,
        },
        Command::Battle(args) => battle::run(args, batch_grid).await,
        Command::Benchmark(args) => benchmark::run(args.mode, batch_grid).await,
    }
}
