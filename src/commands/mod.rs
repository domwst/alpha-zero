mod battle;
mod benchmark;
mod common;
mod play;
mod train;

use anyhow::Result;

use crate::cli::{Command, PlayMode};

pub async fn run(command: Command) -> Result<()> {
    match command {
        Command::Train(args) => train::run(args).await,
        Command::Play(args) => match args.mode {
            PlayMode::Human(args) => play::run_human(args).await,
            PlayMode::Policy(args) => play::run_policy(args).await,
        },
        Command::Battle(args) => battle::run(args).await,
        Command::Benchmark(args) => benchmark::run(args.mode).await,
    }
}
