mod cli;
mod commands;
mod training_snapshot;

use clap::Parser;

use cli::Cli;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    commands::run(Cli::parse().command).await
}
