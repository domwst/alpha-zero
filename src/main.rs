mod cli;
mod commands;
mod logging;

use std::process::ExitCode;

use clap::Parser;

use cli::Cli;

#[tokio::main]
async fn main() -> ExitCode {
    logging::init();
    match commands::run(Cli::parse().command).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            tracing::error!(error = ?error, "command failed");
            ExitCode::FAILURE
        }
    }
}
