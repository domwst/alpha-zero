mod cli;
mod commands;
mod logging;

use std::process::ExitCode;

use clap::Parser;

use cli::Cli;

#[tokio::main]
async fn main() -> ExitCode {
    logging::init();
    let cli = Cli::parse();
    if let Err(error) = alz::engine::validate_batch_grid(&cli.inference_batch_grid) {
        tracing::error!(%error, "invalid batch grid");
        return ExitCode::FAILURE;
    }
    let _ = alz::engine::telemetry::event(
        "native_started",
        serde_json::json!({"pid":std::process::id()}),
    );
    match commands::run(cli.command, &cli.inference_batch_grid).await {
        Ok(()) => {
            let _ = alz::engine::telemetry::event("native_completed", serde_json::json!({}));
            ExitCode::SUCCESS
        }
        Err(error) => {
            let _ = alz::engine::telemetry::event(
                "native_failed",
                serde_json::json!({"error":format!("{error:#}")}),
            );
            tracing::error!(error = ?error, "command failed");
            ExitCode::FAILURE
        }
    }
}
