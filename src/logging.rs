use std::io::{self, IsTerminal};

use tracing_subscriber::{EnvFilter, fmt::time::UtcTime};

pub fn init() {
    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn,alz=info"));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_timer(UtcTime::rfc_3339())
        .with_file(true)
        .with_line_number(true)
        .with_ansi(io::stdout().is_terminal())
        .with_writer(io::stdout)
        .init();
}
