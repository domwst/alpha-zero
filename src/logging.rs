use std::{
    env,
    io::{self, IsTerminal},
};

use time::macros::format_description;
use tracing_subscriber::{EnvFilter, fmt::time::UtcTime};

const TIMESTAMP_FORMAT: &[time::format_description::FormatItem<'static>] =
    format_description!("[year]-[month]-[day]T[hour]:[minute]:[second].[subsecond digits:3]Z");

pub fn init() {
    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn,alz=info"));
    let ansi_setting = env::var("ALZ_LOG_ANSI").ok();
    let ansi = resolve_ansi(ansi_setting.as_deref(), io::stdout().is_terminal());
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_timer(UtcTime::new(TIMESTAMP_FORMAT))
        .with_file(true)
        .with_target(false)
        .with_line_number(true)
        .with_ansi(ansi)
        .with_writer(io::stdout)
        .init();
}

fn resolve_ansi(setting: Option<&str>, stdout_is_terminal: bool) -> bool {
    match setting.map(str::to_ascii_lowercase).as_deref() {
        Some("always" | "1" | "true" | "yes") => true,
        Some("never" | "0" | "false" | "no") => false,
        _ => stdout_is_terminal,
    }
}

#[cfg(test)]
mod tests {
    use time::OffsetDateTime;

    use super::{TIMESTAMP_FORMAT, resolve_ansi};

    #[test]
    fn timestamp_has_exactly_six_fractional_digits() {
        let timestamp = OffsetDateTime::from_unix_timestamp_nanos(123_456_789_000).unwrap();
        assert_eq!(
            timestamp.format(TIMESTAMP_FORMAT).unwrap(),
            "1970-01-01T00:02:03.456Z"
        );
    }

    #[test]
    fn ansi_override_takes_precedence_over_terminal_detection() {
        assert!(resolve_ansi(Some("always"), false));
        assert!(!resolve_ansi(Some("never"), true));
        assert!(resolve_ansi(None, true));
        assert!(!resolve_ansi(Some("auto"), false));
    }
}
