use std::time::{Duration, Instant};

pub struct Timer {
    start: Instant,
}

impl Timer {
    pub fn new() -> Self {
        Self {
            start: Instant::now(),
        }
    }

    pub fn passed(&self) -> Duration {
        Instant::now() - self.start
    }

    pub fn print_if_greater(&self, threshold: Duration, msg: &str) {
        let passed = self.passed();
        if passed < threshold {
            return;
        }
        tracing::info!("{}", msg.replace("{t}", &format!("{passed:?}")));
    }
}

impl Default for Timer {
    fn default() -> Self {
        Self::new()
    }
}
