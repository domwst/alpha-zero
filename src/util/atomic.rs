use std::sync::atomic::{AtomicU64, Ordering};

pub(crate) trait AtomicU64Ext {
    /// Adds with separate relaxed load and store operations.
    ///
    /// This avoids an atomic read-modify-write, but is only correct when at most
    /// one writer updates the value. Concurrent relaxed readers are allowed.
    fn add_single_writer(&self, value: u64);
}

impl AtomicU64Ext for AtomicU64 {
    #[inline]
    fn add_single_writer(&self, value: u64) {
        self.store(
            self.load(Ordering::Relaxed).wrapping_add(value),
            Ordering::Relaxed,
        );
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::AtomicU64Ext;

    #[test]
    fn single_writer_add_updates_the_counter() {
        let counter = AtomicU64::new(7);

        counter.add_single_writer(5);

        assert_eq!(counter.load(Ordering::Relaxed), 12);
    }

    #[test]
    fn single_writer_add_wraps_like_fetch_add() {
        let counter = AtomicU64::new(u64::MAX);

        counter.add_single_writer(1);

        assert_eq!(counter.load(Ordering::Relaxed), 0);
    }
}
