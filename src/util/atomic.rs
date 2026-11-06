use std::sync::atomic::{AtomicU64, Ordering};

pub(crate) trait AtomicU64Ext {
    /// Adds with separate relaxed load and store operations.
    ///
    /// This avoids an atomic read-modify-write, but is only correct when at most
    /// one writer updates the value. Concurrent relaxed readers are allowed.
    fn add_single_writer(&self, value: u64) -> u64;
}

impl AtomicU64Ext for AtomicU64 {
    #[inline]
    fn add_single_writer(&self, value: u64) -> u64 {
        let old = self.load(Ordering::Relaxed);
        self.store(old.wrapping_add(value), Ordering::Relaxed);
        old
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::AtomicU64Ext;

    #[test]
    fn single_writer_add_updates_the_counter() {
        let counter = AtomicU64::new(7);

        assert_eq!(counter.add_single_writer(5), 7);

        assert_eq!(counter.load(Ordering::Relaxed), 12);
    }

    #[test]
    fn single_writer_add_wraps_like_fetch_add() {
        let counter = AtomicU64::new(u64::MAX);

        assert_eq!(counter.add_single_writer(1), u64::MAX);

        assert_eq!(counter.load(Ordering::Relaxed), 0);
    }
}
