use std::time::{Duration, Instant};

use rand::{
    distributions::{Distribution, WeightedIndex},
    Rng,
};

pub fn sample_policy<R: Rng>(policy: &[f32], temp: f32, rng: &mut R) -> usize {
    let mut policy = policy.to_owned();
    let mx = policy
        .iter()
        .copied()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    for v in &mut policy {
        *v /= mx;
        *v = v.powf(1.0 / (temp + 0.01));
    }

    WeightedIndex::new(policy).unwrap().sample(rng)
}

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
        println!(
            "{}",
            msg.to_string().replace("{t}", &format!("{:?}", passed))
        );
    }
}
