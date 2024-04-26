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
        *v = v.powf(1.0 / temp);
    }

    WeightedIndex::new(policy).unwrap().sample(rng)
}
