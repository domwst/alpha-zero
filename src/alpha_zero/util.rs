use rand::{
    distr::{weighted::WeightedIndex, Distribution},
    Rng,
};

pub fn apply_temperature(policy: &[f32], temperature: f32) -> Vec<f32> {
    assert!(!policy.is_empty());
    assert!(temperature.is_finite() && temperature >= 0.0);
    assert!(policy
        .iter()
        .all(|weight| weight.is_finite() && *weight >= 0.0));

    let (max_index, max_weight) = policy
        .iter()
        .copied()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .unwrap();
    assert!(max_weight > 0.0);

    if temperature == 0.0 {
        let mut result = vec![0.0; policy.len()];
        result[max_index] = 1.0;
        return result;
    }

    let mut result = policy
        .iter()
        .map(|weight| {
            if *weight == 0.0 {
                0.0
            } else {
                ((*weight / max_weight).ln() / temperature).exp()
            }
        })
        .collect::<Vec<_>>();
    let sum = result.iter().sum::<f32>();
    assert!(sum.is_finite() && sum > 0.0);
    for weight in &mut result {
        *weight /= sum;
    }
    result
}

pub fn sample_policy<R: Rng + ?Sized>(policy: &[f32], rng: &mut R) -> usize {
    assert!(!policy.is_empty());
    assert!(policy
        .iter()
        .all(|weight| weight.is_finite() && *weight >= 0.0));
    assert!(policy.iter().any(|weight| *weight > 0.0));
    WeightedIndex::new(policy).unwrap().sample(rng)
}

#[cfg(test)]
mod tests {
    use super::apply_temperature;

    #[test]
    fn temperature_is_applied_to_policy() {
        let policy = apply_temperature(&[0.9, 0.1], 0.5);
        assert!((policy[0] - 81.0 / 82.0).abs() < 1e-6);
        assert!((policy[1] - 1.0 / 82.0).abs() < 1e-6);
    }

    #[test]
    fn zero_temperature_selects_the_maximum() {
        assert_eq!(apply_temperature(&[0.1, 0.7, 0.2], 0.0), [0.0, 1.0, 0.0]);
    }
}
