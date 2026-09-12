use rand::{
    Rng,
    distr::{Distribution, weighted::WeightedIndex},
};

pub fn apply_temperature(policy: &[f32], temperature: f32) -> Vec<f32> {
    assert!(!policy.is_empty());
    assert!(temperature.is_finite() && temperature >= 0.0);
    assert!(
        policy
            .iter()
            .all(|weight| weight.is_finite() && *weight >= 0.0)
    );

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
    assert!(
        policy
            .iter()
            .all(|weight| weight.is_finite() && *weight >= 0.0)
    );
    assert!(policy.iter().any(|weight| *weight > 0.0));
    WeightedIndex::new(policy).unwrap().sample(rng)
}

/// Truncate the sampling distribution, preserving all exact ties at the cutoff.
/// This must never be applied to the policy target used for training.
pub fn apply_top_p(policy: &[f32], top_p: f64) -> Vec<f32> {
    assert!(top_p.is_finite() && top_p > 0.0 && top_p <= 1.0);
    assert!(!policy.is_empty());
    assert!(policy.iter().all(|p| p.is_finite() && *p >= 0.0));
    let total: f64 = policy.iter().map(|&p| f64::from(p)).sum();
    assert!(total > 0.0);
    // Preserve the unfiltered path byte-for-byte, including its RNG behavior.
    if top_p == 1.0 {
        return policy.to_vec();
    }
    let mut sorted: Vec<f32> = policy.iter().copied().filter(|&p| p > 0.0).collect();
    sorted.sort_unstable_by(|a, b| b.total_cmp(a));
    let mut cumulative = 0.0;
    let mut cutoff = sorted[0];
    for p in sorted {
        cutoff = p;
        cumulative += f64::from(p);
        if cumulative >= top_p * total {
            break;
        }
    }
    let retained: f64 = policy
        .iter()
        .filter(|&&p| p >= cutoff)
        .map(|&p| f64::from(p))
        .sum();
    policy
        .iter()
        .map(|&p| {
            if p >= cutoff {
                (f64::from(p) / retained) as f32
            } else {
                0.0
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{apply_temperature, apply_top_p};

    #[test]
    fn nucleus_preserves_boundary_ties_and_zero_support() {
        assert_eq!(
            apply_top_p(&[0.25, 0.25, 0.25, 0.25, 0.0], 0.95),
            [0.25, 0.25, 0.25, 0.25, 0.0]
        );
        assert_eq!(
            apply_top_p(&[0.96, 0.03, 0.01, 0.0], 0.95),
            [1.0, 0.0, 0.0, 0.0]
        );
        let p = apply_top_p(&[0.6, 0.15, 0.15, 0.1], 0.7);
        assert!((p[0] - 2.0 / 3.0).abs() < 1e-6);
        assert_eq!(p[1], p[2]);
        assert_eq!(p[3], 0.0);
        assert_eq!(
            apply_top_p(&[0.6, 0.15, 0.15, 0.1], 1.0),
            [0.6, 0.15, 0.15, 0.1]
        );
    }

    #[test]
    fn nucleus_keeps_at_least_requested_mass_under_permutation() {
        for top_p in [0.01, 0.9, 0.95, 0.98, 0.99, 1.0] {
            let policy = [0.0, 0.005, 0.005, 0.02, 0.07, 0.1, 0.8];
            let actual = apply_top_p(&policy, top_p);
            assert!((actual.iter().sum::<f32>() - 1.0).abs() < 1e-6);
            let retained: f64 = policy
                .iter()
                .zip(&actual)
                .filter(|(_, q)| **q > 0.0)
                .map(|(p, _)| f64::from(*p))
                .sum();
            let total: f64 = policy.iter().map(|p| f64::from(*p)).sum();
            assert!(retained >= top_p * total);
            let mut reversed = policy;
            reversed.reverse();
            let mut result = apply_top_p(&reversed, top_p);
            result.reverse();
            assert_eq!(actual, result);
        }
    }

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
