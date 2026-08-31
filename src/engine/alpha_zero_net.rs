use tch::{Kind, Tensor};

#[derive(Debug)]
pub struct NetworkOutput {
    pub values: Tensor,
    pub policy_logits: Tensor,
}

pub trait AlphaZeroNet {
    /// Returns values and unnormalized policy logits.
    fn forward_t(&self, xs: &Tensor, is_training: bool) -> NetworkOutput;
}

/// Applies a policy log-softmax while preserving all non-batch dimensions.
pub fn policy_log_probabilities(policy_logits: &Tensor) -> Tensor {
    assert!(
        policy_logits.size().len() >= 2,
        "policy logits must have a batch and at least one action dimension"
    );
    let shape = policy_logits.size();
    policy_logits
        .flatten(1, -1)
        .log_softmax(1, Kind::Float)
        .view(shape.as_slice())
}

/// Masks illegal actions and applies a policy softmax while preserving shape.
pub fn masked_policy_probabilities(policy_logits: &Tensor, legal_mask: &Tensor) -> Tensor {
    assert_eq!(
        policy_logits.size(),
        legal_mask.size(),
        "policy logits and legal mask must have identical shapes"
    );
    assert_eq!(
        legal_mask.kind(),
        Kind::Bool,
        "legal policy mask must be Boolean"
    );
    let shape = policy_logits.size();
    policy_logits
        .flatten(1, -1)
        .masked_fill(&legal_mask.flatten(1, -1).logical_not(), f64::NEG_INFINITY)
        .softmax(1, Kind::Float)
        .view(shape.as_slice())
}

#[cfg(test)]
mod tests {
    use tch::{Device, Kind, Tensor};

    use super::{masked_policy_probabilities, policy_log_probabilities};

    #[test]
    fn policy_log_softmax_preserves_shape_and_normalizes_actions() {
        let logits = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).view([2, 1, 3]);

        let log_probabilities = policy_log_probabilities(&logits);

        assert_eq!(log_probabilities.size(), [2, 1, 3]);
        let sums = log_probabilities.exp().flatten(1, -1).sum_dim_intlist(
            [1].as_slice(),
            false,
            Kind::Float,
        );
        assert!(sums.allclose(
            &Tensor::ones([2], (Kind::Float, Device::Cpu)),
            1e-6,
            1e-6,
            false
        ));
    }

    #[test]
    fn masked_policy_softmax_zeros_illegal_actions_and_normalizes_legal_actions() {
        let logits = Tensor::from_slice(&[1.0f32, 20.0, 3.0, 4.0, 5.0, 30.0]).view([2, 1, 3]);
        let mask = Tensor::from_slice(&[true, false, true, true, true, false]).view([2, 1, 3]);

        let probabilities = masked_policy_probabilities(&logits, &mask);

        assert_eq!(probabilities.size(), [2, 1, 3]);
        let values = Vec::<f32>::try_from(probabilities.flatten(0, -1)).unwrap();
        assert_eq!(values[1], 0.0);
        assert_eq!(values[5], 0.0);
        assert!((values[0] + values[2] - 1.0).abs() < 1e-6);
        assert!((values[3] + values[4] - 1.0).abs() < 1e-6);
        assert!(values[2] > values[0]);
        assert!(values[4] > values[3]);
    }

    #[test]
    fn legal_softmax_is_unchanged_when_global_log_softmax_is_removed() {
        let logits = Tensor::from_slice(&[-3.0f32, 8.0, 0.5, 2.0, -1.0, 4.0]).view([2, 1, 3]);
        let mask = Tensor::from_slice(&[true, false, true, false, true, true]).view([2, 1, 3]);

        let old_probabilities =
            masked_policy_probabilities(&policy_log_probabilities(&logits), &mask);
        let new_probabilities = masked_policy_probabilities(&logits, &mask);

        assert!(old_probabilities.allclose(&new_probabilities, 1e-6, 1e-6, false));
    }
}
