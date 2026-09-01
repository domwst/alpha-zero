use std::borrow::Borrow;

use serde::{Deserialize, Serialize};
use tch::{Tensor, nn::Path};

use crate::engine::{AlphaZeroNet, NetworkOutput};

use super::{GomokuKataNet, GomokuResNet};

/// Persistent, versioned description of a network architecture.
///
/// A variant's meaning, parameter names, parameter registration order, and
/// forward semantics must never change. Introduce a new versioned variant for
/// any incompatible change.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "architecture", deny_unknown_fields)]
pub enum ModelSpec {
    #[serde(rename = "legacy_resnet_v1")]
    #[default]
    LegacyResNetV1,
    #[serde(rename = "kata_v1")]
    KataV1,
}

impl ModelSpec {
    pub fn architecture_id(&self) -> &'static str {
        match self {
            Self::LegacyResNetV1 => "legacy_resnet_v1",
            Self::KataV1 => "kata_v1",
        }
    }
}

/// Runtime-dispatched network used by commands and the batched executor.
///
/// Keeping all architectures behind one concrete enum allows a battle to load
/// different model variants while retaining static dispatch in the rest of the
/// engine. The match happens once per network batch.
#[derive(Debug)]
pub enum GomokuModel {
    LegacyResNetV1(GomokuResNet),
    KataV1(GomokuKataNet),
}

impl GomokuModel {
    pub fn new<'a, P: Borrow<Path<'a>>>(path: P, spec: &ModelSpec) -> Self {
        match spec {
            ModelSpec::LegacyResNetV1 => Self::LegacyResNetV1(GomokuResNet::new(path)),
            ModelSpec::KataV1 => Self::KataV1(GomokuKataNet::new(path)),
        }
    }
}

impl AlphaZeroNet for GomokuModel {
    fn forward_t(&self, input: &Tensor, is_training: bool) -> NetworkOutput {
        match self {
            Self::LegacyResNetV1(network) => network.forward_t(input, is_training),
            Self::KataV1(network) => network.forward_t(input, is_training),
        }
    }
}

#[cfg(test)]
mod tests {
    use tch::{Device, Kind, Tensor, nn};

    use crate::engine::AlphaZeroNet;

    use super::{GomokuModel, ModelSpec};

    #[test]
    fn model_spec_has_stable_serialization() {
        for (spec, expected_json) in [
            (
                ModelSpec::LegacyResNetV1,
                r#"{"architecture":"legacy_resnet_v1"}"#,
            ),
            (ModelSpec::KataV1, r#"{"architecture":"kata_v1"}"#),
        ] {
            let json = serde_json::to_string(&spec).unwrap();
            assert_eq!(json, expected_json);
            assert_eq!(serde_json::from_str::<ModelSpec>(&json).unwrap(), spec);
        }
    }

    #[test]
    fn runtime_model_preserves_network_contract() {
        for spec in [ModelSpec::LegacyResNetV1, ModelSpec::KataV1] {
            let var_store = nn::VarStore::new(Device::Cpu);
            let network = GomokuModel::new(var_store.root(), &spec);
            let input = Tensor::zeros([2, 2, 19, 19], (Kind::Float, Device::Cpu));

            let output = network.forward_t(&input, false);

            assert_eq!(output.values.size(), [2]);
            assert_eq!(output.policy_logits.size(), [2, 19, 19]);
        }
    }
}
