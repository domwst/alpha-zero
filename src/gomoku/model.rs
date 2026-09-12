use std::borrow::Borrow;

use serde::{Deserialize, Serialize};
use tch::{Tensor, nn::Path};

use crate::engine::{AlphaZeroNet, NetworkOutput};

use super::{GomokuKataNet, GomokuResNet, kata_nn::Activation};

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
    #[serde(rename = "kata_gelu_v1")]
    KataGeluV1,
    #[serde(rename = "kata_value64_v1")]
    KataValue64V1,
    #[serde(rename = "kata_value64x2_v1")]
    KataValue64x2V1,
    #[serde(rename = "kata_gelu_value64_v1")]
    KataGeluValue64V1,
    #[serde(rename = "kata_gelu_value64x2_v1")]
    KataGeluValue64x2V1,
    #[serde(rename = "kata_gelu_boardmask_value64x2_v1")]
    KataGeluBoardMaskValue64x2V1,
    #[serde(rename = "kata_gelu_b16c32_value64x2_v1")]
    KataGeluB16C32Value64x2V1,
    #[serde(rename = "kata_gelu_b16c32g3_value64x2_v1")]
    KataGeluB16C32G3Value64x2V1,
    #[serde(rename = "kata_gelu_b10c48_value64x2_v1")]
    KataGeluB10C48Value64x2V1,
    #[serde(rename = "kata_pool_v1")]
    KataPoolV1,
    #[serde(rename = "kata_gelu_pool_v1")]
    KataGeluPoolV1,
    #[serde(rename = "kata_pool_value64_v1")]
    KataPoolValue64V1,
    #[serde(rename = "kata_pool_value64x2_v1")]
    KataPoolValue64x2V1,
    #[serde(rename = "kata_gelu_pool_value64_v1")]
    KataGeluPoolValue64V1,
    #[serde(rename = "kata_gelu_pool_value64x2_v1")]
    KataGeluPoolValue64x2V1,
}

impl ModelSpec {
    pub fn architecture_id(&self) -> &'static str {
        match self {
            Self::LegacyResNetV1 => "legacy_resnet_v1",
            Self::KataV1 => "kata_v1",
            Self::KataGeluV1 => "kata_gelu_v1",
            Self::KataValue64V1 => "kata_value64_v1",
            Self::KataValue64x2V1 => "kata_value64x2_v1",
            Self::KataGeluValue64V1 => "kata_gelu_value64_v1",
            Self::KataGeluValue64x2V1 => "kata_gelu_value64x2_v1",
            Self::KataGeluBoardMaskValue64x2V1 => "kata_gelu_boardmask_value64x2_v1",
            Self::KataGeluB16C32Value64x2V1 => "kata_gelu_b16c32_value64x2_v1",
            Self::KataGeluB16C32G3Value64x2V1 => "kata_gelu_b16c32g3_value64x2_v1",
            Self::KataGeluB10C48Value64x2V1 => "kata_gelu_b10c48_value64x2_v1",
            Self::KataPoolV1 => "kata_pool_v1",
            Self::KataGeluPoolV1 => "kata_gelu_pool_v1",
            Self::KataPoolValue64V1 => "kata_pool_value64_v1",
            Self::KataPoolValue64x2V1 => "kata_pool_value64x2_v1",
            Self::KataGeluPoolValue64V1 => "kata_gelu_pool_value64_v1",
            Self::KataGeluPoolValue64x2V1 => "kata_gelu_pool_value64x2_v1",
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
    KataGeluV1(GomokuKataNet),
    KataValue64V1(GomokuKataNet),
    KataValue64x2V1(GomokuKataNet),
    KataGeluValue64V1(GomokuKataNet),
    KataGeluValue64x2V1(GomokuKataNet),
    KataGeluBoardMaskValue64x2V1(GomokuKataNet),
    KataGeluB16C32Value64x2V1(GomokuKataNet),
    KataGeluB16C32G3Value64x2V1(GomokuKataNet),
    KataGeluB10C48Value64x2V1(GomokuKataNet),
    KataPoolV1(GomokuKataNet),
    KataGeluPoolV1(GomokuKataNet),
    KataPoolValue64V1(GomokuKataNet),
    KataPoolValue64x2V1(GomokuKataNet),
    KataGeluPoolValue64V1(GomokuKataNet),
    KataGeluPoolValue64x2V1(GomokuKataNet),
}

impl GomokuModel {
    pub fn new<'a, P: Borrow<Path<'a>>>(path: P, spec: &ModelSpec) -> Self {
        match spec {
            ModelSpec::KataGeluBoardMaskValue64x2V1 => {
                Self::KataGeluBoardMaskValue64x2V1(GomokuKataNet::with_board_mask(path))
            }
            ModelSpec::LegacyResNetV1 => Self::LegacyResNetV1(GomokuResNet::new(path)),
            ModelSpec::KataGeluB16C32Value64x2V1 => {
                Self::KataGeluB16C32Value64x2V1(GomokuKataNet::with_trunk(path, 16, 32))
            }
            ModelSpec::KataGeluB16C32G3Value64x2V1 => {
                Self::KataGeluB16C32G3Value64x2V1(GomokuKataNet::with_third_global(path))
            }
            ModelSpec::KataGeluB10C48Value64x2V1 => {
                Self::KataGeluB10C48Value64x2V1(GomokuKataNet::with_trunk(path, 10, 48))
            }
            ModelSpec::KataPoolV1 => Self::KataPoolV1(GomokuKataNet::with_pooling(
                path,
                Activation::Relu,
                &[10],
                true,
            )),
            ModelSpec::KataGeluPoolV1 => Self::KataGeluPoolV1(GomokuKataNet::with_pooling(
                path,
                Activation::Gelu,
                &[10],
                true,
            )),
            ModelSpec::KataPoolValue64V1 => Self::KataPoolValue64V1(GomokuKataNet::with_pooling(
                path,
                Activation::Relu,
                &[64],
                true,
            )),
            ModelSpec::KataPoolValue64x2V1 => Self::KataPoolValue64x2V1(
                GomokuKataNet::with_pooling(path, Activation::Relu, &[64, 64], true),
            ),
            ModelSpec::KataGeluPoolValue64V1 => Self::KataGeluPoolValue64V1(
                GomokuKataNet::with_pooling(path, Activation::Gelu, &[64], true),
            ),
            ModelSpec::KataGeluPoolValue64x2V1 => Self::KataGeluPoolValue64x2V1(
                GomokuKataNet::with_pooling(path, Activation::Gelu, &[64, 64], true),
            ),
            ModelSpec::KataV1 => Self::KataV1(GomokuKataNet::new(path)),
            ModelSpec::KataGeluV1 => Self::KataGeluV1(GomokuKataNet::new_gelu(path)),
            ModelSpec::KataValue64V1 => Self::KataValue64V1(GomokuKataNet::with_value_head(
                path,
                Activation::Relu,
                &[64],
            )),
            ModelSpec::KataValue64x2V1 => Self::KataValue64x2V1(GomokuKataNet::with_value_head(
                path,
                Activation::Relu,
                &[64, 64],
            )),
            ModelSpec::KataGeluValue64V1 => Self::KataGeluValue64V1(
                GomokuKataNet::with_value_head(path, Activation::Gelu, &[64]),
            ),
            ModelSpec::KataGeluValue64x2V1 => Self::KataGeluValue64x2V1(
                GomokuKataNet::with_value_head(path, Activation::Gelu, &[64, 64]),
            ),
        }
    }
}

impl AlphaZeroNet for GomokuModel {
    fn forward_t(&self, input: &Tensor, is_training: bool) -> NetworkOutput {
        match self {
            Self::LegacyResNetV1(network) => network.forward_t(input, is_training),
            Self::KataV1(network) => network.forward_t(input, is_training),
            Self::KataGeluV1(network)
            | Self::KataGeluBoardMaskValue64x2V1(network)
            | Self::KataGeluB16C32Value64x2V1(network)
            | Self::KataGeluB16C32G3Value64x2V1(network)
            | Self::KataGeluB10C48Value64x2V1(network)
            | Self::KataValue64V1(network)
            | Self::KataValue64x2V1(network)
            | Self::KataGeluValue64V1(network)
            | Self::KataGeluValue64x2V1(network)
            | Self::KataPoolV1(network)
            | Self::KataGeluPoolV1(network)
            | Self::KataPoolValue64V1(network)
            | Self::KataPoolValue64x2V1(network)
            | Self::KataGeluPoolValue64V1(network)
            | Self::KataGeluPoolValue64x2V1(network) => network.forward_t(input, is_training),
        }
    }
}

#[cfg(test)]
mod tests {
    use tch::{Device, Kind, Tensor, nn};

    fn check_board_mask(device: Device) {
        use tch::nn::OptimizerConfig;
        let baseline_vs = nn::VarStore::new(device);
        let baseline = GomokuModel::new(baseline_vs.root(), &ModelSpec::KataGeluValue64x2V1);
        let vs = nn::VarStore::new(device);
        let spec = ModelSpec::KataGeluBoardMaskValue64x2V1;
        let network = GomokuModel::new(vs.root(), &spec);
        let tensors = vs.variables();
        assert_eq!(tensors.len(), baseline_vs.variables().len() + 1);
        for (name, tensor) in baseline_vs.variables() {
            assert_eq!(tensor.size(), tensors[&name].size());
            // Avoid global RNG races with other tests. The experiment separately
            // verifies same-seed initialization against archived tensor hashes.
            tch::no_grad(|| tensors[&name].shallow_clone().copy_(&tensor));
        }
        let input = Tensor::randn([4, 2, 19, 19], (Kind::Float, device));
        let mut mask = tensors["board_mask_conv.weight"].shallow_clone();
        tch::no_grad(|| {
            let _ = mask.zero_();
        });
        let a = baseline.forward_t(&input, false);
        let b = network.forward_t(&input, false);
        assert!(a.values.equal(&b.values));
        assert!(a.policy_logits.equal(&b.policy_logits));
        let mut optimizer = nn::Adam::default().build(&vs, 0.001).unwrap();
        for _ in 0..3 {
            let out = network.forward_t(&input, true);
            optimizer.backward_step(
                &(out.values.square().mean(Kind::Float)
                    + out.policy_logits.square().mean(Kind::Float)),
            );
            for tensor in vs.trainable_variables() {
                assert!(bool::try_from(tensor.grad().isfinite().all()).unwrap());
            }
        }
        assert!(mask.abs().max().double_value(&[]) > 0.0);
        let mut saved = Vec::new();
        vs.save_to_stream(&mut saved).unwrap();
        let mut restored_vs = nn::VarStore::new(device);
        let restored = GomokuModel::new(restored_vs.root(), &spec);
        restored_vs
            .load_from_stream(std::io::Cursor::new(saved))
            .unwrap();
        let actual = restored.forward_t(&input, false);
        let expected = network.forward_t(&input, false);
        assert!(actual.values.equal(&expected.values));
        assert!(actual.policy_logits.equal(&expected.policy_logits));
        assert_eq!(
            serde_json::from_str::<ModelSpec>(&serde_json::to_string(&spec).unwrap()).unwrap(),
            spec
        );
    }

    #[test]
    fn board_mask_preserves_baseline_and_trains_and_reloads() {
        check_board_mask(Device::Cpu);
    }

    #[test]
    #[ignore = "requires CUDA; run with --test-threads=1"]
    fn board_mask_cuda_trains_and_reloads() {
        assert!(tch::Cuda::is_available());
        check_board_mask(Device::Cuda(0));
    }

    use crate::engine::AlphaZeroNet;

    use super::{GomokuModel, ModelSpec};

    #[test]
    fn capacity_variants_keep_declared_global_blocks_and_fixed_heads() {
        use tch::nn::OptimizerConfig;
        for (spec, blocks, channels, globals) in [
            (ModelSpec::KataGeluB16C32Value64x2V1, 16, 32, &[3, 7][..]),
            (
                ModelSpec::KataGeluB16C32G3Value64x2V1,
                16,
                32,
                &[3, 7, 11][..],
            ),
            (ModelSpec::KataGeluB10C48Value64x2V1, 10, 48, &[3, 7][..]),
        ] {
            let vs = nn::VarStore::new(Device::Cpu);
            let net = GomokuModel::new(vs.root(), &spec);
            let vars = vs.variables();
            assert_eq!(vars["conv.weight"].size(), [channels, 2, 3, 3]);
            assert_eq!(vars["value_head.conv.weight"].size(), [32, channels, 1, 1]);
            assert_eq!(vars["value_head.fc1.weight"].size(), [64, 64]);
            assert_eq!(
                vars["policy_head.conv1.weight"].size(),
                [10, channels, 1, 1]
            );
            for i in 0..blocks {
                if globals.contains(&i) {
                    assert!(vars.contains_key(&format!("global_block_{i}.ctoc.weight")));
                } else {
                    assert_eq!(
                        vars[&format!("res_block_{i}.conv1.weight")].size(),
                        [channels, channels, 3, 3]
                    );
                    assert_eq!(
                        vars[&format!("res_block_{i}.conv2.weight")].size(),
                        [channels, channels, 3, 3]
                    );
                }
            }
            assert_eq!(
                vars.keys()
                    .filter(|name| name.ends_with("ctoc.weight"))
                    .count(),
                globals.len()
            );
            let mut optimizer = nn::Adam::default().build(&vs, 1e-3).unwrap();
            let out = net.forward_t(
                &Tensor::randn([2, 2, 19, 19], (Kind::Float, Device::Cpu)),
                true,
            );
            optimizer.backward_step(
                &(out.values.square().mean(Kind::Float)
                    + out.policy_logits.square().mean(Kind::Float)),
            );
            for tensor in vs.trainable_variables() {
                assert!(bool::try_from(tensor.grad().isfinite().all()).unwrap());
                assert!(bool::try_from(tensor.isfinite().all()).unwrap());
            }
        }
    }

    #[test]
    #[ignore = "requires CUDA; run before capacity training"]
    fn capacity_cuda_matches_cpu() {
        assert!(tch::Cuda::is_available());
        // The RunPod test wrapper disables TF32 for this CPU/FP32 comparison.
        // Keep initialization reproducible and cover multiple weight samples.
        for seed in (0..32).chain([20260906]) {
            tch::manual_seed(seed);
            for spec in [
                ModelSpec::KataGeluB16C32Value64x2V1,
                ModelSpec::KataGeluB16C32G3Value64x2V1,
                ModelSpec::KataGeluB10C48Value64x2V1,
            ] {
                let cpu_vs = nn::VarStore::new(Device::Cpu);
                let cpu = GomokuModel::new(cpu_vs.root(), &spec);
                let mut gpu_vs = nn::VarStore::new(Device::Cuda(0));
                let gpu = GomokuModel::new(gpu_vs.root(), &spec);
                gpu_vs.copy(&cpu_vs).unwrap();
                let input = Tensor::randn([4, 2, 19, 19], (Kind::Float, Device::Cpu));
                for training in [false, true] {
                    let expected = cpu.forward_t(&input, training);
                    let actual = gpu.forward_t(&input.to_device(Device::Cuda(0)), training);
                    for (head, actual, expected) in [
                        ("value", &actual.values, &expected.values),
                        ("policy", &actual.policy_logits, &expected.policy_logits),
                    ] {
                        let actual = actual.to_device(Device::Cpu);
                        let difference = (&actual - expected).abs();
                        let maximum = f64::try_from(difference.max()).unwrap();
                        let scale = f64::try_from(expected.abs().max()).unwrap();
                        eprintln!(
                            "seed={seed} {spec:?} training={training} {head}: max_error={maximum} max_output={scale}"
                        );
                        assert!(
                            actual.allclose(expected, 2e-3, 2e-3, false),
                            "{spec:?} training={training} {head}: max_error={maximum} max_output={scale}"
                        );
                    }
                    if training {
                        (actual.values.square().mean(Kind::Float)
                            + actual.policy_logits.square().mean(Kind::Float))
                        .backward();
                        for tensor in gpu_vs.trainable_variables() {
                            assert!(bool::try_from(tensor.grad().isfinite().all()).unwrap());
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn model_spec_has_stable_serialization() {
        for (spec, expected_json) in [
            (
                ModelSpec::KataGeluB16C32G3Value64x2V1,
                r#"{"architecture":"kata_gelu_b16c32g3_value64x2_v1"}"#,
            ),
            (
                ModelSpec::KataGeluB16C32Value64x2V1,
                r#"{"architecture":"kata_gelu_b16c32_value64x2_v1"}"#,
            ),
            (
                ModelSpec::KataGeluB10C48Value64x2V1,
                r#"{"architecture":"kata_gelu_b10c48_value64x2_v1"}"#,
            ),
            (
                ModelSpec::LegacyResNetV1,
                r#"{"architecture":"legacy_resnet_v1"}"#,
            ),
            (ModelSpec::KataV1, r#"{"architecture":"kata_v1"}"#),
            (ModelSpec::KataPoolV1, r#"{"architecture":"kata_pool_v1"}"#),
            (
                ModelSpec::KataGeluPoolV1,
                r#"{"architecture":"kata_gelu_pool_v1"}"#,
            ),
            (
                ModelSpec::KataPoolValue64V1,
                r#"{"architecture":"kata_pool_value64_v1"}"#,
            ),
            (
                ModelSpec::KataPoolValue64x2V1,
                r#"{"architecture":"kata_pool_value64x2_v1"}"#,
            ),
            (
                ModelSpec::KataGeluPoolValue64V1,
                r#"{"architecture":"kata_gelu_pool_value64_v1"}"#,
            ),
            (
                ModelSpec::KataGeluPoolValue64x2V1,
                r#"{"architecture":"kata_gelu_pool_value64x2_v1"}"#,
            ),
            (ModelSpec::KataGeluV1, r#"{"architecture":"kata_gelu_v1"}"#),
            (
                ModelSpec::KataValue64V1,
                r#"{"architecture":"kata_value64_v1"}"#,
            ),
            (
                ModelSpec::KataValue64x2V1,
                r#"{"architecture":"kata_value64x2_v1"}"#,
            ),
            (
                ModelSpec::KataGeluValue64V1,
                r#"{"architecture":"kata_gelu_value64_v1"}"#,
            ),
            (
                ModelSpec::KataGeluValue64x2V1,
                r#"{"architecture":"kata_gelu_value64x2_v1"}"#,
            ),
        ] {
            let json = serde_json::to_string(&spec).unwrap();
            assert_eq!(json, expected_json);
            assert_eq!(serde_json::from_str::<ModelSpec>(&json).unwrap(), spec);
        }
    }

    #[test]
    fn runtime_model_preserves_network_contract() {
        for spec in [
            ModelSpec::KataGeluB16C32G3Value64x2V1,
            ModelSpec::KataGeluB16C32Value64x2V1,
            ModelSpec::KataGeluB10C48Value64x2V1,
            ModelSpec::LegacyResNetV1,
            ModelSpec::KataPoolV1,
            ModelSpec::KataGeluPoolV1,
            ModelSpec::KataPoolValue64V1,
            ModelSpec::KataPoolValue64x2V1,
            ModelSpec::KataGeluPoolValue64V1,
            ModelSpec::KataGeluPoolValue64x2V1,
            ModelSpec::KataV1,
            ModelSpec::KataGeluV1,
            ModelSpec::KataValue64V1,
            ModelSpec::KataValue64x2V1,
            ModelSpec::KataGeluValue64V1,
            ModelSpec::KataGeluValue64x2V1,
        ] {
            let var_store = nn::VarStore::new(Device::Cpu);
            let network = GomokuModel::new(var_store.root(), &spec);
            let input = Tensor::zeros([2, 2, 19, 19], (Kind::Float, Device::Cpu));

            let output = network.forward_t(&input, false);

            assert_eq!(output.values.size(), [2]);
            assert_eq!(output.policy_logits.size(), [2, 19, 19]);
        }
    }

    #[test]
    fn gelu_changes_predictions_with_identical_weights_and_has_finite_gradients() {
        use tch::nn::OptimizerConfig;

        let relu_vs = tch::nn::VarStore::new(Device::Cpu);
        let relu = GomokuModel::new(relu_vs.root(), &ModelSpec::KataV1);
        // Keep the value comparison away from random tanh saturation and give
        // the hidden activation a known negative input, independent of test RNG.
        tch::no_grad(|| {
            let variables = relu_vs.variables();
            for (name, value) in [
                ("value_head.fc1.weight", 0.0),
                ("value_head.fc1.bias", -0.5),
                ("value_head.fc2.weight", 0.1),
                ("value_head.fc2.bias", 0.0),
            ] {
                let _ = variables[name].shallow_clone().fill_(value);
            }
        });
        let mut gelu_vs = tch::nn::VarStore::new(Device::Cpu);
        let gelu = GomokuModel::new(gelu_vs.root(), &ModelSpec::KataGeluV1);
        gelu_vs.copy(&relu_vs).unwrap();
        assert_eq!(relu_vs.variables().len(), gelu_vs.variables().len());
        for (name, tensor) in relu_vs.variables() {
            assert!(tensor.equal(&gelu_vs.variables()[&name]));
        }
        let input = Tensor::randn([2, 2, 19, 19], (Kind::Float, Device::Cpu));
        let relu_output = relu.forward_t(&input, false);
        let gelu_output = gelu.forward_t(&input, false);
        assert!(
            !relu_output
                .policy_logits
                .allclose(&gelu_output.policy_logits, 1e-5, 1e-6, false)
        );
        assert!(
            !relu_output
                .values
                .allclose(&gelu_output.values, 1e-5, 1e-6, false)
        );
        let mut optimizer = tch::nn::Adam::default().build(&gelu_vs, 1e-3).unwrap();
        let output = gelu.forward_t(&input, true);
        let loss = output.values.square().mean(Kind::Float)
            + output.policy_logits.square().mean(Kind::Float);
        optimizer.backward_step(&loss);
        for tensor in gelu_vs.trainable_variables() {
            assert!(bool::try_from(tensor.isfinite().all()).unwrap());
            assert!(bool::try_from(tensor.grad().isfinite().all()).unwrap());
        }
    }

    #[test]
    fn wider_value_heads_have_expected_capacity_and_finite_gradients() {
        use tch::nn::OptimizerConfig;

        for (spec, parameters, depth) in [
            (ModelSpec::KataV1, 661, 1),
            (ModelSpec::KataGeluV1, 661, 1),
            (ModelSpec::KataValue64V1, 4225, 1),
            (ModelSpec::KataValue64x2V1, 8385, 2),
            (ModelSpec::KataGeluValue64V1, 4225, 1),
            (ModelSpec::KataGeluValue64x2V1, 8385, 2),
        ] {
            let vs = nn::VarStore::new(Device::Cpu);
            let model = GomokuModel::new(vs.root(), &spec);
            let dense = vs
                .variables()
                .into_iter()
                .filter(|(name, _)| name.starts_with("value_head.fc"))
                .collect::<Vec<_>>();
            assert_eq!(dense.len(), 2 * (depth + 1));
            assert_eq!(
                dense.iter().map(|(_, t)| t.numel()).sum::<usize>(),
                parameters
            );
            let input = Tensor::randn([2, 2, 19, 19], (Kind::Float, Device::Cpu));
            let mut optimizer = nn::Adam::default().build(&vs, 1e-3).unwrap();
            let output = model.forward_t(&input, true);
            assert!(bool::try_from(output.values.abs().le(1.0).all()).unwrap());
            optimizer.backward_step(
                &(output.values.square().mean(Kind::Float)
                    + output.policy_logits.square().mean(Kind::Float)),
            );
            for tensor in vs.trainable_variables() {
                assert!(bool::try_from(tensor.grad().isfinite().all()).unwrap());
                assert!(bool::try_from(tensor.isfinite().all()).unwrap());
            }
        }
    }
}
