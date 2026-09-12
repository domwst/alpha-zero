use std::borrow::Borrow;

use tch::{
    Tensor,
    nn::{
        BatchNorm, Conv2D, ConvConfig, ModuleT, Path, SequentialT, batch_norm2d, conv2d, linear,
        seq_t,
    },
};

use crate::engine::{AlphaZeroNet, NetworkOutput};

#[derive(Clone, Copy, Debug)]
pub(super) enum Activation {
    Relu,
    Gelu,
}

impl Activation {
    fn forward(self, xs: &Tensor) -> Tensor {
        match self {
            Self::Relu => xs.relu(),
            Self::Gelu => xs.gelu("none"),
        }
    }
}

#[derive(Debug)]
struct ResBlock {
    activation: Activation,
    bn1: BatchNorm,
    conv1: Conv2D,
    bn2: BatchNorm,
    conv2: Conv2D,
}

impl ResBlock {
    fn new<'a, P: Borrow<Path<'a>>>(
        path: P,
        channels: i64,
        pad: i64,
        activation: Activation,
    ) -> Self {
        let ker = 2 * pad + 1;
        let path = path.borrow();
        Self {
            activation,
            bn1: batch_norm2d(path / "bn1", channels, Default::default()),
            conv1: conv2d(
                path / "conv1",
                channels,
                channels,
                ker,
                ConvConfig {
                    padding: pad,
                    bias: false,
                    ..Default::default()
                },
            ),
            bn2: batch_norm2d(path / "bn2", channels, Default::default()),
            conv2: conv2d(
                path / "conv2",
                channels,
                channels,
                ker,
                ConvConfig {
                    padding: pad,
                    bias: false,
                    ..Default::default()
                },
            ),
        }
    }
}

impl ModuleT for ResBlock {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let out = self.activation.forward(&self.bn1.forward_t(xs, train));
        let out = self.conv1.forward_t(&out, train);

        let out = self.activation.forward(&self.bn2.forward_t(&out, train));
        let out = self.conv2.forward_t(&out, train);

        out + xs
    }
}

// https://github.com/lightvector/KataGo/blob/v1.18.2/python/katago/train/model_pytorch.py#L545
#[derive(Debug)]
pub struct GlobalBlock {
    activation: Activation,
    bn1: BatchNorm,
    bn2: BatchNorm,
    ctoc: Conv2D,
    ctog: Conv2D,
    gtoc: Conv2D,
}

impl GlobalBlock {
    pub fn new<'a, P: Borrow<Path<'a>>>(path: P, channels: i64, global_channels: i64) -> Self {
        Self::with_activation(path, channels, global_channels, Activation::Relu)
    }

    fn with_activation<'a, P: Borrow<Path<'a>>>(
        path: P,
        channels: i64,
        global_channels: i64,
        activation: Activation,
    ) -> Self {
        let path = path.borrow();

        Self {
            activation,
            bn1: batch_norm2d(path / "bn1", channels, Default::default()),
            bn2: batch_norm2d(path / "bn2", channels, Default::default()),
            ctog: conv2d(
                path / "ctog",
                channels,
                global_channels,
                3,
                ConvConfig {
                    padding: 1,
                    bias: false,
                    ..Default::default()
                },
            ),
            gtoc: conv2d(
                path / "gtoc",
                global_channels * 2,
                channels,
                1,
                ConvConfig {
                    bias: false,
                    ..Default::default()
                },
            ),
            ctoc: conv2d(
                path / "ctoc",
                channels,
                channels,
                3,
                ConvConfig {
                    padding: 1,
                    bias: false,
                    ..Default::default()
                },
            ),
        }
    }
}

const CONV_DIMS: &[i64] = &[2, 3];

impl ModuleT for GlobalBlock {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let out = self.activation.forward(&self.bn1.forward_t(xs, train));

        // Poor man's attention
        let gs = self.activation.forward(&self.ctog.forward_t(&out, train));
        let mean = gs.mean_dim(CONV_DIMS, true, None);
        let max = gs.amax(CONV_DIMS, true);
        let g = Tensor::cat(&[&mean, &max], 1);
        let g = self.gtoc.forward_t(&g, train);

        let out = out + g;
        let out = self.activation.forward(&self.bn2.forward_t(&out, train));
        let out = self.ctoc.forward_t(&out, train);

        xs + out
    }
}

/// KataGo's ordinary global-pooling residual topology, with this project's
/// BatchNorm and initialization. Every spatial cell is on-board (no padding mask).
/// Reference: KataGPool, KataConvAndGPool and ResBlock in KataGo v1.18.2.
#[derive(Debug)]
struct KataGlobalBlock {
    activation: Activation,
    bn1: BatchNorm,
    local: Conv2D,
    global: Conv2D,
    global_bn: BatchNorm,
    project: tch::nn::Linear,
    bn2: BatchNorm,
    conv2: Conv2D,
}

fn kata_pool(xs: &Tensor) -> Tensor {
    let shape = xs.size();
    let mean = xs.mean_dim(CONV_DIMS, false, tch::Kind::Float);
    let scaled_mean = &mean * (((shape[2] * shape[3]) as f64).sqrt() - 14.0) / 10.0;
    // max_dim follows upstream's single-argmax gradient at ties.
    let max = xs.flatten(2, 3).max_dim(2, false).0;
    Tensor::cat(&[mean, scaled_mean, max], 1)
}

impl KataGlobalBlock {
    fn new(p: Path<'_>, channels: i64, global_channels: i64, activation: Activation) -> Self {
        // KataGo partitions the intermediate width between local and global paths.
        let local_channels = channels - global_channels;
        assert!(local_channels > 0 && global_channels > 0);
        let config = ConvConfig {
            padding: 1,
            bias: false,
            ..Default::default()
        };
        Self {
            activation,
            bn1: batch_norm2d(&p / "bn1", channels, Default::default()),
            local: conv2d(&p / "local", channels, local_channels, 3, config),
            global: conv2d(&p / "global", channels, global_channels, 3, config),
            global_bn: batch_norm2d(&p / "global_bn", global_channels, Default::default()),
            project: linear(
                &p / "project",
                global_channels * 3,
                local_channels,
                tch::nn::LinearConfig {
                    bias: false,
                    ..Default::default()
                },
            ),
            bn2: batch_norm2d(&p / "bn2", local_channels, Default::default()),
            conv2: conv2d(&p / "conv2", local_channels, channels, 3, config),
        }
    }
}

impl ModuleT for KataGlobalBlock {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let input = self.activation.forward(&self.bn1.forward_t(xs, train));
        let local = self.local.forward_t(&input, train);
        let global = self.global.forward_t(&input, train);
        let global = self
            .activation
            .forward(&self.global_bn.forward_t(&global, train));
        let bias = self
            .project
            .forward_t(&kata_pool(&global), train)
            .unsqueeze(-1)
            .unsqueeze(-1);
        let merged = self
            .activation
            .forward(&self.bn2.forward_t(&(local + bias), train));
        xs + self.conv2.forward_t(&merged, train)
    }
}

const CHANNELS: i64 = 32;

fn value_head<'a, P: Borrow<Path<'a>>>(
    path: P,
    activation: Activation,
    hidden_dims: &[i64],
    trunk_channels: i64,
) -> SequentialT {
    let p = path.borrow();

    let mut head = seq_t()
        .add(batch_norm2d(p / "bn", trunk_channels, Default::default()))
        .add_fn(move |xs| activation.forward(xs))
        .add(conv2d(
            p / "conv",
            trunk_channels,
            CHANNELS,
            1,
            ConvConfig {
                bias: false,
                ..Default::default()
            },
        ))
        .add_fn(|xs| {
            let max = xs.amax(CONV_DIMS, false);
            let avg = xs.mean_dim(CONV_DIMS, false, None);

            Tensor::cat(&[&max, &avg], 1)
        });
    let mut input_dim = CHANNELS * 2;
    for (i, &hidden_dim) in hidden_dims.iter().enumerate() {
        head = head
            .add(linear(
                p / format!("fc{}", i + 1),
                input_dim,
                hidden_dim,
                Default::default(),
            ))
            .add_fn(move |xs| activation.forward(xs));
        input_dim = hidden_dim;
    }
    head.add(linear(
        p / format!("fc{}", hidden_dims.len() + 1),
        input_dim,
        1,
        Default::default(),
    ))
    .add_fn(Tensor::tanh)
    .add_fn(|t| t.view([t.size()[0]]))
}

fn policy_head<'a, P: Borrow<Path<'a>>>(
    path: P,
    activation: Activation,
    trunk_channels: i64,
) -> SequentialT {
    const HIDDEN_DIM: i64 = 10;

    let path = path.borrow();

    seq_t()
        .add(batch_norm2d(
            path / "bn1",
            trunk_channels,
            Default::default(),
        ))
        .add_fn(move |xs| activation.forward(xs))
        .add(conv2d(
            path / "conv1",
            trunk_channels,
            HIDDEN_DIM,
            1,
            ConvConfig {
                bias: false,
                ..Default::default()
            },
        ))
        .add(batch_norm2d(path / "bn2", HIDDEN_DIM, Default::default()))
        .add_fn(move |xs| activation.forward(xs))
        .add(conv2d(path / "conv2", HIDDEN_DIM, 1, 1, Default::default()))
        .add_fn(|t| {
            let sz = t.size();
            t.view([sz[0], sz[2], sz[3]])
        })
}

#[derive(Debug)]
pub struct GomokuKataNet {
    conv: Conv2D,
    board_mask_conv: Option<Conv2D>,
    blocks: SequentialT,
    value_head: SequentialT,
    policy_head: SequentialT,
}

impl GomokuKataNet {
    /// An implicit third, all-ones input plane, zero-padded by its convolution.
    /// Register it last so the same seed preserves every baseline initial tensor.
    /// Splitting the input convolution also keeps replay caches two-channel.
    pub(super) fn with_board_mask<'a, P: Borrow<Path<'a>>>(path: P) -> Self {
        let p = path.borrow();
        let mut network = Self::with_value_head(p, Activation::Gelu, &[64, 64]);
        network.board_mask_conv = Some(conv2d(
            p / "board_mask_conv",
            1,
            CHANNELS,
            3,
            ConvConfig {
                padding: 1,
                bias: false,
                ..Default::default()
            },
        ));
        network
    }

    pub fn new<'a, P: Borrow<Path<'a>>>(path: P) -> Self {
        Self::with_value_head(path, Activation::Relu, &[10])
    }

    /// Kata v1 with exact GELU at every hidden activation, preserving all other choices.
    pub fn new_gelu<'a, P: Borrow<Path<'a>>>(path: P) -> Self {
        Self::with_value_head(path, Activation::Gelu, &[10])
    }

    pub(super) fn with_value_head<'a, P: Borrow<Path<'a>>>(
        path: P,
        activation: Activation,
        hidden_dims: &[i64],
    ) -> Self {
        Self::with_pooling(path, activation, hidden_dims, false)
    }

    pub(super) fn with_pooling<'a, P: Borrow<Path<'a>>>(
        path: P,
        activation: Activation,
        hidden_dims: &[i64],
        kata_pooling: bool,
    ) -> Self {
        Self::with_configuration(
            path,
            activation,
            hidden_dims,
            kata_pooling,
            10,
            CHANNELS,
            &[3, 7],
        )
    }

    /// Scale only the selected architecture's trunk, preserving both head widths.
    pub(super) fn with_trunk<'a, P: Borrow<Path<'a>>>(
        path: P,
        blocks: usize,
        channels: i64,
    ) -> Self {
        Self::with_configuration(
            path,
            Activation::Gelu,
            &[64, 64],
            false,
            blocks,
            channels,
            &[3, 7],
        )
    }

    /// Sixteen blocks with original global pooling at positions 4, 8 and 12.
    pub(super) fn with_third_global<'a, P: Borrow<Path<'a>>>(path: P) -> Self {
        Self::with_configuration(
            path,
            Activation::Gelu,
            &[64, 64],
            false,
            16,
            32,
            &[3, 7, 11],
        )
    }

    fn with_configuration<'a, P: Borrow<Path<'a>>>(
        path: P,
        activation: Activation,
        hidden_dims: &[i64],
        kata_pooling: bool,
        block_count: usize,
        channels: i64,
        global_blocks: &[usize],
    ) -> Self {
        assert!(block_count >= 8 && channels >= 2 && channels % 2 == 0);

        assert!(global_blocks.iter().all(|&i| i < block_count));
        let p = path.borrow();
        let mut blocks = seq_t();
        for i in 0..block_count {
            let is_global = global_blocks.contains(&i);
            if is_global && kata_pooling {
                blocks = blocks.add(KataGlobalBlock::new(
                    p / format!("global_block_{i}"),
                    channels,
                    channels / 2,
                    activation,
                ));
            } else if is_global {
                blocks = blocks.add(GlobalBlock::with_activation(
                    p / format!("global_block_{i}"),
                    channels,
                    channels / 2,
                    activation,
                ));
            } else {
                blocks = blocks.add(ResBlock::new(
                    p / format!("res_block_{i}"),
                    channels,
                    1,
                    activation,
                ));
            }
        }

        Self {
            conv: conv2d(
                p / "conv",
                2,
                channels,
                3,
                ConvConfig {
                    padding: 1,
                    ..Default::default()
                },
            ),
            board_mask_conv: None,
            blocks,
            value_head: value_head(p / "value_head", activation, hidden_dims, channels),
            policy_head: policy_head(p / "policy_head", activation, channels),
        }
    }
}

impl AlphaZeroNet for GomokuKataNet {
    fn forward_t(&self, xs: &Tensor, is_training: bool) -> crate::engine::NetworkOutput {
        let mut out = self.conv.forward_t(xs, is_training);
        if let Some(mask_conv) = &self.board_mask_conv {
            let shape = xs.size();
            let mask = Tensor::ones([1, 1, shape[2], shape[3]], (xs.kind(), xs.device()));
            // Broadcast the same border features over every position in the batch.
            out += mask_conv.forward_t(&mask, is_training);
        }
        let out = self.blocks.forward_t(&out, is_training);

        NetworkOutput {
            values: self.value_head.forward_t(&out, is_training),
            policy_logits: self.policy_head.forward_t(&out, is_training),
        }
    }
}

#[cfg(test)]
mod tests {
    use tch::{Device, Kind, Tensor, nn};

    #[test]
    fn board_mask_exposes_edges_and_matches_three_channel_convolution() {
        use super::*;
        use tch::IndexOp;
        let vs = nn::VarStore::new(Device::Cpu);
        let net = GomokuKataNet::with_board_mask(vs.root());
        let mask_conv = net.board_mask_conv.as_ref().unwrap();
        let mask = Tensor::ones([1, 1, 19, 19], (Kind::Float, Device::Cpu));
        let mut weight = mask_conv.ws.shallow_clone();
        tch::no_grad(|| {
            let _ = weight.fill_(1.0);
        });
        let borders = mask_conv.forward_t(&mask, false);
        assert_eq!(borders.double_value(&[0, 0, 0, 0]), 4.0);
        assert_eq!(borders.double_value(&[0, 0, 0, 9]), 6.0);
        assert_eq!(borders.double_value(&[0, 0, 9, 9]), 9.0);
        let input = Tensor::randn([3, 2, 19, 19], (Kind::Float, Device::Cpu));
        let split = net.conv.forward_t(&input, false) + &borders;
        let combined = Tensor::cat(&[input, mask.expand([3, 1, 19, 19], true)], 1).conv2d(
            &Tensor::cat(&[&net.conv.ws, &weight], 1),
            net.conv.bs.as_ref(),
            1,
            1,
            1,
            1,
        );
        assert!((split - combined).abs().max().double_value(&[]) < 1e-5);
        borders.sum(Kind::Float).backward();
        assert!(weight.grad().i((0, 0, 1, 1)).double_value(&[]) > 0.0);
    }

    use crate::{engine::AlphaZeroNet, gomoku::GomokuKataNet};

    #[test]
    fn kata_pool_statistics_and_argmax_gradient() {
        let input = Tensor::from_slice(&[1f32, 2., 3., 4., -1., -1., -1., -1.])
            .view([1, 2, 2, 2])
            .set_requires_grad(true);
        let pooled = super::kata_pool(&input);
        let expected = Tensor::from_slice(&[2.5f32, -1., -3., 1.2, 4., -1.]).view([1, 6]);
        assert!(pooled.allclose(&expected, 1e-6, 1e-6, false));
        pooled.narrow(1, 4, 2).sum(Kind::Float).backward();
        let expected_grad =
            Tensor::from_slice(&[0f32, 0., 0., 1., 1., 0., 0., 0.]).view([1, 2, 2, 2]);
        assert!(input.grad().equal(&expected_grad));
    }

    #[test]
    fn kata_pooling_variants_have_finite_training_gradients() {
        use tch::nn::OptimizerConfig;
        for activation in [super::Activation::Relu, super::Activation::Gelu] {
            for head in [&[10][..], &[64][..], &[64, 64][..]] {
                let vs = nn::VarStore::new(Device::Cpu);
                let net = GomokuKataNet::with_pooling(vs.root(), activation, head, true);
                let mut optimizer = nn::Adam::default().build(&vs, 1e-3).unwrap();
                let input = Tensor::randn([2, 2, 9, 13], (Kind::Float, Device::Cpu));
                let out = net.forward_t(&input, true);
                assert_eq!(out.policy_logits.size(), [2, 9, 13]);
                assert!(bool::try_from(out.values.abs().le(1.0).all()).unwrap());
                optimizer.backward_step(
                    &(out.values.square().mean(Kind::Float)
                        + out.policy_logits.square().mean(Kind::Float)),
                );
                for tensor in vs.trainable_variables() {
                    assert!(bool::try_from(tensor.isfinite().all()).unwrap());
                    assert!(bool::try_from(tensor.grad().isfinite().all()).unwrap());
                }
            }
        }
    }

    #[test]
    #[ignore = "requires CUDA; run before the pooling experiment starts"]
    fn pooling_cuda_matches_cpu() {
        assert!(tch::Cuda::is_available());
        for activation in [super::Activation::Relu, super::Activation::Gelu] {
            for head in [&[10][..], &[64][..], &[64, 64][..]] {
                let cpu_vs = nn::VarStore::new(Device::Cpu);
                let cpu = GomokuKataNet::with_pooling(cpu_vs.root(), activation, head, true);
                let mut gpu_vs = nn::VarStore::new(Device::Cuda(0));
                let gpu = GomokuKataNet::with_pooling(gpu_vs.root(), activation, head, true);
                gpu_vs.copy(&cpu_vs).unwrap();
                let input = Tensor::randn([2, 2, 19, 19], (Kind::Float, Device::Cpu));
                let expected = cpu.forward_t(&input, false);
                let actual = gpu.forward_t(&input.to_device(Device::Cuda(0)), false);
                assert!(actual.values.to_device(Device::Cpu).allclose(
                    &expected.values,
                    1e-3,
                    1e-3,
                    false
                ));
                assert!(actual.policy_logits.to_device(Device::Cpu).allclose(
                    &expected.policy_logits,
                    1e-3,
                    1e-3,
                    false
                ));
                let out = gpu.forward_t(&input.to_device(Device::Cuda(0)), true);
                (out.values.square().mean(Kind::Float)
                    + out.policy_logits.square().mean(Kind::Float))
                .backward();
                for tensor in gpu_vs.trainable_variables() {
                    assert!(bool::try_from(tensor.grad().isfinite().all()).unwrap());
                }
            }
        }
    }

    #[test]
    fn gomoku_kata_preserves_output_shapes() {
        let vs = nn::VarStore::new(Device::Cpu);
        let net = GomokuKataNet::new(vs.root());

        const DIMS: &[i64] = &[5, 10, 15, 19];
        for &d1 in DIMS {
            for &d2 in DIMS {
                let inp = Tensor::zeros([3, 2, d1, d2], (Kind::Float, Device::Cpu));
                let out = net.forward_t(&inp, true);

                assert_eq!(out.values.size(), [3]);
                assert_eq!(out.policy_logits.size(), [3, d1, d2]);
            }
        }
    }
}
