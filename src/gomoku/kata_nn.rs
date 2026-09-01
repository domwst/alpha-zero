use std::borrow::Borrow;

use tch::{
    Tensor,
    nn::{
        BatchNorm, Conv2D, ConvConfig, ModuleT, Path, SequentialT, batch_norm2d, conv2d, linear,
        seq_t,
    },
};

use crate::engine::{AlphaZeroNet, NetworkOutput};

#[derive(Debug)]
struct ResBlock {
    bn1: BatchNorm,
    conv1: Conv2D,
    bn2: BatchNorm,
    conv2: Conv2D,
}

impl ResBlock {
    pub fn new<'a, P: Borrow<Path<'a>>>(path: P, channels: i64, pad: i64) -> Self {
        let ker = 2 * pad + 1;
        let path = path.borrow();
        Self {
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
        let out = self.bn1.forward_t(xs, train).relu();
        let out = self.conv1.forward_t(&out, train);

        let out = self.bn2.forward_t(&out, train).relu();
        let out = self.conv2.forward_t(&out, train);

        out + xs
    }
}

// https://github.com/lightvector/KataGo/blob/v1.18.2/python/katago/train/model_pytorch.py#L545
#[derive(Debug)]
pub struct GlobalBlock {
    bn1: BatchNorm,
    bn2: BatchNorm,
    ctoc: Conv2D,
    ctog: Conv2D,
    gtoc: Conv2D,
}

impl GlobalBlock {
    pub fn new<'a, P: Borrow<Path<'a>>>(path: P, channels: i64, global_channels: i64) -> Self {
        let path = path.borrow();

        Self {
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
        let out = self.bn1.forward_t(xs, train).relu();

        // Poor man's attention
        let gs = self.ctog.forward_t(&out, train).relu();
        let mean = gs.mean_dim(CONV_DIMS, true, None);
        let max = gs.amax(CONV_DIMS, true);
        let g = Tensor::cat(&[&mean, &max], 1);
        let g = self.gtoc.forward_t(&g, train);

        let out = out + g;
        let out = self.bn2.forward_t(&out, train).relu();
        let out = self.ctoc.forward_t(&out, train);

        xs + out
    }
}

const CHANNELS: i64 = 32;

fn value_head<'a, P: Borrow<Path<'a>>>(path: P) -> SequentialT {
    const HIDDEN_DIM: i64 = 10;

    let p = path.borrow();

    seq_t()
        .add(batch_norm2d(p / "bn", CHANNELS, Default::default()))
        .add_fn(Tensor::relu)
        .add(conv2d(
            p / "conv",
            CHANNELS,
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
        })
        .add(linear(
            p / "fc1",
            CHANNELS * 2,
            HIDDEN_DIM,
            Default::default(),
        ))
        .add_fn(Tensor::relu)
        .add(linear(p / "fc2", HIDDEN_DIM, 1, Default::default()))
        .add_fn(Tensor::tanh)
        .add_fn(|t| t.view([t.size()[0]]))
}

fn policy_head<'a, P: Borrow<Path<'a>>>(path: P) -> SequentialT {
    const HIDDEN_DIM: i64 = 10;

    let path = path.borrow();

    seq_t()
        .add(batch_norm2d(path / "bn1", CHANNELS, Default::default()))
        .add_fn(Tensor::relu)
        .add(conv2d(
            path / "conv1",
            CHANNELS,
            HIDDEN_DIM,
            1,
            ConvConfig {
                bias: false,
                ..Default::default()
            },
        ))
        .add(batch_norm2d(path / "bn2", HIDDEN_DIM, Default::default()))
        .add_fn(Tensor::relu)
        .add(conv2d(path / "conv2", HIDDEN_DIM, 1, 1, Default::default()))
        .add_fn(|t| {
            let sz = t.size();
            t.view([sz[0], sz[2], sz[3]])
        })
}

#[derive(Debug)]
pub struct GomokuKataNet {
    conv: Conv2D,
    blocks: SequentialT,
    value_head: SequentialT,
    policy_head: SequentialT,
}

impl GomokuKataNet {
    pub fn new<'a, P: Borrow<Path<'a>>>(path: P) -> Self {
        const BLOCKS: usize = 10;
        const IS_GLOBAL: [bool; BLOCKS] = [
            false, false, false, true, false, false, false, true, false, false,
        ];

        let p = path.borrow();
        let mut blocks = seq_t();
        for (i, &is_global) in IS_GLOBAL.iter().enumerate() {
            if is_global {
                blocks = blocks.add(GlobalBlock::new(
                    p / format!("global_block_{i}"),
                    CHANNELS,
                    CHANNELS / 2,
                ));
            } else {
                blocks = blocks.add(ResBlock::new(p / format!("res_block_{i}"), CHANNELS, 1));
            }
        }

        Self {
            conv: conv2d(
                p / "conv",
                2,
                CHANNELS,
                3,
                ConvConfig {
                    padding: 1,
                    ..Default::default()
                },
            ),
            blocks,
            value_head: value_head(p / "value_head"),
            policy_head: policy_head(p / "policy_head"),
        }
    }
}

impl AlphaZeroNet for GomokuKataNet {
    fn forward_t(&self, xs: &Tensor, is_training: bool) -> crate::engine::NetworkOutput {
        let out = self.conv.forward_t(xs, is_training);
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

    use crate::{engine::AlphaZeroNet, gomoku::GomokuKataNet};

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
