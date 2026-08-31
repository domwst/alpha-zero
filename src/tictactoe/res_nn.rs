use std::borrow::Borrow;

use tch::{
    Tensor,
    nn::{
        BatchNorm, Conv2D, ConvConfig, ModuleT, Path, SequentialT, batch_norm2d, conv2d, linear,
        seq_t,
    },
};

use crate::alpha_zero::{AlphaZeroNet, NetworkOutput};

const CHANNELS: i64 = 32;

#[derive(Debug)]
pub struct ResBlock {
    conv1: Conv2D,
    bn1: BatchNorm,
    conv2: Conv2D,
    bn2: BatchNorm,
}

impl ResBlock {
    pub fn new<'a, P: Borrow<Path<'a>>>(path: P, channels: i64, pad: i64) -> Self {
        let ker = 2 * pad + 1;
        let path = path.borrow();
        Self {
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
            bn1: batch_norm2d(path / "bn1", channels, Default::default()),
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
            bn2: batch_norm2d(path / "bn2", channels, Default::default()),
        }
    }
}

impl ModuleT for ResBlock {
    fn forward_t(&self, xs: &tch::Tensor, train: bool) -> tch::Tensor {
        let out = self.conv1.forward_t(xs, train);
        let out = self.bn1.forward_t(&out, train).relu();
        let out = self.conv2.forward_t(&out, train);
        let out = self.bn2.forward_t(&out, train);
        (out + xs).relu()
    }
}

fn value_head<'a, P: Borrow<Path<'a>>>(path: P) -> SequentialT {
    let path = path.borrow();
    seq_t()
        .add(conv2d(
            path / "conv",
            CHANNELS,
            1,
            1,
            ConvConfig {
                bias: false,
                ..Default::default()
            },
        ))
        .add(batch_norm2d(path / "bn_conv", 1, Default::default()))
        .add_fn(Tensor::relu)
        .add_fn(|t| t.view([t.size()[0], -1]))
        .add(linear(path / "fc1", 19 * 19, 256, Default::default()))
        .add_fn(Tensor::relu)
        .add(linear(path / "fc2", 256, 1, Default::default()))
        .add_fn(Tensor::tanh)
        .add_fn(|t| t.view([t.size()[0]]))
}

fn policy_head<'a, P: Borrow<Path<'a>>>(path: P) -> SequentialT {
    let path = path.borrow();
    seq_t()
        .add(conv2d(
            path / "conv",
            CHANNELS,
            2,
            1,
            ConvConfig {
                bias: false,
                ..Default::default()
            },
        ))
        .add(batch_norm2d(path / "bn_conv", 2, Default::default()))
        .add_fn(Tensor::relu)
        .add_fn(|t| t.view([t.size()[0], -1]))
        .add(linear(
            path / "fc",
            2 * 19 * 19,
            19 * 19,
            Default::default(),
        ))
        .add_fn(|t| t.view([t.size()[0], 19, 19]))
}

#[derive(Debug)]
pub struct TicTacToeResNet {
    conv1: Conv2D,
    bn1: BatchNorm,
    blocks: SequentialT,
    value_head: SequentialT,
    policy_head: SequentialT,
}

impl TicTacToeResNet {
    pub fn new<'a, P: Borrow<Path<'a>>>(path: P) -> Self {
        const BLOCKS: usize = 10;
        let path = path.borrow();
        let mut blocks = seq_t();
        for i in 0..BLOCKS {
            blocks = blocks.add(ResBlock::new(path / format!("res_block{i}"), CHANNELS, 1));
        }
        Self {
            conv1: conv2d(
                path / "conv1",
                2,
                CHANNELS,
                3,
                ConvConfig {
                    padding: 1,
                    bias: false,
                    ..Default::default()
                },
            ),
            bn1: batch_norm2d(path / "bn1", CHANNELS, Default::default()),
            blocks,
            value_head: value_head(path / "value_head"),
            policy_head: policy_head(path / "policy_head"),
        }
    }
}

impl AlphaZeroNet for TicTacToeResNet {
    fn forward_t(&self, xs: &tch::Tensor, train: bool) -> NetworkOutput {
        let out = self.conv1.forward_t(xs, train);
        let out = self.bn1.forward_t(&out, train).relu();
        let out = self.blocks.forward_t(&out, train);

        NetworkOutput {
            values: self.value_head.forward_t(&out, train),
            policy_logits: self.policy_head.forward_t(&out, train),
        }
    }
}

#[cfg(test)]
mod tests {
    use tch::{Device, Kind, Tensor, nn};

    use crate::alpha_zero::AlphaZeroNet;

    use super::TicTacToeResNet;

    #[test]
    fn resnet_preserves_alpha_zero_output_shapes() {
        let var_store = nn::VarStore::new(Device::Cpu);
        let network = TicTacToeResNet::new(var_store.root());
        let input = Tensor::zeros([2, 2, 19, 19], (Kind::Float, Device::Cpu));

        let output = network.forward_t(&input, true);

        assert_eq!(output.values.size(), [2]);
        assert_eq!(output.policy_logits.size(), [2, 19, 19]);
    }
}
