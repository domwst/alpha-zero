use std::borrow::Borrow;

use tch::{
    nn::{
        batch_norm1d, batch_norm2d, conv2d, linear, seq_t, Conv2D, ConvConfig, ModuleT, Path,
        SequentialT,
    },
    Tensor,
};

use crate::{alpha_zero::AlphaZeroNet, util::ResBlock};

const CHANNELS: i64 = 32;

fn value_head<'a, P: Borrow<Path<'a>>>(path: P) -> SequentialT {
    let path = path.borrow();
    seq_t()
        .add(batch_norm2d(path / "bn_conv", CHANNELS, Default::default()))
        .add(conv2d(path / "conv", CHANNELS, 1, 1, Default::default()))
        .add_fn(Tensor::relu)
        .add_fn(|t| t.view([t.size()[0], 1, -1]))
        .add(batch_norm1d(path / "bn_fc", 1, Default::default()))
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
        .add(batch_norm2d(path / "bn_conv", CHANNELS, Default::default()))
        .add(conv2d(path / "conv", CHANNELS, 2, 1, Default::default()))
        .add_fn(Tensor::relu)
        .add_fn(|t| t.view([t.size()[0], 1, -1]))
        .add(batch_norm1d(path / "bn_fc", 1, Default::default()))
        .add_fn(|t| t.view([t.size()[0], -1]))
        .add(linear(
            path / "fc",
            2 * 19 * 19,
            19 * 19,
            Default::default(),
        ))
        .add_fn(|t| t.log_softmax(1, None))
        .add_fn(|t| t.view([t.size()[0], 19, 19]))
}

#[derive(Debug)]
pub struct TicTacToeResNet {
    conv1: Conv2D,
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
                    ..Default::default()
                },
            ),
            blocks,
            value_head: value_head(path / "value_head"),
            policy_head: policy_head(path / "policy_head"),
        }
    }
}

impl AlphaZeroNet for TicTacToeResNet {
    fn forward_t(&self, xs: &tch::Tensor, train: bool) -> (Tensor, Tensor) {
        let out = self.conv1.forward_t(xs, train).relu();
        let out = self.blocks.forward_t(&out, train);

        (
            self.value_head.forward_t(&out, train),
            self.policy_head.forward_t(&out, train),
        )
    }
}
