use std::borrow::Borrow;

use tch::nn::{batch_norm2d, conv2d, BatchNorm, Conv2D, ConvConfig, ModuleT, Path};

#[derive(Debug)]
pub struct ResBlock {
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
                    ..Default::default()
                },
            ),
        }
    }
}

impl ModuleT for ResBlock {
    fn forward_t(&self, xs: &tch::Tensor, train: bool) -> tch::Tensor {
        let out = self.conv1.forward_t(&self.bn1.forward_t(xs, train), train);
        let out = out.relu();
        let out = self
            .conv2
            .forward_t(&self.bn2.forward_t(&out, train), train);
        let out = out + xs;
        out.relu()
    }
}
