use tch::{
    nn::{
        self, BatchNorm, Conv2D, ConvConfig, ConvTranspose2D, ConvTransposeConfig, Linear, ModuleT,
    },
    Tensor,
};

use crate::alpha_zero::AlphaZeroNet;

pub struct TicTacToeNet {
    conv1: Conv2D,
    bn_conv2: BatchNorm,
    conv2: Conv2D,
    bn_conv3: BatchNorm,
    conv3: Conv2D,
    bn_fc_mid_1: BatchNorm,
    fc_mid_1: Linear,

    bn_fc_mid_2: BatchNorm,
    fc_mid_2: Linear,
    bn_upconv3: BatchNorm,
    upconv3: ConvTranspose2D,
    bn_upconv2: BatchNorm,
    upconv2: ConvTranspose2D,
    bn_upconv1: BatchNorm,
    upconv1: ConvTranspose2D,

    bn_conv_final: BatchNorm,
    conv_final: Conv2D,

    fc_value_1: Linear,
    bn_fc_value_2: BatchNorm,
    fc_value_2: Linear,
    // bn_fc_value_3: BatchNorm,
    fc_value_3: Linear,
}

impl TicTacToeNet {
    pub fn new(path: &nn::Path) -> Self {
        Self {
            conv1: nn::conv2d(path / "conv1", 2, 10, 4, Default::default()), // 2x19x19 -> 10x16x16
            bn_conv2: nn::batch_norm2d(path / "bn_conv2", 10, Default::default()),
            conv2: nn::conv2d(
                path / "conv2",
                10,
                20,
                5,
                ConvConfig {
                    padding: 2,
                    ..Default::default()
                },
            ), // 10x8x8 -> 20x8x8
            bn_conv3: nn::batch_norm2d(path / "bn_conv3", 20, Default::default()),
            conv3: nn::conv2d(
                path / "conv3",
                20,
                40,
                3,
                ConvConfig {
                    padding: 1,
                    ..Default::default()
                },
            ), // 20x4x4 -> 40x4x4
            bn_fc_mid_1: nn::batch_norm2d(path / "bn_fc_mid_1", 40, Default::default()),
            fc_mid_1: nn::linear(
                path / "fc_mid_1",
                40 * 4 * 4,
                40 * 4 * 4,
                Default::default(),
            ),

            bn_fc_mid_2: nn::batch_norm1d(path / "bn_fc_mid_2", 1, Default::default()),
            fc_mid_2: nn::linear(
                path / "fc_mid_2",
                40 * 4 * 4,
                40 * 4 * 4,
                Default::default(),
            ),
            bn_upconv3: nn::batch_norm2d(path / "bn_upconv3", 40, Default::default()),
            upconv3: nn::conv_transpose2d(
                path / "upconv3",
                80,
                20,
                3,
                ConvTransposeConfig {
                    padding: 1,
                    ..Default::default()
                },
            ), // 80x4x4 -> 20x4x4
            bn_upconv2: nn::batch_norm2d(path / "bn_upconv2", 20, Default::default()),
            upconv2: nn::conv_transpose2d(
                path / "upconv2",
                40,
                10,
                5,
                ConvTransposeConfig {
                    padding: 2,
                    ..Default::default()
                },
            ), // 40x8x8 -> 5x8x8
            bn_upconv1: nn::batch_norm2d(path / "bn_upconv1", 10, Default::default()),
            upconv1: nn::conv_transpose2d(path / "upconv1", 20, 5, 4, Default::default()),

            bn_conv_final: nn::batch_norm2d(path / "bn_conv_final", 5, Default::default()),
            // 20x16x16 -> 5x19x19
            conv_final: nn::conv2d(
                path / "conv_final",
                7,
                1,
                5,
                ConvConfig {
                    padding: 2,
                    ..Default::default()
                },
            ), // 7x19x19 -> 1x19x19

            fc_value_1: nn::linear(path / "fc_value_1", 4 * 4 * 40, 50, Default::default()),
            bn_fc_value_2: nn::batch_norm1d(path / "bn_fc_value_2", 1, Default::default()),
            fc_value_2: nn::linear(path / "fc_value_2", 50, 10, Default::default()),
            // bn_fc_value_3: nn::batch_norm1d(path / "bn_fc_value_3", 1, Default::default()),
            fc_value_3: nn::linear(path / "fc_value_3", 10, 1, Default::default()),
        }
    }
}

impl AlphaZeroNet for TicTacToeNet {
    fn forward_t(&self, xs: &Tensor, is_training: bool) -> (Tensor, Tensor) {
        let layer1 = self.conv1.forward_t(xs, is_training);
        assert_eq!(layer1.size()[1..], [10, 16, 16]);
        let layer1 = layer1.relu();
        let layer1 = self.bn_conv2.forward_t(&layer1, is_training);
        // 10x16x16
        assert_eq!(layer1.size()[1..], [10, 16, 16]);

        let layer2 = self.conv2.forward_t(&layer1, is_training);
        assert_eq!(layer2.size()[1..], [20, 16, 16]);
        let (layer2, indices2) = layer2.max_pool2d_with_indices(2, 2, 0, 1, false);
        assert_eq!(layer2.size()[1..], [20, 8, 8]);
        let layer2 = layer2.relu();
        let layer2 = self.bn_conv3.forward_t(&layer2, is_training);
        // 20x8x8
        assert_eq!(layer2.size()[1..], [20, 8, 8]);

        let layer3 = self.conv3.forward_t(&layer2, is_training);
        assert_eq!(layer3.size()[1..], [40, 8, 8]);
        let (layer3, indices3) = layer3.max_pool2d_with_indices(2, 2, 0, 1, false);
        assert_eq!(layer3.size()[1..], [40, 4, 4]);
        let layer3 = layer3.relu();
        let layer3 = self.bn_fc_mid_1.forward_t(&layer3, is_training);
        // 40x4x4
        assert_eq!(layer3.size()[1..], [40, 4, 4]);

        let mid = self
            .fc_mid_1
            .forward_t(&layer3.view([layer3.size()[0], -1]), is_training);
        assert_eq!(mid.size()[1..], [4 * 4 * 40]);
        let mid = mid.relu();
        let mid = self
            .bn_fc_mid_2
            .forward_t(&mid.view([mid.size()[0], 1, -1]), is_training)
            .view([mid.size()[0], -1]);

        // Value
        let val = self.fc_value_1.forward_t(&mid, is_training);
        let val = self
            .bn_fc_value_2
            .forward_t(&val.view([val.size()[0], 1, -1]), is_training)
            .view([val.size()[0], -1]);
        let val = self.fc_value_2.forward_t(&val, is_training);
        // let val = self.bn_fc_value_3.forward_t(&val, is_training);
        let val = self.fc_value_3.forward_t(&val, is_training);
        let val = val.view([val.size()[0]]);
        let val = val.tanh();

        // Policy
        let policy = self.fc_mid_2.forward_t(&mid, is_training);
        let policy = policy.relu();
        let policy = policy.view([policy.size()[0], 40, 4, 4]);
        let policy = self.bn_upconv3.forward_t(&policy, is_training);
        let policy = Tensor::concat(&[policy, layer3], 1);
        let policy = policy.max_unpool2d(&Tensor::concat(&[&indices3, &indices3], 1), &[8, 8]);
        let policy = self.upconv3.forward_t(&policy, is_training); // 20x8x8
        let policy = policy.relu();

        let policy = self.bn_upconv2.forward_t(&policy, is_training);
        let policy = Tensor::concat(&[policy, layer2], 1);
        let policy = policy.max_unpool2d(&Tensor::concat(&[&indices2, &indices2], 1), &[16, 16]);
        let policy = self.upconv2.forward_t(&policy, is_training); // 10x16x16
        let policy = policy.relu();

        let policy = self.bn_upconv1.forward_t(&policy, is_training);
        let policy = Tensor::concat(&[policy, layer1], 1);
        let policy = self.upconv1.forward_t(&policy, is_training); // 5x19x19
        let policy = policy.relu();

        let policy = self.bn_conv_final.forward_t(&policy, is_training);
        let policy = Tensor::concat(&[&policy, xs], 1);
        let policy = self.conv_final.forward_t(&policy, is_training);
        assert_eq!(policy.size()[1..], [1, 19, 19]);
        let policy = policy
            .view([policy.size()[0], -1])
            .log_softmax(1, None)
            .view([policy.size()[0], 19, 19]);

        (val, policy)
    }
}
