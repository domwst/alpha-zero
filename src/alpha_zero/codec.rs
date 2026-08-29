use anyhow::Result;
use tch::Tensor;

use super::Game;

pub trait PositionCodec<TGame: Game> {
    fn encode_position(state: &TGame) -> Tensor;
    fn decode_policy(policy: &Tensor, moves: &[TGame::Move]) -> Result<Vec<f32>>;
}

pub trait TrainingCodec<TGame: Game>: PositionCodec<TGame> {
    type Policy;

    fn encode_policy_target(state: &TGame, policy: &[f32]) -> Result<Self::Policy>;
    fn policy_to_tensor(policy: &Self::Policy) -> Tensor;

    fn augmentation_count() -> usize {
        1
    }

    fn augment(state: &Tensor, policy: &Tensor, augmentation: usize) -> (Tensor, Tensor) {
        assert_eq!(augmentation, 0);
        (state.copy(), policy.copy())
    }
}
