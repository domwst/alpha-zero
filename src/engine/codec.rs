use anyhow::Result;
use tch::Tensor;

use super::Game;

pub trait PositionCodec<TGame: Game> {
    fn encode_position(state: &TGame) -> Tensor;

    /// Encodes a CPU Boolean mask with the same unbatched shape as the policy logits.
    fn encode_policy_mask(state: &TGame, moves: &[TGame::Move]) -> Result<Tensor>;

    /// Selects normalized policy probabilities in the supplied legal-move order.
    fn decode_policy(policy: &Tensor, moves: &[TGame::Move]) -> Result<Vec<f32>>;

    /// Number of spatial symmetries supported during inference. Symmetry zero must
    /// be the canonical, untransformed representation.
    fn inference_symmetry_count() -> usize {
        1
    }

    fn encode_position_with_symmetry(state: &TGame, symmetry: usize) -> Tensor {
        assert_eq!(symmetry, 0);
        Self::encode_position(state)
    }

    /// Encodes the legal moves in the same transformed coordinates as the input.
    fn encode_policy_mask_with_symmetry(
        state: &TGame,
        moves: &[TGame::Move],
        symmetry: usize,
    ) -> Result<Tensor> {
        assert_eq!(symmetry, 0);
        Self::encode_policy_mask(state, moves)
    }

    /// Reads a transformed network policy back in the original legal-move order.
    fn decode_policy_with_symmetry(
        policy: &Tensor,
        moves: &[TGame::Move],
        symmetry: usize,
    ) -> Result<Vec<f32>> {
        assert_eq!(symmetry, 0);
        Self::decode_policy(policy, moves)
    }
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
