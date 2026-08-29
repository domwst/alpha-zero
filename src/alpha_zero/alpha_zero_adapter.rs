use tch::Tensor;

use super::{AlphaZeroNet, Game};

pub trait AlphaZeroAdapter<TGame: Game, Net: AlphaZeroNet> {
    // TODO: Are there games where augmentations count depends on a position?
    fn augmentation_count() -> usize {
        1
    }

    fn augment(state: &Tensor, policy: &Tensor, augmentation: usize) -> (Tensor, Tensor) {
        assert_eq!(augmentation, 0);
        (state.copy(), policy.copy())
    }

    fn convert_game_to_nn_input(state: &TGame) -> Tensor;
    fn get_estimated_policy(policy: &Tensor, moves: &[TGame::Move]) -> Vec<f32>;

    fn convert_policy_to_nn(policy: &[f32], moves: &[TGame::Move]) -> Tensor;
}
