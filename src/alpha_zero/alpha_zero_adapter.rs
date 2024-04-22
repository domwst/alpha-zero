use tch::{Device, Kind, Tensor};

use super::{AlphaZeroNet, Game};

pub trait AlphaZeroAdapter<TGame: Game, Net: AlphaZeroNet> {
    fn convert_game_to_nn_input(state: &TGame, options: (Kind, Device)) -> Tensor;
    fn get_estimated_policy(policy: Tensor, moves: &[TGame::Move]) -> Vec<f32>;

    fn convert_policy_to_nn(
        policy: &[f32],
        moves: &[TGame::Move],
        options: (Kind, Device),
    ) -> Tensor;
}
