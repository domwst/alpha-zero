use tch::Tensor;

use crate::alpha_zero::{AlphaZeroAdapter, Game};

use super::{BoardState, CellState, TicTacToeConvNet, TicTacToeMove, TicTacToeResNet};

pub struct TicTacToeConvAlphaZeroAdapter;

pub struct TicTacToeResAlphaZeroAdapter;

fn game_to_nn_input(state: &BoardState) -> Tensor {
    let mut fld = [[[0; 19]; 19]; 2];
    for i in 0..19 {
        for j in 0..19 {
            let l = match state[(i, j)] {
                CellState::X => 0,
                CellState::O => 1,
                CellState::Empty => continue,
            };
            fld[l][i][j] = 1;
        }
    }
    Tensor::from_slice(fld.as_flattened().as_flattened()).view([2, 19, 19])
}

fn estimated_policy(policy: &Tensor, moves: &[<BoardState as Game>::Move]) -> Vec<f32> {
    let policy = <Vec<f32>>::try_from(policy.view([-1])).unwrap();
    assert_eq!(policy.len(), 19 * 19);
    assert!(!moves.is_empty());

    let mut res = Vec::with_capacity(moves.len());
    for &TicTacToeMove(i, j) in moves {
        res.push(policy[i * 19 + j]);
    }

    assert!(res
        .iter()
        .all(|value| !value.is_nan() && *value != f32::INFINITY));
    let max = res.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    assert!(
        max.is_finite(),
        "All legal moves have zero network probability"
    );
    for value in &mut res {
        *value = (*value - max).exp();
    }
    let sum = res.iter().sum::<f32>();
    assert!(sum.is_finite() && sum > 0.0);
    for value in &mut res {
        *value /= sum;
    }

    res
}

fn policy_to_nn(policy: &[f32], moves: &[<BoardState as Game>::Move]) -> tch::Tensor {
    assert_eq!(policy.len(), moves.len());
    assert!(policy
        .iter()
        .all(|value| value.is_finite() && *value >= 0.0));
    assert!((policy.iter().sum::<f32>() - 1.0).abs() < 1e-4);
    let mut res = [[0f32; 19]; 19];
    for (&TicTacToeMove(i, j), &pol) in moves.iter().zip(policy) {
        res[i][j] = pol;
    }
    Tensor::from_slice(res.as_flattened()).view([19, 19])
}

fn augment(state: &Tensor, policy: &Tensor, augmentation: usize) -> (Tensor, Tensor) {
    assert!(augmentation < 8);
    let reflect = augmentation >= 4;
    let rotations = (augmentation % 4) as i64;
    let transform = |input: &Tensor, dim: i64| {
        if reflect {
            input.flip([dim])
        } else {
            input.copy()
        }
        .rot90(rotations, [dim, dim + 1])
    };
    (transform(state, 1), transform(policy, 0))
}

impl AlphaZeroAdapter<BoardState, TicTacToeConvNet> for TicTacToeConvAlphaZeroAdapter {
    fn convert_game_to_nn_input(state: &BoardState) -> tch::Tensor {
        game_to_nn_input(state)
    }

    fn get_estimated_policy(policy: &Tensor, moves: &[<BoardState as Game>::Move]) -> Vec<f32> {
        estimated_policy(policy, moves)
    }

    fn convert_policy_to_nn(policy: &[f32], moves: &[<BoardState as Game>::Move]) -> tch::Tensor {
        policy_to_nn(policy, moves)
    }

    fn augmentation_count() -> usize {
        8
    }

    fn augment(state: &Tensor, policy: &Tensor, augmentation: usize) -> (Tensor, Tensor) {
        augment(state, policy, augmentation)
    }
}

impl AlphaZeroAdapter<BoardState, TicTacToeResNet> for TicTacToeResAlphaZeroAdapter {
    fn convert_game_to_nn_input(state: &BoardState) -> tch::Tensor {
        game_to_nn_input(state)
    }

    fn get_estimated_policy(policy: &Tensor, moves: &[<BoardState as Game>::Move]) -> Vec<f32> {
        estimated_policy(policy, moves)
    }

    fn convert_policy_to_nn(policy: &[f32], moves: &[<BoardState as Game>::Move]) -> tch::Tensor {
        policy_to_nn(policy, moves)
    }

    fn augmentation_count() -> usize {
        8
    }

    fn augment(state: &Tensor, policy: &Tensor, augmentation: usize) -> (Tensor, Tensor) {
        augment(state, policy, augmentation)
    }
}

#[cfg(test)]
mod tests {
    use tch::{Device, IndexOp, Kind, Tensor};

    use crate::tictactoe::{
        alpha_zero_adapter::{estimated_policy, game_to_nn_input},
        BoardState, CellState, TicTacToeMove,
    };

    #[test]
    fn convert_board_to_tensor() {
        let mut game = BoardState::new();
        game.set_inplace((10, 0), CellState::O);
        game.set_inplace((1, 3), CellState::X);

        let tensor = game_to_nn_input(&game);
        assert_eq!(tensor.size(), [2, 19, 19]);

        let ones = [(1, 10, 0), (0, 1, 3)];

        for i in 0..2 {
            for j in 0..19 {
                for k in 0..19 {
                    let goal = if ones.contains(&(i, j, k)) { 1. } else { 0. };
                    assert_eq!(f32::try_from(tensor.i((i, j, k))).unwrap(), goal);
                }
            }
        }
    }

    #[test]
    fn legal_policy_normalization_is_stable() {
        let policy = Tensor::full([19, 19], -1000.0, (Kind::Float, Device::Cpu));
        let _ = policy.i((0, 0)).fill_(0.0);
        let policy = estimated_policy(&policy, &[TicTacToeMove(0, 1), TicTacToeMove(0, 2)]);

        assert!((policy[0] - 0.5).abs() < 1e-6);
        assert!((policy[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn augmentations_keep_state_and_policy_aligned() {
        let state = Tensor::zeros([2, 19, 19], (Kind::Float, Device::Cpu));
        let policy = Tensor::zeros([19, 19], (Kind::Float, Device::Cpu));
        let _ = state.i((0, 2, 3)).fill_(1.0);
        let _ = policy.i((2, 3)).fill_(1.0);

        for augmentation in 0..8 {
            let (state, policy) = super::augment(&state, &policy, augmentation);
            assert!(state.i(0).equal(&policy));
        }
    }
}
