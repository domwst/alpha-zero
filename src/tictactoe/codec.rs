use std::ops::Index;

use anyhow::{bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};
use tch::Tensor;

use crate::alpha_zero::{Game, PositionCodec, TerminationState, TrainingCodec};

use super::{BoardState, CellState, TicTacToeMove};

pub const ACTION_SCHEMA: &str = "board19-row-major-v1";

pub struct TicTacToeCodec;

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct TicTacToePolicy {
    values: [[f32; BoardState::N]; BoardState::N],
}

impl TicTacToePolicy {
    pub fn one_hot(r#move: TicTacToeMove) -> Self {
        let (row, column) = r#move.to_xy();
        let mut values = [[0.0; BoardState::N]; BoardState::N];
        values[row][column] = 1.0;
        Self { values }
    }

    pub fn as_flattened(&self) -> &[f32] {
        self.values.as_flattened()
    }

    pub fn validate_for(&self, state: &BoardState) -> Result<()> {
        ensure!(
            state.is_win() == CellState::Empty,
            "terminal state cannot have a policy target"
        );
        ensure!(
            self.as_flattened()
                .iter()
                .all(|value| value.is_finite() && *value >= 0.0),
            "canonical policy contains invalid probabilities"
        );
        ensure!(
            (self.as_flattened().iter().sum::<f32>() - 1.0).abs() < 1e-4,
            "canonical policy is not normalized"
        );
        let mut has_legal_move = false;
        for row in 0..BoardState::N {
            for column in 0..BoardState::N {
                if state[(row, column)] == CellState::Empty {
                    has_legal_move = true;
                } else {
                    ensure!(
                        self[(row, column)] == 0.0,
                        "canonical policy assigns probability to an occupied cell"
                    );
                }
            }
        }
        ensure!(has_legal_move, "terminal state cannot have a policy target");
        Ok(())
    }
}

impl Index<(usize, usize)> for TicTacToePolicy {
    type Output = f32;

    fn index(&self, (row, column): (usize, usize)) -> &Self::Output {
        &self.values[row][column]
    }
}

fn game_to_nn_input(state: &BoardState) -> Tensor {
    let mut field = [[[0; BoardState::N]; BoardState::N]; 2];
    for row in 0..BoardState::N {
        for column in 0..BoardState::N {
            let plane = match state[(row, column)] {
                CellState::X => 0,
                CellState::O => 1,
                CellState::Empty => continue,
            };
            field[plane][row][column] = 1;
        }
    }
    Tensor::from_slice(field.as_flattened().as_flattened()).view([
        2,
        BoardState::N as i64,
        BoardState::N as i64,
    ])
}

fn estimated_policy(policy: &Tensor, moves: &[TicTacToeMove]) -> Result<Vec<f32>> {
    let policy = <Vec<f32>>::try_from(policy.view([-1]))
        .context("converting canonical network policy to host values")?;
    ensure!(
        policy.len() == BoardState::N * BoardState::N,
        "expected {} policy values, got {}",
        BoardState::N * BoardState::N,
        policy.len()
    );
    ensure!(
        !moves.is_empty(),
        "cannot decode policy without legal moves"
    );

    let mut result = Vec::with_capacity(moves.len());
    for r#move in moves {
        let (row, column) = r#move.to_xy();
        result.push(policy[row * BoardState::N + column]);
    }

    ensure!(
        result
            .iter()
            .all(|value| !value.is_nan() && *value != f32::INFINITY),
        "network returned an invalid policy score"
    );
    let max = result.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    ensure!(
        max.is_finite(),
        "all legal moves have zero network probability"
    );
    for value in &mut result {
        *value = (*value - max).exp();
    }
    let sum = result.iter().sum::<f32>();
    ensure!(
        sum.is_finite() && sum > 0.0,
        "legal policy cannot be normalized"
    );
    for value in &mut result {
        *value /= sum;
    }

    Ok(result)
}

fn canonical_policy(policy: &[f32], moves: &[TicTacToeMove]) -> Result<TicTacToePolicy> {
    ensure!(
        policy.len() == moves.len(),
        "policy has {} entries for {} legal moves",
        policy.len(),
        moves.len()
    );
    ensure!(
        policy
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0),
        "policy contains invalid probabilities"
    );
    ensure!(
        (policy.iter().sum::<f32>() - 1.0).abs() < 1e-4,
        "policy is not normalized"
    );

    let mut values = [[0.0; BoardState::N]; BoardState::N];
    for (r#move, &probability) in moves.iter().zip(policy) {
        let (row, column) = r#move.to_xy();
        values[row][column] = probability;
    }
    let policy = TicTacToePolicy { values };
    Ok(policy)
}

fn augment(state: &Tensor, policy: &Tensor, augmentation: usize) -> (Tensor, Tensor) {
    assert!(augmentation < 8);
    let reflect = augmentation >= 4;
    let rotations = (augmentation % 4) as i64;
    let transform = |input: &Tensor, dimension: i64| {
        if reflect {
            input.flip([dimension])
        } else {
            input.copy()
        }
        .rot90(rotations, [dimension, dimension + 1])
    };
    (transform(state, 1), transform(policy, 0))
}

impl PositionCodec<BoardState> for TicTacToeCodec {
    fn encode_position(state: &BoardState) -> Tensor {
        game_to_nn_input(state)
    }

    fn decode_policy(policy: &Tensor, moves: &[TicTacToeMove]) -> Result<Vec<f32>> {
        estimated_policy(policy, moves)
    }
}

impl TrainingCodec<BoardState> for TicTacToeCodec {
    type Policy = TicTacToePolicy;

    fn encode_policy_target(state: &BoardState, policy: &[f32]) -> Result<Self::Policy> {
        let moves = match state.get_state() {
            TerminationState::Moves(moves) => moves,
            TerminationState::Terminal(_) => bail!("terminal state cannot be a training sample"),
        };
        canonical_policy(policy, &moves)
    }

    fn policy_to_tensor(policy: &Self::Policy) -> Tensor {
        Tensor::from_slice(policy.as_flattened()).view([BoardState::N as i64, BoardState::N as i64])
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
        codec::{canonical_policy, estimated_policy, game_to_nn_input},
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
        let policy = estimated_policy(
            &policy,
            &[TicTacToeMove::from_xy(0, 1), TicTacToeMove::from_xy(0, 2)],
        )
        .unwrap();

        assert!((policy[0] - 0.5).abs() < 1e-6);
        assert!((policy[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn canonical_policy_uses_fixed_action_indices() {
        let policy = canonical_policy(
            &[0.25, 0.75],
            &[TicTacToeMove::from_xy(2, 3), TicTacToeMove::from_xy(10, 12)],
        )
        .unwrap();

        assert_eq!(policy[(2, 3)], 0.25);
        assert_eq!(policy[(10, 12)], 0.75);
        assert_eq!(policy[(0, 0)], 0.0);
        assert_eq!(policy.as_flattened().len(), 19 * 19);
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
