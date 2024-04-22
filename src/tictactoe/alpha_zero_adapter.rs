use tch::Tensor;

use crate::alpha_zero::{AlphaZeroAdapter, Game};

use super::{BoardState, CellState, TicTacToeMove, TicTacToeNet};

pub struct TicTacToeAlphaZeroAdapter;

impl AlphaZeroAdapter<BoardState, TicTacToeNet> for TicTacToeAlphaZeroAdapter {
    fn convert_game_to_nn_input(state: &BoardState) -> tch::Tensor {
        // let start = Instant::now();
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
        let res = Tensor::from_slice(fld.flatten().flatten()).view([2, 19, 19]);
        // println!("Converted input to tensor in {:?}", Instant::now() - start);
        res
    }

    fn get_estimated_policy(policy: Tensor, moves: &[<BoardState as Game>::Move]) -> Vec<f32> {
        // let start = Instant::now();
        let policy = policy.exp();
        let mut res = Vec::with_capacity(moves.len());
        let policy = <Vec<f32>>::try_from(policy.view([-1])).unwrap();
        for &TicTacToeMove(i, j) in moves {
            res.push(policy[i * 19 + j]);
        }

        let sum = res.iter().sum::<f32>();
        if sum > 0. {
            for x in &mut res {
                *x /= sum;
            }
        }

        // println!("Calculated policy in {:?}", Instant::now() - start);

        res
    }

    fn convert_policy_to_nn(policy: &[f32], moves: &[<BoardState as Game>::Move]) -> tch::Tensor {
        let mut res = [[0f32; 19]; 19];
        for (&TicTacToeMove(i, j), &pol) in moves.iter().zip(policy) {
            res[i][j] = pol;
        }
        Tensor::from_slice(res.flatten()).view([19, 19])
    }
}

#[cfg(test)]
mod tests {
    use tch::IndexOp;

    use crate::{
        alpha_zero::AlphaZeroAdapter,
        tictactoe::{BoardState, CellState},
    };

    use super::TicTacToeAlphaZeroAdapter;

    #[test]
    fn convert_board_to_tensor() {
        let mut game = BoardState::new();
        game.set_inplace((10, 0), CellState::O);
        game.set_inplace((1, 3), CellState::X);

        let tensor = TicTacToeAlphaZeroAdapter::convert_game_to_nn_input(&game);
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
}
