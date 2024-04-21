use tch::{Device, IndexOp, Kind};

use crate::alpha_zero::{AlphaZeroAdapter, Game, MoveDescription};

use super::{BoardState, CellState, TicTacToeNet};

pub struct TicTacToeAlphaZeroAdapter;

impl AlphaZeroAdapter<BoardState, TicTacToeNet> for TicTacToeAlphaZeroAdapter {
    fn convert_game_to_nn_input(state: &BoardState, (kind, dev): (Kind, Device)) -> tch::Tensor {
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
        Tensor::from_slice(fld.flatten().flatten())
            .view([2, 19, 19])
            .to_kind(kind)
            .to_device(dev)
    }

    fn get_estimated_policy(
        policy: Tensor,
        moves: &[MoveDescription<<BoardState as Game>::Move>],
    ) -> Vec<f32> {
        let policy = policy.exp();
        let mut res = Vec::with_capacity(moves.len());
        for &MoveDescription {
            r#move: (i, j),
            player_switch: _,
        } in moves
        {
            res.push(f32::try_from(policy.i((i as i64, j as i64))).unwrap());
        }

        let mx = res
            .iter()
            .copied()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        if mx > 0. {
            for x in &mut res {
                *x /= mx;
            }
        }

        res
    }

    fn convert_policy_to_nn(
        policy: &[f32],
        moves: &[MoveDescription<<BoardState as Game>::Move>],
        (kind, dev): (Kind, Device),
    ) -> tch::Tensor {
        let mut res = [[0f32; 19]; 19];
        for (&(i, j), &pol) in moves
            .iter()
            .map(
                |MoveDescription {
                     r#move,
                     player_switch: _,
                 }| r#move,
            )
            .zip(policy)
        {
            res[i][j] = pol;
        }
        Tensor::from_slice(res.flatten())
            .view([19, 19])
            .to_kind(kind)
            .to_device(dev)
    }
}

#[cfg(test)]
mod tests {
    use tch::{Device, IndexOp, Kind, Tensor};

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

        let tensor =
            TicTacToeAlphaZeroAdapter::convert_game_to_nn_input(&game, (Kind::Float, Device::Cpu));
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
