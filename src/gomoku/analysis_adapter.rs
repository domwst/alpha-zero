use super::{BoardState, CellState};
use crate::engine::GameAdapter;
use anyhow::{Result, ensure};
use serde::Deserialize;
use serde_json::{Value, json};

pub struct GomokuAdapter;

impl GameAdapter for GomokuAdapter {
    type Position = BoardState;
    const ID: &'static str = "gomoku19_five_v1";

    fn decode(state: Value) -> Result<BoardState> {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct State {
            cells: Vec<u8>,
        }
        let State { cells } = serde_json::from_value(state)?;
        ensure!(
            cells.len() == BoardState::N * BoardState::N && cells.iter().all(|&n| n <= 2),
            "Invalid position cells"
        );
        let current = cells.iter().filter(|&&n| n == 1).count();
        let opponent = cells.iter().filter(|&&n| n == 2).count();
        ensure!(
            opponent == current || opponent == current + 1,
            "Illegal canonical player counts"
        );
        let mut board = BoardState::new();
        for (i, cell) in cells.into_iter().enumerate() {
            board.set_inplace(
                (i / BoardState::N, i % BoardState::N),
                match cell {
                    1 => CellState::X,
                    2 => CellState::O,
                    _ => CellState::Empty,
                },
            );
        }
        Ok(board)
    }

    fn describe() -> Value {
        json!({"id":Self::ID, "rows":BoardState::N, "columns":BoardState::N,
            "perspective":"player_to_move", "actions":"board_coordinates", "state_schema":"canonical_cells_v1"})
    }
}
