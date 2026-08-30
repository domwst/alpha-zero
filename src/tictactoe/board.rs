use std::ops::{Index, Range};

use serde::{Deserialize, Serialize};
use serde_big_array::BigArray;

use crate::alpha_zero::{Game, MoveParameters, TerminationState, TurnChange};

const N: usize = 19;

const BYTES: usize = (N * N - 1) / (std::mem::size_of::<u8>() * (8 / 2)) + 1;

#[derive(Serialize, Deserialize, Clone, Hash, PartialEq, Eq)]
#[repr(align(8))]
pub struct BoardState {
    #[serde(with = "BigArray")]
    state: [u8; BYTES],
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CellState {
    Empty = 0b00,
    X = 0b01,
    O = 0b10,
}

impl Default for BoardState {
    fn default() -> Self {
        Self::new()
    }
}

impl BoardState {
    pub const N: usize = N;

    pub fn new() -> Self {
        Self { state: [0; BYTES] }
    }

    pub fn set_inplace(&mut self, (x, y): (usize, usize), state: CellState) {
        debug_assert!(x < N && y < N);
        let idx = x * N + y;
        let chunk = idx / 4;
        let offset = idx % 4;

        let chunk = &mut self.state[chunk];
        *chunk &= !(3 << (2 * offset));

        *chunk |= (state as u8) << (2 * offset);
    }

    pub fn set(mut self, coord: (usize, usize), state: CellState) -> Self {
        self.set_inplace(coord, state);
        self
    }

    pub fn flip_players_inplace(&mut self) {
        const U8_MAGIC: u8 = u8::MAX / 0b11;
        for v in &mut self.state {
            let x = *v & U8_MAGIC;
            let o = (*v & (U8_MAGIC << 1)) >> 1;
            *v = (x << 1) | o;
        }
    }

    pub fn flip_players(mut self) -> Self {
        self.flip_players_inplace();
        self
    }

    pub fn is_win(&self) -> CellState {
        const RANGES: [Range<usize>; 3] = [4..N, 0..N, 0..(N - 4)];
        const DIRECTIONS: [(i32, i32); 4] = [(-1, 1), (0, 1), (1, 1), (1, 0)];

        for (dx, dy) in DIRECTIONS {
            for x in RANGES[(dx + 1) as usize].clone() {
                'cell: for y in RANGES[(dy + 1) as usize].clone() {
                    let goal = match self[(x, y)] {
                        CellState::Empty => continue,
                        v => v,
                    };

                    for k in 1..5 {
                        if self[((x as i32 + dx * k) as usize, (y as i32 + dy * k) as usize)]
                            != goal
                        {
                            continue 'cell;
                        }
                    }

                    return goal;
                }
            }
        }

        CellState::Empty
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TicTacToeMove(pub usize, pub usize);

impl MoveParameters for TicTacToeMove {
    fn turn_change(&self) -> TurnChange {
        TurnChange::SwitchPlayer
    }
}

impl Game for BoardState {
    type Move = TicTacToeMove;

    fn get_state(&self) -> TerminationState<Self::Move> {
        match self.is_win() {
            CellState::X => return TerminationState::Terminal(1.),
            CellState::O => return TerminationState::Terminal(-1.),
            CellState::Empty => {}
        }

        let moves = (0..N)
            .flat_map(|i| (0..N).map(move |j| (i, j)))
            .filter(|&crd| self[crd] == CellState::Empty)
            .map(|(i, j)| TicTacToeMove(i, j))
            .collect::<Vec<_>>();

        if moves.is_empty() {
            TerminationState::Terminal(0.0)
        } else {
            TerminationState::Moves(moves.into())
        }
    }

    fn make_move(&self, m: &Self::Move) -> Self {
        let mut new_state = self.clone();
        let &TicTacToeMove(i, j) = m;
        new_state.set_inplace((i, j), CellState::X);
        new_state.flip_players()
    }
}

impl Index<(usize, usize)> for BoardState {
    type Output = CellState;

    fn index(&self, (x, y): (usize, usize)) -> &Self::Output {
        assert!(x < N && y < N);
        let idx = x * N + y;
        let chunk = idx / 4;
        let offset = idx % 4;
        let val = (self.state[chunk] >> (2 * offset)) & 3;
        match val {
            0 => &CellState::Empty,
            1 => &CellState::X,
            2 => &CellState::O,
            _ => unreachable!(),
        }
    }
}

#[cfg(test)]
mod tests {
    extern crate test;

    use crate::{
        alpha_zero::{Game, TerminationState},
        tictactoe::{CellState, TicTacToeMove},
    };

    use super::BoardState;
    use test::Bencher;

    #[test]
    fn tic_tac_toe_win() {
        struct BoardWrapper(BoardState, usize, usize);

        impl BoardWrapper {
            fn set_inplace(&mut self, (x, y): (usize, usize), s: CellState) {
                self.0.set_inplace((x + self.1, y + self.2), s)
            }

            fn is_win(&self) -> CellState {
                self.0.is_win()
            }

            fn flip_players_inplace(&mut self) {
                self.0.flip_players_inplace()
            }
        }

        fn test_with_offset(ox: usize, oy: usize) {
            let mut board = BoardWrapper(BoardState::new(), ox, oy);
            assert_eq!(board.is_win(), CellState::Empty);
            board.set_inplace((0, 4), CellState::X);
            board.set_inplace((1, 3), CellState::X);
            board.set_inplace((2, 2), CellState::X);
            board.set_inplace((3, 1), CellState::X);
            assert_eq!(board.is_win(), CellState::Empty);
            board.set_inplace((4, 0), CellState::X);

            assert_eq!(board.is_win(), CellState::X);
            board.flip_players_inplace();
            assert_eq!(board.is_win(), CellState::O);
            board.set_inplace((2, 2), CellState::Empty);

            board.set_inplace((0, 0), CellState::O);
            board.set_inplace((0, 1), CellState::O);
            board.set_inplace((0, 2), CellState::O);
            assert_eq!(board.is_win(), CellState::Empty);
            board.set_inplace((0, 3), CellState::O);
            assert_eq!(board.is_win(), CellState::O);

            board.set_inplace((0, 4), CellState::Empty);
            assert_eq!(board.is_win(), CellState::Empty);

            board.set_inplace((1, 0), CellState::O);
            board.set_inplace((2, 0), CellState::O);
            assert_eq!(board.is_win(), CellState::Empty);
            board.set_inplace((3, 0), CellState::O);
            assert_eq!(board.is_win(), CellState::O);

            board.set_inplace((1, 0), CellState::X);
            assert_eq!(board.is_win(), CellState::Empty);

            board.set_inplace((1, 1), CellState::O);
            board.set_inplace((2, 2), CellState::O);
            board.set_inplace((3, 3), CellState::O);
            assert_eq!(board.is_win(), CellState::Empty);
            board.set_inplace((4, 4), CellState::O);
            assert_eq!(board.is_win(), CellState::O);
        }

        for ox in [0, 7, 14] {
            for oy in [0, 7, 14] {
                test_with_offset(ox, oy);
            }
        }
    }

    #[test]
    fn tic_tac_toe_draw() {
        let mut board = BoardState::new();
        for i in 0..19 {
            for j in 0..19 {
                let s = if (i / 2 + j) % 2 == 0 {
                    CellState::X
                } else {
                    CellState::O
                };
                board.set_inplace((i, j), s);
            }
        }

        assert_eq!(board.get_state(), TerminationState::Terminal(0.0));
    }

    #[bench]
    fn make_move(b: &mut Bencher) {
        let mut board = BoardState::new();

        b.iter(|| {
            for _ in 0..65536 {
                board = core::hint::black_box(board.make_move(&TicTacToeMove(0, 0)));
            }
            board.clone()
        });
    }
}
