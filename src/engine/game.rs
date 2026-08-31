#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TurnChange {
    SamePlayer = 0,
    SwitchPlayer = 1,
}

impl TurnChange {
    pub fn switches_player(self) -> bool {
        self == Self::SwitchPlayer
    }
}

pub trait MoveParameters {
    fn turn_change(&self) -> TurnChange;
}

#[derive(Debug, Clone, PartialEq)]
pub enum TerminationState<Move> {
    /// Value from the current canonical player's perspective.
    Terminal(f32),
    Moves(Box<[Move]>),
}

impl<Move> TerminationState<Move> {
    pub fn get_terminal(&self) -> Option<f32> {
        match self {
            TerminationState::Terminal(f) => Some(*f),
            TerminationState::Moves(_) => None,
        }
    }

    pub fn get_moves(&self) -> Option<&[Move]> {
        match self {
            TerminationState::Terminal(_) => None,
            TerminationState::Moves(moves) => Some(moves),
        }
    }
}

/// A canonical, two-player, zero-sum game state.
///
/// When a move switches players, `make_move` must return the successor from the
/// next player's perspective. Terminal values use the perspective represented
/// by the terminal state. Legal moves must use a deterministic order because
/// evaluator and training policies are aligned with that order.
pub trait Game {
    type Move: MoveParameters;

    fn get_state(&self) -> TerminationState<Self::Move>;
    fn make_move(&self, m: &Self::Move) -> Self;
}
