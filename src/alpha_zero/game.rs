pub trait MoveParameters {
    fn is_player_switch(&self) -> bool;
}

#[derive(Debug, Clone, PartialEq)]
pub enum TerminationState<Move> {
    Terminal(f32),
    Moves(Vec<Move>),
}

impl<Move> TerminationState<Move> {
    pub fn get_terminal(&self) -> Option<f32> {
        match self {
            TerminationState::Terminal(f) => Some(*f),
            TerminationState::Moves(_) => None,
        }
    }

    pub fn get_moves(self) -> Option<Vec<Move>> {
        match self {
            TerminationState::Terminal(_) => None,
            TerminationState::Moves(moves) => Some(moves),
        }
    }
}

pub trait Game {
    type Move: MoveParameters;

    fn get_state(&self) -> TerminationState<Self::Move>;
    // Should "switch" player if the move does so
    fn make_move(&self, m: &Self::Move) -> Self;
}
