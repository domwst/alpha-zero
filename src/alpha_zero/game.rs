#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MoveDescription<Move> {
    pub r#move: Move,
    pub player_switch: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TerminationState<Move> {
    Terminal(f32),
    Moves(Vec<MoveDescription<Move>>),
}

pub trait Game {
    type Move;

    fn get_state(&self) -> TerminationState<Self::Move>;
    // Should "switch" player if the move does so
    fn make_move(&self, m: Self::Move) -> Self;
}
