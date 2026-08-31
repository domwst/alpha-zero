use std::future::Future;

use anyhow::{Result, ensure};

use super::{Game, MoveParameters, TerminationState, TurnChange};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Seat {
    First,
    Second,
}

impl Seat {
    pub fn other(self) -> Self {
        match self {
            Self::First => Self::Second,
            Self::Second => Self::First,
        }
    }
}

pub struct Turn<'a, TGame: Game> {
    pub state: &'a TGame,
    pub legal_moves: &'a [TGame::Move],
    pub seat: Seat,
    pub ply: usize,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct DecisionDiagnostics {
    pub value_estimate: Option<f32>,
    pub sampling_policy: Option<Vec<f32>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MoveDecision {
    pub move_index: usize,
    pub training_policy: Option<Vec<f32>>,
    pub diagnostics: DecisionDiagnostics,
}

impl MoveDecision {
    pub fn new(move_index: usize) -> Self {
        Self {
            move_index,
            training_policy: None,
            diagnostics: DecisionDiagnostics::default(),
        }
    }
}

pub struct AppliedMove<'a, TGame: Game> {
    pub previous_state: &'a TGame,
    pub next_state: &'a TGame,
    pub action: &'a TGame::Move,
    pub move_index: usize,
    pub actor: Seat,
    pub ply: usize,
    pub turn_change: TurnChange,
}

pub trait Agent<TGame: Game> {
    fn select_move<'a>(
        &'a mut self,
        turn: Turn<'a, TGame>,
    ) -> impl Future<Output = Result<MoveDecision>> + Send + 'a;

    fn observe_move(&mut self, _applied: &AppliedMove<'_, TGame>) -> Result<()> {
        Ok(())
    }
}

pub trait MatchController<TGame: Game> {
    fn select_move<'a>(
        &'a mut self,
        turn: Turn<'a, TGame>,
    ) -> impl Future<Output = Result<MoveDecision>> + Send + 'a;

    fn observe_move(&mut self, applied: &AppliedMove<'_, TGame>) -> Result<()>;
}

pub struct Shared<OneAgent> {
    agent: OneAgent,
}

impl<OneAgent> Shared<OneAgent> {
    pub fn new(agent: OneAgent) -> Self {
        Self { agent }
    }

    pub fn agent(&self) -> &OneAgent {
        &self.agent
    }

    pub fn agent_mut(&mut self) -> &mut OneAgent {
        &mut self.agent
    }

    pub fn into_inner(self) -> OneAgent {
        self.agent
    }
}

impl<TGame, OneAgent> MatchController<TGame> for Shared<OneAgent>
where
    TGame: Game + Sync,
    TGame::Move: Sync,
    OneAgent: Agent<TGame> + Send,
{
    async fn select_move<'a>(&'a mut self, turn: Turn<'a, TGame>) -> Result<MoveDecision> {
        self.agent.select_move(turn).await
    }

    fn observe_move(&mut self, applied: &AppliedMove<'_, TGame>) -> Result<()> {
        self.agent.observe_move(applied)
    }
}

pub struct Versus<FirstAgent, SecondAgent> {
    first: FirstAgent,
    second: SecondAgent,
}

impl<FirstAgent, SecondAgent> Versus<FirstAgent, SecondAgent> {
    pub fn new(first: FirstAgent, second: SecondAgent) -> Self {
        Self { first, second }
    }

    pub fn agents(&self) -> (&FirstAgent, &SecondAgent) {
        (&self.first, &self.second)
    }

    pub fn agents_mut(&mut self) -> (&mut FirstAgent, &mut SecondAgent) {
        (&mut self.first, &mut self.second)
    }

    pub fn into_inner(self) -> (FirstAgent, SecondAgent) {
        (self.first, self.second)
    }
}

impl<TGame, FirstAgent, SecondAgent> MatchController<TGame> for Versus<FirstAgent, SecondAgent>
where
    TGame: Game + Sync,
    TGame::Move: Sync,
    FirstAgent: Agent<TGame> + Send,
    SecondAgent: Agent<TGame> + Send,
{
    async fn select_move<'a>(&'a mut self, turn: Turn<'a, TGame>) -> Result<MoveDecision> {
        match turn.seat {
            Seat::First => self.first.select_move(turn).await,
            Seat::Second => self.second.select_move(turn).await,
        }
    }

    fn observe_move(&mut self, applied: &AppliedMove<'_, TGame>) -> Result<()> {
        self.first.observe_move(applied)?;
        self.second.observe_move(applied)
    }
}

pub struct MatchPly<TGame: Game> {
    pub state: TGame,
    pub action: TGame::Move,
    pub actor: Seat,
    pub turn_change: TurnChange,
    pub decision: MoveDecision,
}

pub struct MatchRecord<TGame: Game> {
    pub plies: Vec<MatchPly<TGame>>,
    pub terminal_state: TGame,
    pub terminal_actor: Seat,
    pub terminal_value: f32,
}

impl<TGame: Game> MatchRecord<TGame> {
    pub fn value_for(&self, seat: Seat) -> f32 {
        if seat == self.terminal_actor {
            self.terminal_value
        } else {
            -self.terminal_value
        }
    }
}

fn validate_training_policy(policy: &[f32], legal_moves: usize) -> Result<()> {
    ensure!(
        policy.len() == legal_moves,
        "training policy has {} entries for {legal_moves} legal moves",
        policy.len()
    );
    ensure!(
        policy
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0),
        "training policy contains invalid probabilities"
    );
    ensure!(
        (policy.iter().sum::<f32>() - 1.0).abs() < 1e-4,
        "training policy is not normalized"
    );
    Ok(())
}

pub async fn run_match<TGame, Controller>(
    mut state: TGame,
    controller: &mut Controller,
) -> Result<MatchRecord<TGame>>
where
    TGame: Game + Clone + Send + Sync,
    TGame::Move: Clone + Send + Sync,
    Controller: MatchController<TGame> + Send,
{
    let mut plies = Vec::new();
    let mut actor = Seat::First;

    loop {
        let legal_moves = match state.get_state() {
            TerminationState::Terminal(value) => {
                ensure!(
                    value.is_finite(),
                    "game returned a non-finite terminal value"
                );
                return Ok(MatchRecord {
                    plies,
                    terminal_state: state,
                    terminal_actor: actor,
                    terminal_value: value,
                });
            }
            TerminationState::Moves(moves) => moves,
        };

        let ply = plies.len();
        let decision = controller
            .select_move(Turn {
                state: &state,
                legal_moves: &legal_moves,
                seat: actor,
                ply,
            })
            .await?;
        ensure!(
            decision.move_index < legal_moves.len(),
            "agent selected move {} from {} legal moves",
            decision.move_index,
            legal_moves.len()
        );
        if let Some(policy) = &decision.training_policy {
            validate_training_policy(policy, legal_moves.len())?;
        }

        let action = legal_moves[decision.move_index].clone();
        let turn_change = action.turn_change();
        let next_state = state.make_move(&action);
        controller.observe_move(&AppliedMove {
            previous_state: &state,
            next_state: &next_state,
            action: &action,
            move_index: decision.move_index,
            actor,
            ply,
            turn_change,
        })?;

        let previous_state = std::mem::replace(&mut state, next_state);
        plies.push(MatchPly {
            state: previous_state,
            action,
            actor,
            turn_change,
            decision,
        });
        if turn_change.switches_player() {
            actor = actor.other();
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{future::Future, future::ready};

    use super::*;

    #[derive(Clone)]
    struct TwoPlyGame(usize);

    #[derive(Clone, Copy, Debug, PartialEq)]
    struct OnlyMove;

    impl MoveParameters for OnlyMove {
        fn turn_change(&self) -> TurnChange {
            TurnChange::SwitchPlayer
        }
    }

    impl Game for TwoPlyGame {
        type Move = OnlyMove;

        fn get_state(&self) -> TerminationState<Self::Move> {
            if self.0 == 2 {
                TerminationState::Terminal(0.5)
            } else {
                TerminationState::Moves(vec![OnlyMove].into())
            }
        }

        fn make_move(&self, _move: &Self::Move) -> Self {
            Self(self.0 + 1)
        }
    }

    #[derive(Clone)]
    struct BonusTurnGame(usize);

    #[derive(Clone, Copy)]
    struct BonusMove(TurnChange);

    impl MoveParameters for BonusMove {
        fn turn_change(&self) -> TurnChange {
            self.0
        }
    }

    impl Game for BonusTurnGame {
        type Move = BonusMove;

        fn get_state(&self) -> TerminationState<Self::Move> {
            match self.0 {
                0 => TerminationState::Moves(vec![BonusMove(TurnChange::SamePlayer)].into()),
                1 => TerminationState::Moves(vec![BonusMove(TurnChange::SwitchPlayer)].into()),
                _ => TerminationState::Terminal(0.25),
            }
        }

        fn make_move(&self, _move: &Self::Move) -> Self {
            Self(self.0 + 1)
        }
    }

    #[derive(Default)]
    struct CountingAgent {
        selections: Vec<Seat>,
        observations: Vec<Seat>,
    }

    impl Agent<TwoPlyGame> for CountingAgent {
        fn select_move<'a>(
            &'a mut self,
            turn: Turn<'a, TwoPlyGame>,
        ) -> impl Future<Output = Result<MoveDecision>> + Send + 'a {
            self.selections.push(turn.seat);
            ready(Ok(MoveDecision::new(0)))
        }

        fn observe_move(&mut self, applied: &AppliedMove<'_, TwoPlyGame>) -> Result<()> {
            self.observations.push(applied.actor);
            Ok(())
        }
    }

    impl Agent<BonusTurnGame> for CountingAgent {
        fn select_move<'a>(
            &'a mut self,
            turn: Turn<'a, BonusTurnGame>,
        ) -> impl Future<Output = Result<MoveDecision>> + Send + 'a {
            self.selections.push(turn.seat);
            ready(Ok(MoveDecision::new(0)))
        }

        fn observe_move(&mut self, applied: &AppliedMove<'_, BonusTurnGame>) -> Result<()> {
            self.observations.push(applied.actor);
            Ok(())
        }
    }

    #[tokio::test]
    async fn shared_uses_one_agent_and_observes_each_transition_once() {
        let mut controller = Shared::new(CountingAgent::default());
        let record = run_match(TwoPlyGame(0), &mut controller).await.unwrap();

        assert_eq!(controller.agent().selections, [Seat::First, Seat::Second]);
        assert_eq!(controller.agent().observations, [Seat::First, Seat::Second]);
        assert_eq!(record.plies.len(), 2);
        assert_eq!(record.terminal_actor, Seat::First);
        assert_eq!(record.value_for(Seat::First), 0.5);
        assert_eq!(record.value_for(Seat::Second), -0.5);
    }

    #[tokio::test]
    async fn versus_routes_selection_and_broadcasts_transitions() {
        let mut controller = Versus::new(CountingAgent::default(), CountingAgent::default());
        run_match(TwoPlyGame(0), &mut controller).await.unwrap();
        let (first, second) = controller.agents();

        assert_eq!(first.selections, [Seat::First]);
        assert_eq!(second.selections, [Seat::Second]);
        assert_eq!(first.observations, [Seat::First, Seat::Second]);
        assert_eq!(second.observations, [Seat::First, Seat::Second]);
    }

    #[tokio::test]
    async fn same_player_transition_keeps_the_active_seat() {
        let mut controller = Shared::new(CountingAgent::default());
        let record = run_match(BonusTurnGame(0), &mut controller).await.unwrap();

        assert_eq!(controller.agent().selections, [Seat::First, Seat::First]);
        assert_eq!(record.terminal_actor, Seat::Second);
        assert_eq!(record.value_for(Seat::First), -0.25);
    }
}
