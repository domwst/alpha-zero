use std::{future::Future, marker::PhantomData};

use anyhow::{ensure, Result};
use rand::Rng;

use super::{
    apply_temperature, sample_policy, Agent, AppliedMove, DecisionDiagnostics, Game,
    MonteCarloTree, MoveDecision, PositionEvaluator, RootNoise, Turn,
};

pub struct MctsAgent<TGame: Game, Evaluator: PositionEvaluator<TGame>, Random, Temperature> {
    tree: MonteCarloTree<TGame, Evaluator>,
    simulations: usize,
    c_puct: f32,
    random: Random,
    temperature: Temperature,
}

impl<TGame, Evaluator, Random, Temperature> MctsAgent<TGame, Evaluator, Random, Temperature>
where
    TGame: Game + PartialEq + Send + Sync,
    TGame::Move: Clone + PartialEq + Send + Sync,
    Evaluator: PositionEvaluator<TGame> + Send + Sync,
{
    pub fn new(
        start: TGame,
        evaluator: Evaluator,
        root_noise: RootNoise,
        simulations: usize,
        c_puct: f32,
        random: Random,
        temperature: Temperature,
    ) -> Self {
        assert!(simulations > 0);
        assert!(c_puct.is_finite() && c_puct >= 0.0);
        Self {
            tree: MonteCarloTree::new(start, evaluator, root_noise),
            simulations,
            c_puct,
            random,
            temperature,
        }
    }

    pub fn tree(&self) -> &MonteCarloTree<TGame, Evaluator> {
        &self.tree
    }
}

impl<TGame, Evaluator, Random, Temperature> Agent<TGame>
    for MctsAgent<TGame, Evaluator, Random, Temperature>
where
    TGame: Game + Clone + PartialEq + Send + Sync,
    TGame::Move: Clone + PartialEq + Send + Sync,
    Evaluator: PositionEvaluator<TGame> + Send + Sync,
    Random: Rng + Send,
    Temperature: Fn(usize) -> f32 + Send,
{
    fn select_move<'a>(
        &'a mut self,
        turn: Turn<'a, TGame>,
    ) -> impl Future<Output = Result<MoveDecision>> + Send + 'a {
        async move {
            ensure!(
                self.tree.matches_position(turn.state, turn.legal_moves),
                "MCTS root does not match the authoritative game state"
            );
            self.tree
                .do_simulations(self.simulations, self.c_puct, &mut self.random)
                .await?;
            ensure!(
                self.tree.matches_position(turn.state, turn.legal_moves),
                "MCTS root does not match the authoritative game state"
            );
            let search_policy = self.tree.get_policy();
            let sampling_policy = apply_temperature(&search_policy, (self.temperature)(turn.ply));
            let move_index = sample_policy(&sampling_policy, &mut self.random);

            Ok(MoveDecision {
                move_index,
                training_policy: Some(sampling_policy),
                diagnostics: DecisionDiagnostics {
                    value_estimate: self.tree.get_network_state_estimation(),
                    sampling_policy: None,
                },
            })
        }
    }

    fn observe_move(&mut self, applied: &AppliedMove<'_, TGame>) -> Result<()> {
        self.tree.advance(
            applied.move_index,
            applied.action,
            applied.next_state.clone(),
        )
    }
}

pub struct PolicyAgent<TGame: Game, Evaluator, Random, Temperature> {
    evaluator: Evaluator,
    random: Random,
    temperature: Temperature,
    _game: PhantomData<fn() -> TGame>,
}

impl<TGame: Game, Evaluator, Random, Temperature>
    PolicyAgent<TGame, Evaluator, Random, Temperature>
{
    pub fn new(evaluator: Evaluator, random: Random, temperature: Temperature) -> Self {
        Self {
            evaluator,
            random,
            temperature,
            _game: PhantomData,
        }
    }
}

impl<TGame, Evaluator, Random, Temperature> Agent<TGame>
    for PolicyAgent<TGame, Evaluator, Random, Temperature>
where
    TGame: Game + Send + Sync,
    TGame::Move: Send + Sync,
    Evaluator: PositionEvaluator<TGame> + Send + Sync,
    Random: Rng + Send,
    Temperature: Fn(usize) -> f32 + Send,
{
    fn select_move<'a>(
        &'a mut self,
        turn: Turn<'a, TGame>,
    ) -> impl Future<Output = Result<MoveDecision>> + Send + 'a {
        async move {
            let evaluation = self
                .evaluator
                .evaluate(turn.state, turn.legal_moves)
                .await?;
            evaluation.validate_for(turn.legal_moves.len())?;
            let sampling_policy =
                apply_temperature(&evaluation.legal_policy, (self.temperature)(turn.ply));
            let move_index = sample_policy(&sampling_policy, &mut self.random);
            Ok(MoveDecision {
                move_index,
                training_policy: None,
                diagnostics: DecisionDiagnostics {
                    value_estimate: Some(evaluation.value),
                    sampling_policy: Some(sampling_policy),
                },
            })
        }
    }
}
