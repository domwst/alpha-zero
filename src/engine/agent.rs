use std::marker::PhantomData;

use anyhow::{Result, ensure};
use rand::Rng;

use super::{
    Agent, AppliedMove, DecisionDiagnostics, Game, MonteCarloTree, MoveDecision, PositionEvaluator,
    RootNoise, Turn, apply_temperature, apply_top_p, sample_policy,
};

pub struct MctsAgent<TGame: Game, Evaluator: PositionEvaluator<TGame>, Random, Temperature> {
    tree: MonteCarloTree<TGame, Evaluator>,
    simulations: usize,
    c_puct: f32,
    random: Random,
    temperature: Temperature,
    top_p: f64,
}

impl<TGame, Evaluator, Random, Temperature> MctsAgent<TGame, Evaluator, Random, Temperature>
where
    TGame: Game + Clone + PartialEq + Send + Sync,
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
            top_p: 1.0,
        }
    }

    pub fn with_top_p(mut self, top_p: f64) -> Self {
        assert!(top_p.is_finite() && top_p > 0.0 && top_p <= 1.0);
        self.top_p = top_p;
        self
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
    async fn select_move<'a>(&'a mut self, turn: Turn<'a, TGame>) -> Result<MoveDecision> {
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
        let sampling_policy = apply_top_p(&sampling_policy, self.top_p);
        let move_index = sample_policy(&sampling_policy, &mut self.random);

        Ok(MoveDecision {
            move_index,
            // Exploration temperature controls the trajectory, not the search target.
            training_policy: Some(search_policy),
            diagnostics: DecisionDiagnostics {
                value_estimate: self.tree.get_network_state_estimation(),
                sampling_policy: Some(sampling_policy),
            },
        })
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
    async fn select_move<'a>(&'a mut self, turn: Turn<'a, TGame>) -> Result<MoveDecision> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        engine::{PositionEvaluation, Seat},
        gomoku::BoardState,
    };
    use rand::{SeedableRng, rngs::SmallRng};

    struct UniformEvaluator;

    struct PeakedEvaluator;

    impl PositionEvaluator<BoardState> for PeakedEvaluator {
        async fn evaluate(
            &self,
            _state: &BoardState,
            moves: &[<BoardState as Game>::Move],
        ) -> Result<PositionEvaluation> {
            let mut policy = vec![0.0; moves.len()];
            policy[0] = 0.97;
            policy[1] = 0.03;
            Ok(PositionEvaluation {
                value: 0.0,
                legal_policy: policy,
            })
        }
    }

    #[tokio::test]
    async fn nucleus_changes_sampling_without_changing_search_target() {
        let state = BoardState::new();
        let termination = state.get_state();
        let legal_moves = termination.get_moves().unwrap();
        let mut decisions = Vec::new();
        for top_p in [1.0, 0.95] {
            let mut agent = MctsAgent::new(
                state.clone(),
                PeakedEvaluator,
                RootNoise::None,
                256,
                1.0,
                SmallRng::seed_from_u64(7),
                |_| 1.0,
            )
            .with_top_p(top_p);
            decisions.push(
                agent
                    .select_move(Turn {
                        state: &state,
                        legal_moves,
                        seat: Seat::First,
                        ply: 0,
                    })
                    .await
                    .unwrap(),
            );
        }
        assert_eq!(decisions[0].training_policy, decisions[1].training_policy);
        let before = decisions[0].diagnostics.sampling_policy.as_ref().unwrap();
        let after = decisions[1].diagnostics.sampling_policy.as_ref().unwrap();
        assert!(
            before.iter().filter(|&&p| p > 0.0).count()
                > after.iter().filter(|&&p| p > 0.0).count()
        );
        assert!(after[decisions[1].move_index] > 0.0);
        assert!((after.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }

    impl PositionEvaluator<BoardState> for UniformEvaluator {
        async fn evaluate<'a>(
            &'a self,
            _state: &'a BoardState,
            moves: &'a [<BoardState as Game>::Move],
        ) -> Result<PositionEvaluation> {
            Ok(PositionEvaluation {
                value: 0.0,
                legal_policy: vec![1.0 / moves.len() as f32; moves.len()],
            })
        }
    }

    #[tokio::test]
    async fn temperature_changes_sampling_but_preserves_the_search_training_target() {
        let state = BoardState::new();
        let termination = state.get_state();
        let legal_moves = termination.get_moves().unwrap();
        let mut targets = Vec::new();
        for temperature in [0.0, 0.7, 1.0] {
            let mut agent = MctsAgent::new(
                state.clone(),
                UniformEvaluator,
                RootNoise::None,
                64,
                1.0,
                SmallRng::seed_from_u64(7),
                move |_| temperature,
            );
            let decision = agent
                .select_move(Turn {
                    state: &state,
                    legal_moves,
                    seat: Seat::First,
                    ply: 0,
                })
                .await
                .unwrap();
            let target = decision.training_policy.unwrap();
            let sampling = decision.diagnostics.sampling_policy.unwrap();
            assert_eq!(target, agent.tree().get_policy());
            assert!((target.iter().sum::<f32>() - 1.0).abs() < 1e-6);
            assert!(target.iter().filter(|&&p| p > 0.0).count() > 1);
            assert!(sampling[decision.move_index] > 0.0);
            if temperature == 0.0 {
                assert_eq!(sampling.iter().filter(|&&p| p > 0.0).count(), 1);
                assert_eq!(sampling[decision.move_index], 1.0);
                assert_ne!(target, sampling);
            }
            targets.push(target);
        }
        assert!(targets.windows(2).all(|pair| pair[0] == pair[1]));
    }
}
