use anyhow::Result;
use rand::{SeedableRng, rngs::SmallRng};

use super::{Game, MatchRecord, MctsAgent, PositionEvaluator, RootNoise, Versus, run_match};

#[derive(Clone, Copy, Debug)]
pub struct BattleSettings {
    pub simulations: usize,
    pub c_puct: f32,
    pub first_temperature: f32,
    pub second_temperature: f32,
    pub first_seed: u64,
    pub second_seed: u64,
}

pub async fn do_battle<TGame, FirstEvaluator, SecondEvaluator>(
    start: TGame,
    settings: BattleSettings,
    first_evaluator: FirstEvaluator,
    second_evaluator: SecondEvaluator,
) -> Result<MatchRecord<TGame>>
where
    TGame: Game + Clone + PartialEq + Send + Sync,
    TGame::Move: Clone + PartialEq + Send + Sync,
    FirstEvaluator: PositionEvaluator<TGame> + Send + Sync,
    SecondEvaluator: PositionEvaluator<TGame> + Send + Sync,
{
    let first = MctsAgent::new(
        start.clone(),
        first_evaluator,
        RootNoise::None,
        settings.simulations,
        settings.c_puct,
        SmallRng::seed_from_u64(settings.first_seed),
        move |_| settings.first_temperature,
    );
    let second = MctsAgent::new(
        start.clone(),
        second_evaluator,
        RootNoise::None,
        settings.simulations,
        settings.c_puct,
        SmallRng::seed_from_u64(settings.second_seed),
        move |_| settings.second_temperature,
    );
    let mut controller = Versus::new(first, second);
    run_match(start, &mut controller).await
}
