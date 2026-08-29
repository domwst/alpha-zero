use anyhow::Result;
use rand::{rngs::SmallRng, SeedableRng};

use super::{run_match, Game, MatchRecord, MctsAgent, PositionEvaluator, RootNoise, Versus};

pub async fn do_battle<TGame, FirstEvaluator, SecondEvaluator, Temperature>(
    start: TGame,
    simulations: usize,
    c_puct: f32,
    temperature: Temperature,
    first_evaluator: FirstEvaluator,
    second_evaluator: SecondEvaluator,
) -> Result<MatchRecord<TGame>>
where
    TGame: Game + Clone + PartialEq + Send + Sync,
    TGame::Move: Clone + PartialEq + Send + Sync,
    FirstEvaluator: PositionEvaluator<TGame> + Send + Sync,
    SecondEvaluator: PositionEvaluator<TGame> + Send + Sync,
    Temperature: Fn(usize) -> f32 + Clone + Send,
{
    let first = MctsAgent::new(
        start.clone(),
        first_evaluator,
        RootNoise::None,
        simulations,
        c_puct,
        SmallRng::from_rng(&mut rand::rng()),
        temperature.clone(),
    );
    let second = MctsAgent::new(
        start.clone(),
        second_evaluator,
        RootNoise::None,
        simulations,
        c_puct,
        SmallRng::from_rng(&mut rand::rng()),
        temperature,
    );
    let mut controller = Versus::new(first, second);
    run_match(start, &mut controller).await
}
