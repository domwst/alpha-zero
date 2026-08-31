use anyhow::Result;
use rand::Rng;

use super::{Game, MatchRecord, MctsAgent, PositionEvaluator, RootNoise, Shared, run_match};

pub async fn generate_self_played_game<TGame, Evaluator, Temperature, Random>(
    start: TGame,
    simulations: usize,
    c_puct: f32,
    temperature: Temperature,
    evaluator: Evaluator,
    random: Random,
) -> Result<MatchRecord<TGame>>
where
    TGame: Game + Clone + PartialEq + Send + Sync,
    TGame::Move: Clone + PartialEq + Send + Sync,
    Evaluator: PositionEvaluator<TGame> + Send + Sync,
    Temperature: Fn(usize) -> f32 + Send,
    Random: Rng + Send,
{
    let agent = MctsAgent::new(
        start.clone(),
        evaluator,
        RootNoise::Dirichlet {
            alpha: 0.1,
            epsilon: 0.25,
        },
        simulations,
        c_puct,
        random,
        temperature,
    );
    let mut controller = Shared::new(agent);
    run_match(start, &mut controller).await
}
