use anyhow::Result;
use rand::{rngs::SmallRng, SeedableRng};

use super::{run_match, Game, MatchRecord, MctsAgent, PositionEvaluator, RootNoise, Shared};

pub async fn generate_self_played_game<TGame, Evaluator, Temperature>(
    start: TGame,
    simulations: usize,
    c_puct: f32,
    temperature: Temperature,
    evaluator: Evaluator,
) -> Result<MatchRecord<TGame>>
where
    TGame: Game + Clone + PartialEq + Send + Sync,
    TGame::Move: Clone + PartialEq + Send + Sync,
    Evaluator: PositionEvaluator<TGame> + Send + Sync,
    Temperature: Fn(usize) -> f32 + Send,
{
    let random = SmallRng::from_rng(&mut rand::rng());
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
