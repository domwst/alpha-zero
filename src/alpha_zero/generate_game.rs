use rand::{rngs::SmallRng, SeedableRng};

use crate::alpha_zero::{
    AlphaZeroAdapter, AlphaZeroNet, Game, MonteCarloTree, MoveParameters, RootNoise,
};

use super::{apply_temperature, sample_policy, NetworkBatchedExecutorHandle, TerminationState};

pub async fn generate_self_played_game<
    TGame: Game + Clone,
    TNet: AlphaZeroNet,
    TAdapter: AlphaZeroAdapter<TGame, TNet>,
    F: FnMut(usize) -> f32,
>(
    start: TGame,
    samples: usize,
    c_puct: f32,
    mut temp: F,
    executor: NetworkBatchedExecutorHandle<TNet>,
) -> Vec<(TGame, Vec<f32>, f32)> {
    let mut tree = MonteCarloTree::<TGame, TNet, TAdapter>::new(
        start.clone(),
        executor,
        RootNoise::Dirichlet {
            alpha: 0.1,
            epsilon: 0.25,
        },
    );
    let mut turn = 0;
    let mut rng = SmallRng::from_rng(&mut rand::rng());

    let mut state = start;

    let mut history = vec![];

    let mut value = loop {
        let moves = match state.get_state() {
            TerminationState::Moves(moves) => moves,
            TerminationState::Terminal(value) => break value,
        };
        tree.do_simulations(samples, c_puct, &mut rng).await;
        let policy = apply_temperature(&tree.get_policy(), temp(turn));

        let r#move = sample_policy(&policy, &mut rng);

        let new_state = state.make_move(&moves[r#move]);
        tree.do_move(r#move);

        history.push((state, policy, moves[r#move].is_player_switch()));
        state = new_state;
        turn += 1;
    };

    let mut result = Vec::with_capacity(history.len());
    while let Some((state, policy, switch)) = history.pop() {
        if switch {
            value = -value;
        }
        result.push((state, policy, value));
    }
    result.reverse();
    result
}
