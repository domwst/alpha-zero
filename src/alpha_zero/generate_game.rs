use rand::{
    distributions::{Distribution, WeightedIndex},
    thread_rng,
};
use tch::{Device, Kind};

use crate::alpha_zero::{AlphaZeroAdapter, AlphaZeroNet, Game, MonteCarloTree, MoveParameters};

use super::{NetworkBatchedExecutorHandle, TerminationState};

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
    options: (Kind, Device),
) -> Vec<(TGame, Vec<f32>, f32)> {
    let mut tree = MonteCarloTree::<TGame, TNet, TAdapter>::new(start.clone(), executor, options);
    let mut turn = 0;

    let mut state_id = 0;
    let mut state = start;

    let mut history = vec![];

    let mut value = loop {
        let moves = match state.get_state() {
            TerminationState::Moves(moves) => moves,
            TerminationState::Terminal(value) => break value,
        };
        tree.do_simulations(samples, c_puct, state_id).await;
        let mut policy = tree.get_policy(state_id);
        let original_policy = policy.clone();

        let mx = policy
            .iter()
            .copied()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        let temp = temp(turn);
        for v in &mut policy {
            *v /= mx;
            *v = v.powf(1.0 / (temp + 0.01));
        }

        let r#move = WeightedIndex::new(policy)
            .unwrap()
            .sample(&mut thread_rng());
        let new_state = state.make_move(&moves[r#move]);
        state_id = tree.get_next_state(state_id, r#move);

        history.push((state, original_policy, moves[r#move].is_player_switch()));
        state = new_state;
        turn += 1;
    };

    let mut result = Vec::with_capacity(history.len());
    while let Some((state, policy, switch)) = history.pop() {
        if switch {
            value = 1.0 - value;
        }
        result.push((state, policy, value));
    }
    result
}
