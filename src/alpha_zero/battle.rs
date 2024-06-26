use rand::{thread_rng, Rng};

use super::{
    sample_policy, AlphaZeroAdapter, AlphaZeroNet, Game, MonteCarloTree, MoveParameters,
    NetworkBatchedExecutorHandle, TerminationState,
};

async fn make_move<
    TNet1: AlphaZeroNet,
    TNet2: AlphaZeroNet,
    TGame: Game + Clone,
    TAdapter1: AlphaZeroAdapter<TGame, TNet1>,
    TAdapter2: AlphaZeroAdapter<TGame, TNet2>,
    R: Rng,
>(
    samples: usize,
    c_puct: f32,
    temp: f32,
    tree1: &mut MonteCarloTree<TGame, TNet1, TAdapter1>,
    tree2: &mut MonteCarloTree<TGame, TNet2, TAdapter2>,
    rng: &mut R,
) -> (usize, Vec<f32>) {
    tree1.do_simulations(samples, c_puct).await;
    tree2.do_simulations(2, c_puct).await;
    let policy = tree1.get_policy();
    let r#move = sample_policy(&policy, temp, rng);

    tree1.do_move(r#move);
    tree2.do_move(r#move);

    (r#move, policy)
}

pub async fn do_battle<
    TNet1: AlphaZeroNet,
    TNet2: AlphaZeroNet,
    TGame: Game + Clone,
    TAdapter1: AlphaZeroAdapter<TGame, TNet1>,
    TAdapter2: AlphaZeroAdapter<TGame, TNet2>,
    F: FnMut(usize) -> f32,
>(
    start: TGame,
    samples: usize,
    c_puct: f32,
    mut temp: F,
    executor1: NetworkBatchedExecutorHandle<TNet1>,
    executor2: NetworkBatchedExecutorHandle<TNet2>,
) -> Vec<(TGame, Vec<f32>, f32, bool)> {
    let mut tree1 = MonteCarloTree::<TGame, TNet1, TAdapter1>::new(start.clone(), executor1);
    let mut tree2 = MonteCarloTree::<TGame, TNet2, TAdapter2>::new(start.clone(), executor2);
    let mut turn = 0;
    let mut first = true;

    let mut state = start;

    let mut history = vec![];

    let score = loop {
        let moves = match state.get_state() {
            TerminationState::Terminal(v) => break v,
            TerminationState::Moves(moves) => moves,
        };
        let temp = temp(turn);
        let (r#move, policy) = if first {
            make_move(
                samples,
                c_puct,
                temp,
                &mut tree1,
                &mut tree2,
                &mut thread_rng(),
            )
            .await
        } else {
            make_move(
                samples,
                c_puct,
                temp,
                &mut tree2,
                &mut tree1,
                &mut thread_rng(),
            )
            .await
        };

        let new_state = state.make_move(&moves[r#move]);
        history.push((state, policy, 0.0, first));

        state = new_state;
        first ^= moves[r#move].is_player_switch();
        turn += 1;
    };

    for h in &mut history {
        h.2 = if h.3 == first { score } else { 1.0 - score };
    }

    history
}
