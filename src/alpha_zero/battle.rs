use rand::{rngs::SmallRng, Rng, SeedableRng};
use std::future::Future;

use super::{
    apply_temperature, sample_policy, AlphaZeroAdapter, AlphaZeroNet, Game, MonteCarloTree,
    MoveParameters, NetworkBatchedExecutorHandle, RootNoise, TerminationState,
};

pub trait Agent<TGame: Game> {
    fn make_move<'a>(&'a mut self) -> impl Future<Output = usize> + Send + 'a;
    fn record_opponents_move<'a>(
        &'a mut self,
        r#move: usize,
    ) -> impl Future<Output = ()> + Send + 'a;
}

pub struct MCTSAgent<
    TGame: Game,
    TNet: AlphaZeroNet,
    TAdapter: AlphaZeroAdapter<TGame, TNet>,
    R: Rng,
    F: FnMut(usize) -> f32,
> {
    tree: MonteCarloTree<TGame, TNet, TAdapter>,
    c_puct: f32,
    samples: usize,
    rng: R,
    temp: F,
    r#move: usize,
}

impl<
        TGame: Game + Send + Sync,
        TNet: AlphaZeroNet + Send,
        TAdapter: AlphaZeroAdapter<TGame, TNet> + Send,
        R: Rng + Send,
        F: FnMut(usize) -> f32 + Send,
    > Agent<TGame> for MCTSAgent<TGame, TNet, TAdapter, R, F>
where
    TGame::Move: Send,
{
    fn make_move<'a>(&'a mut self) -> impl Future<Output = usize> + Send + 'a {
        async {
            self.tree
                .do_simulations(self.samples, self.c_puct, &mut self.rng)
                .await;
            let policy = apply_temperature(&self.tree.get_policy(), (self.temp)(self.r#move));
            let r#move = sample_policy(&policy, &mut self.rng);
            self.r#move += 1;
            self.tree.do_move(r#move);
            r#move
        }
    }

    fn record_opponents_move<'a>(
        &'a mut self,
        r#move: usize,
    ) -> impl Future<Output = ()> + Send + 'a {
        async move {
            self.tree
                .do_simulations(2, self.c_puct, &mut self.rng)
                .await;
            self.tree.do_move(r#move);
        }
    }
}

async fn make_move<
    TNet1: AlphaZeroNet,
    TNet2: AlphaZeroNet,
    TGame: Game + Clone,
    TAdapter1: AlphaZeroAdapter<TGame, TNet1>,
    TAdapter2: AlphaZeroAdapter<TGame, TNet2>,
    R: Rng + Send + ?Sized,
>(
    samples: usize,
    c_puct: f32,
    temp: f32,
    tree1: &mut MonteCarloTree<TGame, TNet1, TAdapter1>,
    tree2: &mut MonteCarloTree<TGame, TNet2, TAdapter2>,
    rng: &mut R,
) -> (usize, Vec<f32>) {
    tree1.do_simulations(samples, c_puct, rng).await;
    tree2.do_simulations(2, c_puct, rng).await;
    let policy = apply_temperature(&tree1.get_policy(), temp);
    let r#move = sample_policy(&policy, rng);

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
    let mut tree1 =
        MonteCarloTree::<TGame, TNet1, TAdapter1>::new(start.clone(), executor1, RootNoise::None);
    let mut tree2 =
        MonteCarloTree::<TGame, TNet2, TAdapter2>::new(start.clone(), executor2, RootNoise::None);
    let mut turn = 0;
    let mut first = true;

    let mut state = start;

    let mut history = vec![];

    let mut rng = SmallRng::from_rng(&mut rand::rng());

    let score = loop {
        let moves = match state.get_state() {
            TerminationState::Terminal(v) => break v,
            TerminationState::Moves(moves) => moves,
        };
        let temp = temp(turn);
        let (r#move, policy) = if first {
            make_move(samples, c_puct, temp, &mut tree1, &mut tree2, &mut rng).await
        } else {
            make_move(samples, c_puct, temp, &mut tree2, &mut tree1, &mut rng).await
        };

        let new_state = state.make_move(&moves[r#move]);
        history.push((state, policy, 0.0, first));

        state = new_state;
        first ^= moves[r#move].is_player_switch();
        turn += 1;
    };

    for h in &mut history {
        h.2 = if h.3 == first { score } else { -score };
    }

    history
}
