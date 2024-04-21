use std::{future::Future, marker::PhantomData};

use tch::{Device, Kind};

use crate::alpha_zero::TerminationState;

use super::{AlphaZeroAdapter, AlphaZeroNet, Game, NetworkBatchedExecutorHandle};

#[derive(Clone, Copy, Debug)]
struct MoveStatistics {
    total_score: f32,
    descends: usize,
}

#[derive(Clone, Copy, Debug)]
struct MoveParameters {
    priority: f32,
    player_switch: bool,
}

struct NodeState {
    value: f32,
    is_terminal: bool,
    children: Vec<(usize, MoveParameters, MoveStatistics)>,
}

struct MonteCarloNode<T> {
    game_state: T,
    node_state: Option<NodeState>,
}

impl<T> MonteCarloNode<T> {
    async fn get_state<'a, F, TAdditional>(
        &'a mut self,
        calc_state: impl FnOnce(&'a T) -> F,
    ) -> (&mut NodeState, Option<TAdditional>)
    where
        F: Future<Output = (NodeState, TAdditional)> + 'a,
    {
        if let Some(ref mut v) = self.node_state {
            return (v, None);
        }
        let (state, additional) = calc_state(&self.game_state).await;
        (self.node_state.insert(state), Some(additional))
    }
}

impl NodeState {
    async fn pick_next_move(&self, c: f32) -> usize {
        let total_visits: usize = self
            .children
            .iter()
            .map(|(_, _, MoveStatistics { descends, .. })| descends)
            .sum();
        let sqrt_total_visits = f32::sqrt(total_visits as f32);

        self.children
            .iter()
            .map(
                |(
                    _,
                    MoveParameters { priority, .. },
                    MoveStatistics {
                        total_score,
                        descends,
                    },
                )| {
                    total_score / *descends as f32
                        + c * priority * sqrt_total_visits / (1 + total_visits) as f32
                },
            )
            .enumerate()
            .map(|(i, v)| (v, i))
            .max_by(|(a, _), (b, _)| a.partial_cmp(b).unwrap())
            .unwrap()
            .1
    }
}

pub struct MonteCarloTree<TGame: Game, TNet: AlphaZeroNet, TAdapter: AlphaZeroAdapter<TGame, TNet>>
{
    nodes: Vec<MonteCarloNode<TGame>>,
    executor: NetworkBatchedExecutorHandle<TNet>,
    options: (Kind, Device),
    _p: PhantomData<TAdapter>,
}

impl<TGame: Game, TNet: AlphaZeroNet, TAdapter: AlphaZeroAdapter<TGame, TNet>>
    MonteCarloTree<TGame, TNet, TAdapter>
{
    pub fn new(
        state: TGame,
        executor: NetworkBatchedExecutorHandle<TNet>,
        options: (Kind, Device),
    ) -> Self {
        Self {
            nodes: vec![MonteCarloNode {
                game_state: state,
                node_state: None,
            }],
            executor,
            options,
            _p: PhantomData,
        }
    }

    pub async fn do_simulations(&mut self, n: usize, cpuct: f32)
    where
        TGame::Move: Clone,
    {
        // let mut state_stack = vec![];
        let mut cur = 0;
        let mut value = 0f32;
        loop {
            let (node_state, moves): (_, Option<Vec<()>>) = self.nodes[cur]
                .get_state(|state: &TGame| async {
                    let moves = match state.get_state() {
                        TerminationState::Terminal(val) => {
                            return (
                                NodeState {
                                    value: val,
                                    is_terminal: true,
                                    children: vec![],
                                },
                                vec![],
                            );
                        }
                        TerminationState::Moves(moves) => moves,
                    };
                    let (value, policy) = self
                        .executor
                        .execute(TAdapter::convert_game_to_nn_input(state, self.options))
                        .await;
                    let value = f32::try_from(value).unwrap();
                    let policy = TAdapter::get_estimated_policy(policy, &moves);
                    todo!()
                })
                .await;
            if node_state.is_terminal {
                value = node_state.value;
                break;
            }
        }
    }
}
