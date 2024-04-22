use std::{future::Future, marker::PhantomData};

use crate::alpha_zero::TerminationState;

use super::{AlphaZeroAdapter, AlphaZeroNet, Game, MoveParameters, NetworkBatchedExecutorHandle};

#[derive(Clone, Copy, Debug)]
struct MoveDynamicInfo {
    total_score: f32,
    descends: usize,
}

#[derive(Clone, Copy, Debug)]
struct MoveStaticInfo {
    priority: f32,
    player_switch: bool,
}

struct NodeState {
    value: f32,
    is_terminal: bool,
    children: Vec<(usize, MoveStaticInfo, MoveDynamicInfo)>,
}

struct MonteCarloNode<T> {
    game_state: T,
    node_state: Option<NodeState>,
}

impl<T> MonteCarloNode<T> {
    async fn get_state<'a, F, TAdditional>(
        &'a mut self,
        calc_state: impl FnOnce(&'a T) -> F,
    ) -> Option<TAdditional>
    where
        F: Future<Output = (NodeState, TAdditional)> + 'a,
    {
        if self.node_state.is_some() {
            return None;
        }
        let (state, additional) = calc_state(&self.game_state).await;
        self.node_state = Some(state);
        Some(additional)
    }
}

impl NodeState {
    fn pick_next_move(&self, c: f32) -> usize {
        let total_visits: usize = self
            .children
            .iter()
            .map(|(_, _, MoveDynamicInfo { descends, .. })| descends)
            .sum();
        let sqrt_total_visits = f32::sqrt(total_visits as f32);

        self.children
            .iter()
            .map(
                |(
                    _,
                    MoveStaticInfo { priority, .. },
                    MoveDynamicInfo {
                        total_score,
                        descends,
                    },
                )| {
                    (if descends != &0 {
                        total_score / *descends as f32
                    } else {
                        0.0
                    }) + c * priority * sqrt_total_visits / (1 + total_visits) as f32
                },
            )
            .enumerate()
            .map(|(i, v)| (v, i))
            .max_by(|(a, _), (b, _)| match a.partial_cmp(b) {
                None => panic!("Failed to compare {a} with {b}"),
                Some(res) => res,
            })
            .unwrap()
            .1
    }
}

pub struct MonteCarloTree<TGame: Game, TNet: AlphaZeroNet, TAdapter: AlphaZeroAdapter<TGame, TNet>>
{
    nodes: Vec<MonteCarloNode<TGame>>,
    executor: NetworkBatchedExecutorHandle<TNet>,
    // inner_bytes_size: usize,
    _p: PhantomData<TAdapter>,
}

impl<TGame: Game, TNet: AlphaZeroNet, TAdapter: AlphaZeroAdapter<TGame, TNet>>
    MonteCarloTree<TGame, TNet, TAdapter>
{
    pub fn new(state: TGame, executor: NetworkBatchedExecutorHandle<TNet>) -> Self {
        let mut nodes = Vec::with_capacity(1024);
        nodes.push(MonteCarloNode {
            game_state: state,
            node_state: None,
        });
        Self {
            nodes,
            executor,
            // inner_bytes_size: 0,
            _p: PhantomData,
        }
    }

    pub async fn do_simulations(&mut self, n: usize, cpuct: f32, state: usize) {
        let mut state_stack = vec![];
        for _ in 0..n {
            let mut cur = state;
            let mut value;
            // let start = Instant::now();
            loop {
                let nodes_cnt = self.nodes.len();
                let moves = self.nodes[cur]
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
                        // println!("Found target state in {:?}", Instant::now() - start);
                        let (value, policy) = self
                            .executor
                            .execute(TAdapter::convert_game_to_nn_input(state))
                            .await;
                        let value = f32::try_from(value).unwrap();
                        let policy = TAdapter::get_estimated_policy(policy, &moves);

                        let node_state = NodeState {
                            value,
                            is_terminal: false,
                            children: moves
                                .iter()
                                .zip(policy)
                                .enumerate()
                                .map(|(i, (r#move, policy))| {
                                    (
                                        nodes_cnt + i,
                                        MoveStaticInfo {
                                            priority: policy,
                                            player_switch: r#move.is_player_switch(),
                                        },
                                        MoveDynamicInfo {
                                            total_score: 0.0,
                                            descends: 0,
                                        },
                                    )
                                })
                                .collect(),
                        };
                        (node_state, moves)
                    })
                    .await;

                let mut finish = self.nodes[cur].node_state.as_ref().unwrap().is_terminal;
                if let Some(moves) = moves {
                    // self.inner_bytes_size += bytes;
                    self.nodes.reserve(moves.len());
                    for r#move in &moves {
                        self.nodes.push(MonteCarloNode {
                            game_state: self.nodes[cur].game_state.make_move(r#move),
                            node_state: None,
                        });
                    }
                    finish = true;
                }
                let node_state = self.nodes[cur].node_state.as_ref().unwrap();
                if finish {
                    value = node_state.value;
                    break;
                }

                let m = node_state.pick_next_move(cpuct);
                state_stack.push((cur, m));
                cur = node_state.children[m].0;
            }

            while let Some((state, r#move)) = state_stack.pop() {
                let node_state = self.nodes[state].node_state.as_mut().unwrap();

                if node_state.children[r#move].1.player_switch {
                    value = 1.0 - value;
                }

                let dyn_params = &mut node_state.children[r#move].2;
                dyn_params.total_score += value;
                dyn_params.descends += 1;
            }
        }
    }

    pub fn get_policy(&self, state: usize) -> Vec<f32> {
        let node = &self.nodes[state];
        let state = node.node_state.as_ref().unwrap();
        let iter = state
            .children
            .iter()
            .map(|(_, _, MoveDynamicInfo { descends, .. })| *descends);
        let sm: usize = iter.clone().sum();

        iter.map(move |v| v as f32 / sm as f32).collect::<Vec<_>>()
    }

    pub fn get_next_state(&self, state: usize, move_id: usize) -> usize {
        self.nodes[state].node_state.as_ref().unwrap().children[move_id].0
    }
}
