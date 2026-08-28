use std::{borrow::Borrow, convert::identity, marker::PhantomData, ptr, sync::OnceLock};

use atomic_refcell::AtomicRefCell;
use rand::Rng;
use rand_distr::{Dirichlet, Distribution};

use crate::alpha_zero::TerminationState;

use super::{AlphaZeroAdapter, AlphaZeroNet, Game, MoveParameters, NetworkBatchedExecutorHandle};

#[derive(Clone, Copy, Debug)]
pub struct MoveDynamicInfo {
    pub total_score: f32,
    pub descends: usize,
}

impl MoveDynamicInfo {
    pub fn get_avg_score(&self) -> f32 {
        if self.descends != 0 {
            self.total_score / self.descends as f32
        } else {
            0.0
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct MoveStaticInfo {
    pub priority: f32,
    pub player_switch: bool,
}

struct NodeChild<T> {
    node: MonteCarloNode<T>,
    static_info: MoveStaticInfo,
    dyn_info: AtomicRefCell<MoveDynamicInfo>,
}

struct NodeState<T> {
    value: f32,
    is_terminal: bool,
    children: Box<[NodeChild<T>]>,
}

struct MonteCarloNode<T> {
    game_state: T,
    node_state: OnceLock<NodeState<T>>,
}

impl<T> MonteCarloNode<T> {
    fn new(state: T) -> Self {
        Self {
            game_state: state,
            node_state: OnceLock::new(),
        }
    }
}

impl<T> NodeState<T> {
    fn pick_next_move_internal<F: Fn(f32, usize) -> f32>(
        &self,
        c_puct: f32,
        priority_adj: F,
    ) -> usize {
        let sqrt_total_visits = f32::sqrt(self.count_total_visits() as f32);

        self.children
            .iter()
            .enumerate()
            .map(
                |(
                    i,
                    NodeChild {
                        node: _,
                        static_info: MoveStaticInfo { priority, .. },
                        dyn_info,
                    },
                )| {
                    let dyn_info = dyn_info.borrow();
                    (
                        dyn_info.get_avg_score()
                            + c_puct
                                * priority_adj(*priority, i)
                                * (sqrt_total_visits / (1 + dyn_info.descends) as f32 + 1e-9),
                        i,
                    )
                },
            )
            .max_by(|(a, _), (b, _)| match a.partial_cmp(b) {
                None => panic!("Failed to compare {a} with {b}"),
                Some(res) => res,
            })
            .unwrap()
            .1
    }

    fn get_max_visits(&self) -> usize {
        self.children
            .iter()
            .map(|child| child.dyn_info.borrow().descends)
            .max()
            .unwrap_or(0)
    }

    fn count_total_visits(&self) -> usize {
        self.children
            .iter()
            .map(|child| child.dyn_info.borrow().descends)
            .sum()
    }

    fn pick_next_move(&self, c_puct: f32) -> usize {
        self.pick_next_move_internal(c_puct, |p, _| p)
    }

    fn pick_next_move_root(&self, c_puct: f32, eps: f32, adj: &[f32]) -> usize {
        self.pick_next_move_internal(c_puct, move |p, i| p * (1.0 - eps) + adj[i] * eps)
    }

    fn get_policy(&self) -> Vec<f32> {
        let iter = self
            .children
            .iter()
            .map(|child| child.dyn_info.borrow().descends);
        let sm: usize = iter.clone().sum();

        iter.map(move |v| v as f32 / sm as f32).collect::<Vec<_>>()
    }
}

pub struct MonteCarloTree<TGame: Game, TNet: AlphaZeroNet, TAdapter: AlphaZeroAdapter<TGame, TNet>>
{
    root: MonteCarloNode<TGame>,
    executor: NetworkBatchedExecutorHandle<TNet>,
    _p: PhantomData<TAdapter>,
}

impl<TGame: Game, TNet: AlphaZeroNet, TAdapter: AlphaZeroAdapter<TGame, TNet>>
    MonteCarloTree<TGame, TNet, TAdapter>
{
    pub fn new(state: TGame, executor: NetworkBatchedExecutorHandle<TNet>) -> Self {
        let root = MonteCarloNode::new(state);
        Self {
            root,
            executor,
            _p: PhantomData,
        }
    }

    pub fn get_network_state_estimation(&self) -> Option<f32> {
        Some(self.root.node_state.get()?.value)
    }

    pub fn get_move_stats(&self, r#move: usize) -> Option<(MoveStaticInfo, MoveDynamicInfo)> {
        let state = self.root.node_state.get()?;
        let child = &state.children[r#move];
        Some((child.1, *child.2.borrow()))
    }

    pub fn get_total_descends(&self) -> Option<usize> {
        Some(self.root.node_state.get()?.count_total_visits())
    }

    pub fn most_descends(&self) -> Option<usize> {
        Some(self.root.node_state.get()?.get_max_visits())
    }

    async fn create_node_state(
        executor: &mut NetworkBatchedExecutorHandle<TNet>,
        state: &TGame,
    ) -> NodeState<TGame> {
        let moves = match state.get_state() {
            TerminationState::Terminal(val) => {
                return NodeState {
                    value: val,
                    is_terminal: true,
                    children: Box::from(vec![]),
                };
            }
            TerminationState::Moves(moves) => moves,
        };
        let (value, policy) = executor
            .execute(TAdapter::convert_game_to_nn_input(state))
            .await;
        let value = f32::try_from(value).unwrap();
        let policy = TAdapter::get_estimated_policy(&policy, &moves);

        NodeState {
            value,
            is_terminal: false,
            children: moves
                .iter()
                .zip(policy)
                .map(|(r#move, policy)| {
                    assert!(policy >= 0. && policy <= 1.);
                    (
                        MonteCarloNode::new(state.make_move(r#move)),
                        MoveStaticInfo {
                            priority: policy,
                            player_switch: r#move.is_player_switch(),
                        },
                        AtomicRefCell::new(MoveDynamicInfo {
                            total_score: 0.0,
                            descends: 0,
                        }),
                    )
                })
                .collect(),
        }
    }

    pub async fn do_simulations<R: Rng + Send + ?Sized>(
        &mut self,
        samples: usize,
        cpuct: f32,
        rng: &mut R,
    ) {
        let mut state_stack = vec![];
        let adjustment = {
            let root_state = self.root.node_state.get().unwrap();
            let size = root_state.children.len();
            if size >= 2 {
                let distr = Dirichlet::new_with_size(0.1, size).unwrap();
                distr.sample(rng)
            } else {
                vec![1.; size]
            }
        };

        let is_root = |node: &MonteCarloNode<TGame>| ptr::eq(node, &self.root);
        for _ in 0..samples {
            let mut cur = &self.root;
            let mut value = loop {
                let (node_state, created) = 'cl: {
                    if let Some(r) = cur.node_state.get() {
                        break 'cl (r, false);
                    }
                    let state = Self::create_node_state(&mut self.executor, &cur.game_state).await;
                    let r = cur.node_state.try_insert(state).unwrap();
                    (r, true)
                };

                if created || node_state.is_terminal {
                    break node_state.value;
                }

                let m = if is_root(cur) {
                    node_state.pick_next_move_root(cpuct, 0.25, &adjustment)
                } else {
                    node_state.pick_next_move(cpuct)
                };
                cur = &node_state.children[m].0;
                state_stack.push((node_state, m));
            };

            while let Some((state, r#move)) = state_stack.pop() {
                let child = &state.children[r#move];

                if child.1.player_switch {
                    value *= -0.97;
                }

                let mut dyn_info = child.2.borrow_mut();
                dyn_info.total_score += value;
                dyn_info.descends += 1;
            }
        }
    }

    pub fn get_policy(&self) -> Vec<f32> {
        self.root.node_state.get().unwrap().get_policy()
    }

    pub fn do_move(&mut self, move_id: usize) {
        let root = self
            .root
            .node_state
            .get_mut()
            .unwrap()
            .children
            .swap_remove(move_id)
            .0;
        self.root = root;
    }
}
