use std::{marker::PhantomData, ptr, sync::OnceLock};

use atomic_refcell::AtomicRefCell;
use rand::Rng;
use rand_distr::{Dirichlet, Distribution};

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

struct NodeState<T> {
    value: f32,
    is_terminal: bool,
    children: Vec<(
        MonteCarloNode<T>,
        MoveStaticInfo,
        AtomicRefCell<MoveDynamicInfo>,
    )>,
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
    fn pick_next_move_internal(
        &self,
        c_puct: f32,
        priority_adj: impl Fn(f32, usize) -> f32,
    ) -> usize {
        let total_visits: usize = self
            .children
            .iter()
            .map(|(_, _, d)| d.borrow().descends)
            .sum();
        let sqrt_total_visits = f32::sqrt(total_visits as f32);

        self.children
            .iter()
            .enumerate()
            .map(|(i, (_, MoveStaticInfo { priority, .. }, d))| {
                let MoveDynamicInfo {
                    total_score,
                    descends,
                } = *d.borrow();
                (
                    (if descends != 0 {
                        total_score / descends as f32
                    } else {
                        0.0
                    }) + c_puct
                        * priority_adj(*priority, i)
                        * (sqrt_total_visits / (1 + descends) as f32 + 1e-9),
                    i,
                )
            })
            .max_by(|(a, _), (b, _)| match a.partial_cmp(b) {
                None => panic!("Failed to compare {a} with {b}"),
                Some(res) => res,
            })
            .unwrap()
            .1
    }

    fn pick_next_move(&self, c_puct: f32) -> usize {
        self.pick_next_move_internal(c_puct, |p, _| p)
    }

    fn pick_next_move_root(&self, c_puct: f32, eps: f32, adj: &[f32]) -> usize {
        self.pick_next_move_internal(c_puct, move |p, i| p * (1.0 - eps) + adj[i] * eps)
    }

    fn get_policy(&self) -> Vec<f32> {
        let iter = self.children.iter().map(|(_, _, d)| d.borrow().descends);
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

    async fn create_node_state(
        executor: &mut NetworkBatchedExecutorHandle<TNet>,
        state: &TGame,
    ) -> NodeState<TGame> {
        let moves = match state.get_state() {
            TerminationState::Terminal(val) => {
                return NodeState {
                    value: val,
                    is_terminal: true,
                    children: vec![],
                };
            }
            TerminationState::Moves(moves) => moves,
        };
        // println!("Found target state in {:?}", Instant::now() - start);
        let (value, policy) = executor
            .execute(TAdapter::convert_game_to_nn_input(state))
            .await;
        let value = f32::try_from(value).unwrap();
        let policy = TAdapter::get_estimated_policy(&policy, &moves);

        let node_state = NodeState {
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
        };
        node_state
    }

    pub async fn do_simulations<R: Rng + Send + ?Sized>(
        &mut self,
        samples: usize,
        cpuct: f32,
        rng: &mut R,
    ) {
        let mut state_stack = vec![];
        let mut distr = None;
        let is_root = |node: &MonteCarloNode<TGame>| ptr::eq(node, &self.root);
        for _ in 0..samples {
            let mut cur = &self.root;
            // let start = Instant::now();
            let mut value = loop {
                let (node_state, created) = 'cl: {
                    if let Some(r) = cur.node_state.get() {
                        break 'cl (r, false);
                    }
                    let state = Self::create_node_state(&mut self.executor, &cur.game_state).await;
                    cur.node_state.set(state).map_err(|_| ()).unwrap();
                    (cur.node_state.get().unwrap(), true)
                };
                if distr.is_none() && is_root(cur) && node_state.children.len() >= 2 {
                    distr = Some(Dirichlet::new_with_size(0.1, node_state.children.len()).unwrap());
                }

                if created || node_state.is_terminal {
                    break node_state.value;
                }

                let m = if is_root(cur) && node_state.children.len() >= 2 {
                    let adj = distr.as_ref().unwrap().sample(rng);
                    node_state.pick_next_move_root(cpuct, 0.25, &adj)
                } else {
                    node_state.pick_next_move(cpuct)
                };
                cur = &node_state.children[m].0;
                state_stack.push((node_state, m));
            };

            while let Some((state, r#move)) = state_stack.pop() {
                let child = &state.children[r#move];

                if child.1.player_switch {
                    value *= -1.0;
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
