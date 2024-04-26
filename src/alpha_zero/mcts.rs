use std::{marker::PhantomData, sync::OnceLock};

use atomic_refcell::AtomicRefCell;

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
    fn pick_next_move(&self, c_puct: f32) -> usize {
        let total_visits: usize = self
            .children
            .iter()
            .map(|(_, _, d)| d.borrow().descends)
            .sum();
        let sqrt_total_visits = f32::sqrt(total_visits as f32);

        self.children
            .iter()
            .map(|(_, MoveStaticInfo { priority, .. }, d)| {
                let MoveDynamicInfo {
                    total_score,
                    descends,
                } = *d.borrow();
                (if descends != 0 {
                    total_score / descends as f32
                } else {
                    0.0
                }) + c_puct * priority * (sqrt_total_visits / (1 + descends) as f32 + 1e-9)
            })
            .enumerate()
            .map(|(i, v)| (v, i))
            .max_by(|(a, _), (b, _)| match a.partial_cmp(b) {
                None => panic!("Failed to compare {a} with {b}"),
                Some(res) => res,
            })
            .unwrap()
            .1
    }

    fn get_policy(&self) -> Vec<f32> {
        // println!("{:?}", self.children);
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

    pub async fn do_simulations(&mut self, samples: usize, cpuct: f32) {
        let mut state_stack = vec![];
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

                if created || node_state.is_terminal {
                    break node_state.value;
                }

                let m = node_state.pick_next_move(cpuct);
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
