use std::{marker::PhantomData, ptr, sync::OnceLock};

use atomic_refcell::AtomicRefCell;
use rand::Rng;
use rand_distr::{multi::Dirichlet, Distribution};

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

#[derive(Clone, Copy, Debug)]
pub enum RootNoise {
    None,
    Dirichlet { alpha: f32, epsilon: f32 },
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

        if self.children.is_empty() {
            return vec![];
        }
        assert!(sm > 0, "No simulations have visited a root move");

        iter.map(move |v| v as f32 / sm as f32).collect::<Vec<_>>()
    }
}

pub struct MonteCarloTree<TGame: Game, TNet: AlphaZeroNet, TAdapter: AlphaZeroAdapter<TGame, TNet>>
{
    root: MonteCarloNode<TGame>,
    executor: NetworkBatchedExecutorHandle<TNet>,
    root_noise: RootNoise,
    root_noise_sample: Option<Box<[f32]>>,
    _p: PhantomData<TAdapter>,
}

impl<TGame: Game, TNet: AlphaZeroNet, TAdapter: AlphaZeroAdapter<TGame, TNet>>
    MonteCarloTree<TGame, TNet, TAdapter>
{
    pub fn new(
        state: TGame,
        executor: NetworkBatchedExecutorHandle<TNet>,
        root_noise: RootNoise,
    ) -> Self {
        if let RootNoise::Dirichlet { alpha, epsilon } = root_noise {
            assert!(alpha.is_finite() && alpha > 0.0);
            assert!(epsilon.is_finite() && (0.0..=1.0).contains(&epsilon));
        }

        let root = MonteCarloNode::new(state);
        Self {
            root,
            executor,
            root_noise,
            root_noise_sample: None,
            _p: PhantomData,
        }
    }

    pub fn get_network_state_estimation(&self) -> Option<f32> {
        Some(self.root.node_state.get()?.value)
    }

    pub fn get_move_stats(&self, r#move: usize) -> Option<(MoveStaticInfo, MoveDynamicInfo)> {
        let state = self.root.node_state.get()?;
        let child = &state.children[r#move];
        Some((child.static_info, *child.dyn_info.borrow()))
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
        assert!(value.is_finite());
        assert_eq!(policy.len(), moves.len());
        assert!(policy
            .iter()
            .all(|value| value.is_finite() && (0.0..=1.0).contains(value)));
        assert!((policy.iter().sum::<f32>() - 1.0).abs() < 1e-4);

        NodeState {
            value,
            is_terminal: false,
            children: moves
                .iter()
                .zip(policy)
                .map(|(r#move, policy)| NodeChild {
                    node: MonteCarloNode::new(state.make_move(r#move)),
                    static_info: MoveStaticInfo {
                        priority: policy,
                        player_switch: r#move.is_player_switch(),
                    },
                    dyn_info: AtomicRefCell::new(MoveDynamicInfo {
                        total_score: 0.0,
                        descends: 0,
                    }),
                })
                .collect::<Vec<_>>()
                .into(),
        }
    }

    async fn initialize_root(&mut self) {
        if self.root.node_state.get().is_none() {
            let state = Self::create_node_state(&mut self.executor, &self.root.game_state).await;
            assert!(self.root.node_state.set(state).is_ok());
        }
    }

    fn initialize_root_noise<R: Rng + ?Sized>(&mut self, rng: &mut R) {
        if self.root_noise_sample.is_some() {
            return;
        }

        let RootNoise::Dirichlet { alpha, .. } = self.root_noise else {
            return;
        };
        let size = self.root.node_state.get().unwrap().children.len();
        let adjustment = if size >= 2 {
            Dirichlet::new(&vec![alpha; size]).unwrap().sample(rng)
        } else {
            vec![1.0; size]
        };
        self.root_noise_sample = Some(adjustment.into());
    }

    pub async fn do_simulations<R: Rng + Send + ?Sized>(
        &mut self,
        samples: usize,
        cpuct: f32,
        rng: &mut R,
    ) {
        assert!(samples > 0, "At least one simulation is required");
        self.initialize_root().await;
        self.initialize_root_noise(rng);

        let mut state_stack = vec![];
        let root_noise_sample = self.root_noise_sample.clone();

        let is_root = |node: &MonteCarloNode<TGame>| ptr::eq(node, &self.root);
        for _ in 0..samples {
            let mut cur = &self.root;
            let mut value = loop {
                let (node_state, created) = 'cl: {
                    if let Some(r) = cur.node_state.get() {
                        break 'cl (r, false);
                    }
                    let state = Self::create_node_state(&mut self.executor, &cur.game_state).await;
                    assert!(cur.node_state.set(state).is_ok());
                    (cur.node_state.get().unwrap(), true)
                };

                if created || node_state.is_terminal {
                    break node_state.value;
                }

                let m = if is_root(cur) {
                    match self.root_noise {
                        RootNoise::None => node_state.pick_next_move(cpuct),
                        RootNoise::Dirichlet { epsilon, .. } => node_state.pick_next_move_root(
                            cpuct,
                            epsilon,
                            root_noise_sample.as_deref().unwrap(),
                        ),
                    }
                } else {
                    node_state.pick_next_move(cpuct)
                };
                cur = &node_state.children[m].node;
                state_stack.push((node_state, m));
            };

            while let Some((state, r#move)) = state_stack.pop() {
                let child = &state.children[r#move];

                if child.static_info.player_switch {
                    value *= -1.;
                }

                let mut dyn_info = child.dyn_info.borrow_mut();
                dyn_info.total_score += value;
                dyn_info.descends += 1;
            }
        }
    }

    pub fn get_policy(&self) -> Vec<f32> {
        self.root.node_state.get().unwrap().get_policy()
    }

    pub fn do_move(&mut self, move_id: usize) {
        self.root = core::mem::take(&mut self.root.node_state.get_mut().unwrap().children)
            .into_vec()
            .swap_remove(move_id)
            .node;
        self.root_noise_sample = None;
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use rand::{rngs::SmallRng, SeedableRng};
    use tch::{Device, Kind, Tensor};

    use crate::alpha_zero::{AlphaZeroAdapter, AlphaZeroNet, ExecutorScope, Game, MoveParameters};

    use super::{MonteCarloTree, RootNoise, TerminationState};

    #[derive(Clone)]
    struct TestGame(Option<f32>);

    #[derive(Clone, Copy)]
    enum TestMove {
        Win,
        Lose,
    }

    impl MoveParameters for TestMove {
        fn is_player_switch(&self) -> bool {
            true
        }
    }

    impl Game for TestGame {
        type Move = TestMove;

        fn get_state(&self) -> TerminationState<Self::Move> {
            match self.0 {
                Some(value) => TerminationState::Terminal(value),
                None => TerminationState::Moves(vec![TestMove::Win, TestMove::Lose].into()),
            }
        }

        fn make_move(&self, r#move: &Self::Move) -> Self {
            match r#move {
                TestMove::Win => Self(Some(-1.0)),
                TestMove::Lose => Self(Some(1.0)),
            }
        }
    }

    struct TestNet;

    impl AlphaZeroNet for TestNet {
        fn forward_t(&self, input: &Tensor, _is_training: bool) -> (Tensor, Tensor) {
            let batch = input.size()[0];
            (
                Tensor::zeros([batch], (Kind::Float, Device::Cpu)),
                Tensor::zeros([batch, 2], (Kind::Float, Device::Cpu)),
            )
        }
    }

    struct TestAdapter;

    impl AlphaZeroAdapter<TestGame, TestNet> for TestAdapter {
        fn convert_game_to_nn_input(_state: &TestGame) -> Tensor {
            Tensor::zeros([1], (Kind::Float, Device::Cpu))
        }

        fn get_estimated_policy(_policy: &Tensor, _moves: &[TestMove]) -> Vec<f32> {
            vec![0.75, 0.25]
        }

        fn convert_policy_to_nn(_policy: &[f32], _moves: &[TestMove]) -> Tensor {
            unreachable!()
        }
    }

    #[tokio::test]
    async fn first_search_initializes_root_and_records_every_simulation() {
        let mut scope = ExecutorScope::new(
            TestNet,
            1,
            1,
            Duration::from_millis(1),
            (Kind::Float, Device::Cpu),
        );
        std::mem::drop(scope.spawn(|handle| async move {
            let mut tree = MonteCarloTree::<TestGame, TestNet, TestAdapter>::new(
                TestGame(None),
                handle,
                RootNoise::None,
            );
            let mut rng = SmallRng::seed_from_u64(1);
            tree.do_simulations(1, 1.0, &mut rng).await;

            (
                tree.get_total_descends().unwrap(),
                tree.get_policy(),
                tree.get_move_stats(0).unwrap().1.get_avg_score(),
            )
        }));

        let (visits, policy, winning_score) = scope.next().await.unwrap();
        scope.join().await;
        assert_eq!(visits, 1);
        assert_eq!(policy, [1.0, 0.0]);
        assert_eq!(winning_score, 1.0);
    }

    #[tokio::test]
    async fn dirichlet_noise_is_reused_for_the_same_root() {
        let mut scope = ExecutorScope::new(
            TestNet,
            1,
            1,
            Duration::from_millis(1),
            (Kind::Float, Device::Cpu),
        );
        std::mem::drop(scope.spawn(|handle| async move {
            let mut tree = MonteCarloTree::<TestGame, TestNet, TestAdapter>::new(
                TestGame(None),
                handle,
                RootNoise::Dirichlet {
                    alpha: 0.1,
                    epsilon: 0.25,
                },
            );
            let mut rng = SmallRng::seed_from_u64(1);
            tree.do_simulations(1, 1.0, &mut rng).await;
            let first_noise = tree.root_noise_sample.clone();
            tree.do_simulations(1, 1.0, &mut rng).await;
            assert_eq!(tree.root_noise_sample, first_noise);
            tree.do_move(0);
            assert!(tree.root_noise_sample.is_none());
        }));

        scope.next().await.unwrap();
        scope.join().await;
    }
}
