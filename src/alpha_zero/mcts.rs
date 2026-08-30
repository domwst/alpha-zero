use std::{cell::SyncUnsafeCell, ptr, sync::OnceLock};

use anyhow::{Result, ensure};
use rand::Rng;
use rand_distr::{Distribution, multi::Dirichlet};

use crate::alpha_zero::TerminationState;

use super::{Game, MoveParameters, PositionEvaluation, PositionEvaluator, TurnChange};

#[derive(Clone, Copy, Debug)]
pub struct MoveDynamicInfo {
    pub total_score: f32,
    pub descends: u32,
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

#[derive(Clone, Copy)]
pub struct MoveStaticInfoRepr {
    blob: u32,
}

const SIGN_BIT: u32 = 1 << 31;

impl From<MoveStaticInfo> for MoveStaticInfoRepr {
    fn from(value: MoveStaticInfo) -> Self {
        Self {
            blob: (value.priority.to_bits() & !SIGN_BIT) | (value.turn_change as u32 * SIGN_BIT),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct MoveStaticInfo {
    pub priority: f32,
    pub turn_change: TurnChange,
}

impl From<MoveStaticInfoRepr> for MoveStaticInfo {
    fn from(value: MoveStaticInfoRepr) -> Self {
        Self {
            priority: f32::from_bits(value.blob & !SIGN_BIT),
            turn_change: match value.blob & SIGN_BIT {
                0 => TurnChange::SamePlayer,
                SIGN_BIT => TurnChange::SwitchPlayer,
                _ => unreachable!(),
            },
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum RootNoise {
    None,
    Dirichlet { alpha: f32, epsilon: f32 },
}

struct NodeChild<T: Game> {
    action: T::Move,
    node: MonteCarloNode<T>,
    static_info: MoveStaticInfoRepr,
    dyn_info: SyncUnsafeCell<MoveDynamicInfo>,
}

impl<T: Game> NodeChild<T> {
    fn static_info(&self) -> MoveStaticInfo {
        MoveStaticInfo::from(self.static_info)
    }
}

struct NodeState<T: Game> {
    value: f32,
    is_terminal: bool,
    children: Box<[NodeChild<T>]>,
}

struct MonteCarloNode<T: Game> {
    node_state: OnceLock<NodeState<T>>,
}

impl<T: Game> MonteCarloNode<T> {
    fn new() -> Self {
        Self {
            node_state: OnceLock::new(),
        }
    }
}

impl<T: Game> NodeState<T> {
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
                        action: _,
                        node: _,
                        static_info,
                        dyn_info,
                    },
                )| {
                    let priority = MoveStaticInfo::from(*static_info).priority;
                    let dyn_info = unsafe { &*dyn_info.get() };
                    (
                        dyn_info.get_avg_score()
                            + c_puct
                                * priority_adj(priority, i)
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
            .map(|child| unsafe { *child.dyn_info.get() }.descends)
            .max()
            .unwrap_or(0) as _
    }

    fn count_total_visits(&self) -> usize {
        self.children
            .iter()
            .map(|child| unsafe { *child.dyn_info.get() }.descends)
            .sum::<u32>() as _
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
            .map(|child| unsafe { *child.dyn_info.get() }.descends);
        let sm: u32 = iter.clone().sum();

        if self.children.is_empty() {
            return vec![];
        }
        assert!(sm > 0, "No simulations have visited a root node");

        iter.map(move |v| v as f32 / sm as f32).collect::<Vec<_>>()
    }
}

pub struct MonteCarloTree<TGame: Game, Evaluator: PositionEvaluator<TGame>> {
    root_game_state: TGame,
    root: MonteCarloNode<TGame>,
    evaluator: Evaluator,
    root_noise: RootNoise,
    root_noise_sample: Option<Box<[f32]>>,
}

impl<TGame, Evaluator> MonteCarloTree<TGame, Evaluator>
where
    TGame: Game + Clone + PartialEq + Send + Sync,
    TGame::Move: Clone + PartialEq + Send + Sync,
    Evaluator: PositionEvaluator<TGame> + Send + Sync,
{
    pub fn new(state: TGame, evaluator: Evaluator, root_noise: RootNoise) -> Self {
        if let RootNoise::Dirichlet { alpha, epsilon } = root_noise {
            assert!(alpha.is_finite() && alpha > 0.0);
            assert!(epsilon.is_finite() && (0.0..=1.0).contains(&epsilon));
        }

        let root = MonteCarloNode::new();
        Self {
            root_game_state: state,
            root,
            evaluator,
            root_noise,
            root_noise_sample: None,
        }
    }

    pub fn get_network_state_estimation(&self) -> Option<f32> {
        Some(self.root.node_state.get()?.value)
    }

    pub fn get_move_stats(&self, r#move: usize) -> Option<(MoveStaticInfo, MoveDynamicInfo)> {
        let state = self.root.node_state.get()?;
        let child = &state.children[r#move];
        Some((child.static_info(), unsafe { *child.dyn_info.get() }))
    }

    pub fn get_total_descends(&self) -> Option<usize> {
        Some(self.root.node_state.get()?.count_total_visits())
    }

    pub fn most_descends(&self) -> Option<usize> {
        Some(self.root.node_state.get()?.get_max_visits())
    }

    async fn create_node_state(evaluator: &Evaluator, state: &TGame) -> Result<NodeState<TGame>> {
        let moves = match state.get_state() {
            TerminationState::Terminal(val) => {
                return Ok(NodeState {
                    value: val,
                    is_terminal: true,
                    children: Box::from(vec![]),
                });
            }
            TerminationState::Moves(moves) => moves,
        };
        let evaluation = evaluator.evaluate(state, &moves).await?;
        evaluation.validate_for(moves.len())?;
        let PositionEvaluation {
            value,
            legal_policy,
        } = evaluation;

        Ok(NodeState {
            value,
            is_terminal: false,
            children: moves
                .iter()
                .zip(legal_policy)
                .map(|(r#move, policy)| NodeChild {
                    action: r#move.clone(),
                    node: MonteCarloNode::new(),
                    static_info: MoveStaticInfo {
                        priority: policy,
                        turn_change: r#move.turn_change(),
                    }
                    .into(),
                    dyn_info: SyncUnsafeCell::new(MoveDynamicInfo {
                        total_score: 0.0,
                        descends: 0,
                    }),
                })
                .collect::<Vec<_>>()
                .into(),
        })
    }

    async fn initialize_root(&mut self) -> Result<()> {
        if self.root.node_state.get().is_none() {
            let state = Self::create_node_state(&self.evaluator, &self.root_game_state).await?;
            assert!(self.root.node_state.set(state).is_ok());
        }
        Ok(())
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
    ) -> Result<()> {
        assert!(samples > 0, "At least one simulation is required");
        self.initialize_root().await?;
        self.initialize_root_noise(rng);

        let mut state_stack = vec![];
        let root_noise_sample = self.root_noise_sample.clone();

        let is_root = |node: &MonteCarloNode<TGame>| ptr::eq(node, &self.root);
        for _ in 0..samples {
            let mut cur = &self.root;
            let mut cur_game_state = self.root_game_state.clone();

            let mut value = loop {
                let (node_state, created) = 'cl: {
                    if let Some(r) = cur.node_state.get() {
                        break 'cl (r, false);
                    }
                    let state = Self::create_node_state(&self.evaluator, &cur_game_state).await?;
                    assert!(cur.node_state.set(state).is_ok());
                    (cur.node_state.get().unwrap(), true)
                };

                if created || node_state.is_terminal {
                    break node_state.value;
                }

                let m = if is_root(cur)
                    && let RootNoise::Dirichlet { epsilon, .. } = self.root_noise
                {
                    node_state.pick_next_move_root(
                        cpuct,
                        epsilon,
                        root_noise_sample.as_deref().unwrap(),
                    )
                } else {
                    node_state.pick_next_move(cpuct)
                };

                let child = &node_state.children[m];
                cur_game_state = cur_game_state.make_move(&child.action);
                cur = &child.node;
                state_stack.push((node_state, m));
            };

            while let Some((state, r#move)) = state_stack.pop() {
                let child = &state.children[r#move];

                if child.static_info().turn_change.switches_player() {
                    value *= -1.;
                }

                let dyn_info = unsafe { &mut *child.dyn_info.get() };
                dyn_info.total_score += value;
                dyn_info.descends += 1;
            }
        }
        Ok(())
    }

    pub fn get_policy(&self) -> Vec<f32> {
        self.root.node_state.get().unwrap().get_policy()
    }

    pub fn matches_position(&self, state: &TGame, moves: &[TGame::Move]) -> bool {
        if &self.root_game_state != state {
            return false;
        }
        self.root.node_state.get().is_none_or(|node_state| {
            node_state.children.len() == moves.len()
                && node_state
                    .children
                    .iter()
                    .zip(moves)
                    .all(|(child, action)| &child.action == action)
        })
    }

    pub fn advance(
        &mut self,
        move_id: usize,
        action: &TGame::Move,
        next_state: TGame,
    ) -> Result<()> {
        ensure!(
            self.root_game_state.make_move(action) == next_state,
            "MCTS successor does not match the authoritative game state"
        );

        let next_root = if let Some(state) = self.root.node_state.get_mut() {
            ensure!(
                move_id < state.children.len(),
                "move index {move_id} is outside the MCTS root"
            );
            ensure!(
                &state.children[move_id].action == action,
                "applied action does not match MCTS move index {move_id}"
            );
            Some(
                core::mem::take(&mut state.children)
                    .into_vec()
                    .swap_remove(move_id)
                    .node,
            )
        } else {
            None
        };

        self.root_game_state = next_state;
        self.root = next_root.unwrap_or_else(|| MonteCarloNode::new());
        self.root_noise_sample = None;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::{
        future::{Future, ready},
        sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        },
    };

    use rand::{SeedableRng, rngs::SmallRng};

    use crate::alpha_zero::{Game, MoveParameters, PositionEvaluation, PositionEvaluator};

    use super::{MonteCarloTree, RootNoise, TerminationState, TurnChange};

    #[derive(Clone, PartialEq)]
    struct TestGame(Option<f32>);

    #[derive(Clone, Copy, PartialEq)]
    enum TestMove {
        Win,
        Lose,
    }

    impl MoveParameters for TestMove {
        fn turn_change(&self) -> TurnChange {
            TurnChange::SwitchPlayer
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

    struct TestEvaluator;

    impl PositionEvaluator<TestGame> for TestEvaluator {
        fn evaluate<'a>(
            &'a self,
            _state: &'a TestGame,
            _moves: &'a [TestMove],
        ) -> impl Future<Output = anyhow::Result<PositionEvaluation>> + Send + 'a {
            ready(Ok(PositionEvaluation {
                value: 0.0,
                legal_policy: vec![0.75, 0.25],
            }))
        }
    }

    #[derive(Clone, PartialEq)]
    struct DepthGame(usize);

    #[derive(Clone, Copy, PartialEq)]
    struct Step;

    impl MoveParameters for Step {
        fn turn_change(&self) -> TurnChange {
            TurnChange::SwitchPlayer
        }
    }

    impl Game for DepthGame {
        type Move = Step;

        fn get_state(&self) -> TerminationState<Self::Move> {
            if self.0 == 0 {
                TerminationState::Terminal(-1.0)
            } else {
                TerminationState::Moves(vec![Step].into())
            }
        }

        fn make_move(&self, _move: &Self::Move) -> Self {
            Self(self.0 - 1)
        }
    }

    #[derive(Clone)]
    struct CountingEvaluator(Arc<AtomicUsize>);

    impl PositionEvaluator<DepthGame> for CountingEvaluator {
        fn evaluate<'a>(
            &'a self,
            _state: &'a DepthGame,
            _moves: &'a [Step],
        ) -> impl Future<Output = anyhow::Result<PositionEvaluation>> + Send + 'a {
            self.0.fetch_add(1, Ordering::Relaxed);
            ready(Ok(PositionEvaluation {
                value: 0.0,
                legal_policy: vec![1.0],
            }))
        }
    }

    #[tokio::test]
    async fn first_search_initializes_root_and_records_every_simulation() {
        let mut tree = MonteCarloTree::new(TestGame(None), TestEvaluator, RootNoise::None);
        let mut rng = SmallRng::seed_from_u64(1);
        tree.do_simulations(1, 1.0, &mut rng).await.unwrap();

        assert_eq!(tree.get_total_descends(), Some(1));
        assert_eq!(tree.get_policy(), [1.0, 0.0]);
        assert_eq!(tree.get_move_stats(0).unwrap().1.get_avg_score(), 1.0);
    }

    #[tokio::test]
    async fn dirichlet_noise_is_reused_for_the_same_root() {
        let mut tree = MonteCarloTree::new(
            TestGame(None),
            TestEvaluator,
            RootNoise::Dirichlet {
                alpha: 0.1,
                epsilon: 0.25,
            },
        );
        let mut rng = SmallRng::seed_from_u64(1);
        tree.do_simulations(1, 1.0, &mut rng).await.unwrap();
        let first_noise = tree.root_noise_sample.clone();
        tree.do_simulations(1, 1.0, &mut rng).await.unwrap();
        assert_eq!(tree.root_noise_sample, first_noise);
        tree.advance(0, &TestMove::Win, TestGame(Some(-1.0)))
            .unwrap();
        assert!(tree.root_noise_sample.is_none());
    }

    #[tokio::test]
    async fn advance_reuses_subtree_without_evaluation() {
        let evaluations = Arc::new(AtomicUsize::new(0));
        let evaluator = CountingEvaluator(evaluations.clone());
        let mut tree = MonteCarloTree::new(DepthGame(2), evaluator, RootNoise::None);
        let mut rng = SmallRng::seed_from_u64(1);

        tree.do_simulations(2, 1.0, &mut rng).await.unwrap();
        assert_eq!(evaluations.load(Ordering::Relaxed), 2);
        tree.advance(0, &Step, DepthGame(1)).unwrap();

        assert_eq!(tree.get_total_descends(), Some(1));
        assert_eq!(evaluations.load(Ordering::Relaxed), 2);
        tree.do_simulations(1, 1.0, &mut rng).await.unwrap();
        assert_eq!(evaluations.load(Ordering::Relaxed), 2);
    }

    #[tokio::test]
    async fn position_and_successor_mismatches_are_rejected() {
        let mut tree = MonteCarloTree::new(TestGame(None), TestEvaluator, RootNoise::None);
        let mut rng = SmallRng::seed_from_u64(1);
        tree.do_simulations(1, 1.0, &mut rng).await.unwrap();

        assert!(!tree.matches_position(&TestGame(None), &[TestMove::Lose, TestMove::Win]));
        assert!(
            tree.advance(0, &TestMove::Win, TestGame(Some(1.0)))
                .is_err()
        );
    }
}
