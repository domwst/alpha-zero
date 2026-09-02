use std::{
    future::Future,
    marker::PhantomData,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::Instant,
};

use anyhow::{Context, Result, ensure};

use super::{AlphaZeroNet, Game, NetworkBatchedExecutorHandle, PositionCodec};

#[derive(Clone, Debug, PartialEq)]
pub struct PositionEvaluation {
    pub value: f32,
    pub legal_policy: Vec<f32>,
}

impl PositionEvaluation {
    pub fn validate_for(&self, legal_moves: usize) -> Result<()> {
        ensure!(
            self.value.is_finite(),
            "evaluator returned a non-finite value"
        );
        ensure!(
            self.legal_policy.len() == legal_moves,
            "evaluator returned {} priors for {legal_moves} legal moves",
            self.legal_policy.len()
        );
        ensure!(
            self.legal_policy
                .iter()
                .all(|value| value.is_finite() && (0.0..=1.0).contains(value)),
            "evaluator returned an invalid legal policy"
        );
        ensure!(
            (self.legal_policy.iter().sum::<f32>() - 1.0).abs() < 1e-4,
            "evaluator legal policy is not normalized"
        );
        Ok(())
    }
}

/// Evaluates a position from its current canonical player's perspective.
/// The returned policy must follow the supplied legal-move order.
pub trait PositionEvaluator<TGame: Game> {
    fn evaluate<'a>(
        &'a self,
        state: &'a TGame,
        moves: &'a [TGame::Move],
    ) -> impl Future<Output = Result<PositionEvaluation>> + Send + 'a;
}

#[derive(Clone)]
enum InferenceSymmetrySelector {
    Identity,
    Random { seed: u64, sequence: Arc<AtomicU64> },
}

impl InferenceSymmetrySelector {
    fn random(seed: u64) -> Self {
        Self::Random {
            seed,
            sequence: Arc::new(AtomicU64::new(0)),
        }
    }

    fn select(&self, symmetry_count: usize) -> usize {
        assert!(symmetry_count > 0);
        match self {
            Self::Identity => 0,
            Self::Random { seed, sequence } => {
                let sequence = sequence.fetch_add(1, Ordering::Relaxed);
                ((random_sequence_value(*seed, sequence) as u32 as u64 * symmetry_count as u64)
                    >> 32) as usize
            }
        }
    }
}

fn random_sequence_value(seed: u64, sequence: u64) -> u64 {
    let mut value = seed.wrapping_add(sequence.wrapping_add(1).wrapping_mul(0x9e37_79b9_7f4a_7c15));
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

pub struct NetworkPositionEvaluator<Net: AlphaZeroNet, Codec> {
    executor: NetworkBatchedExecutorHandle<Net>,
    symmetry: InferenceSymmetrySelector,
    _codec: PhantomData<fn() -> Codec>,
}

impl<Net: AlphaZeroNet, Codec> Clone for NetworkPositionEvaluator<Net, Codec> {
    fn clone(&self) -> Self {
        Self {
            executor: self.executor.clone(),
            symmetry: self.symmetry.clone(),
            _codec: PhantomData,
        }
    }
}

impl<Net: AlphaZeroNet, Codec> NetworkPositionEvaluator<Net, Codec> {
    pub fn new(executor: NetworkBatchedExecutorHandle<Net>) -> Self {
        Self {
            executor,
            symmetry: InferenceSymmetrySelector::Identity,
            _codec: PhantomData,
        }
    }

    /// Creates an evaluator that independently samples one codec symmetry for
    /// every request. The sequence is deterministic for a given seed.
    pub fn with_random_symmetry(executor: NetworkBatchedExecutorHandle<Net>, seed: u64) -> Self {
        Self {
            executor,
            symmetry: InferenceSymmetrySelector::random(seed),
            _codec: PhantomData,
        }
    }
}

impl<TGame, Net, Codec> PositionEvaluator<TGame> for NetworkPositionEvaluator<Net, Codec>
where
    TGame: Game + Sync,
    TGame::Move: Sync,
    Net: AlphaZeroNet,
    Codec: PositionCodec<TGame>,
{
    async fn evaluate<'a>(
        &'a self,
        state: &'a TGame,
        moves: &'a [TGame::Move],
    ) -> Result<PositionEvaluation> {
        let phase_started = Instant::now();
        let symmetry_count = Codec::inference_symmetry_count();
        ensure!(
            symmetry_count > 0,
            "position codec must support at least one inference symmetry"
        );
        let symmetry = self.symmetry.select(symmetry_count);
        let input = Codec::encode_position_with_symmetry(state, symmetry);
        self.executor
            .record_position_encoding(phase_started.elapsed());

        let phase_started = Instant::now();
        let policy_mask = Codec::encode_policy_mask_with_symmetry(state, moves, symmetry);
        self.executor
            .record_policy_mask_construction(phase_started.elapsed());
        let policy_mask = policy_mask?;

        let (value, policy) = self.executor.execute(input, policy_mask).await?;
        let value = f32::try_from(value).context("converting network value to f32")?;
        let phase_started = Instant::now();
        let legal_policy = Codec::decode_policy_with_symmetry(&policy, moves, symmetry);
        self.executor.record_policy_decode(phase_started.elapsed());
        let legal_policy = legal_policy?;

        let evaluation = PositionEvaluation {
            value,
            legal_policy,
        };
        evaluation.validate_for(moves.len())?;
        Ok(evaluation)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::{InferenceSymmetrySelector, PositionEvaluation};

    #[test]
    fn random_symmetry_sequence_is_seeded_and_covers_all_d4_transforms() {
        let first = InferenceSymmetrySelector::random(17);
        let second = InferenceSymmetrySelector::random(17);
        let first = (0..256).map(|_| first.select(8)).collect::<Vec<_>>();
        let second = (0..256).map(|_| second.select(8)).collect::<Vec<_>>();

        assert_eq!(first, second);
        assert_eq!(first.iter().copied().collect::<BTreeSet<_>>().len(), 8);
        assert!(first.iter().all(|symmetry| *symmetry < 8));

        let selector = InferenceSymmetrySelector::random(17);
        let mut counts = [0usize; 8];
        for _ in 0..8_192 {
            counts[selector.select(8)] += 1;
        }
        assert!(
            counts
                .into_iter()
                .all(|count| (800..=1_250).contains(&count))
        );
    }

    #[test]
    fn evaluation_validation_rejects_malformed_policies() {
        assert!(
            PositionEvaluation {
                value: 0.0,
                legal_policy: vec![1.0],
            }
            .validate_for(2)
            .is_err()
        );
        assert!(
            PositionEvaluation {
                value: 0.0,
                legal_policy: vec![f32::NAN, 0.0],
            }
            .validate_for(2)
            .is_err()
        );
    }
}
