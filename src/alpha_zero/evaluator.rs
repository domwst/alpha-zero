use std::{future::Future, marker::PhantomData};

use anyhow::{ensure, Context, Result};

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

pub struct NetworkPositionEvaluator<Net: AlphaZeroNet, Codec> {
    executor: NetworkBatchedExecutorHandle<Net>,
    _codec: PhantomData<fn() -> Codec>,
}

impl<Net: AlphaZeroNet, Codec> Clone for NetworkPositionEvaluator<Net, Codec> {
    fn clone(&self) -> Self {
        Self::new(self.executor.clone())
    }
}

impl<Net: AlphaZeroNet, Codec> NetworkPositionEvaluator<Net, Codec> {
    pub fn new(executor: NetworkBatchedExecutorHandle<Net>) -> Self {
        Self {
            executor,
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
        let (value, policy) = self.executor.execute(Codec::encode_position(state)).await?;
        let value = f32::try_from(value).context("converting network value to f32")?;
        let legal_policy = Codec::decode_policy(&policy, moves)?;

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
    use super::PositionEvaluation;

    #[test]
    fn evaluation_validation_rejects_malformed_policies() {
        assert!(PositionEvaluation {
            value: 0.0,
            legal_policy: vec![1.0],
        }
        .validate_for(2)
        .is_err());
        assert!(PositionEvaluation {
            value: 0.0,
            legal_policy: vec![f32::NAN, 0.0],
        }
        .validate_for(2)
        .is_err());
    }
}
