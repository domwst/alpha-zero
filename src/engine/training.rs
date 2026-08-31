use anyhow::Result;
use serde::{Deserialize, Serialize};

use super::{Game, MatchRecord, TrainingCodec};

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct TrainingSample<State, Policy> {
    pub state: State,
    pub policy: Policy,
    pub value: f32,
}

pub type TrainingGame<State, Policy> = Vec<TrainingSample<State, Policy>>;

pub fn extract_training_game<TGame, Codec>(
    record: MatchRecord<TGame>,
) -> Result<TrainingGame<TGame, Codec::Policy>>
where
    TGame: Game,
    Codec: TrainingCodec<TGame>,
{
    let terminal_actor = record.terminal_actor;
    let terminal_value = record.terminal_value;
    let mut samples = Vec::with_capacity(record.plies.len());

    for ply in record.plies {
        let Some(policy) = ply.decision.training_policy else {
            continue;
        };
        let policy = Codec::encode_policy_target(&ply.state, &policy)?;
        let value = if ply.actor == terminal_actor {
            terminal_value
        } else {
            -terminal_value
        };
        samples.push(TrainingSample {
            state: ply.state,
            policy,
            value,
        });
    }

    Ok(samples)
}

#[cfg(test)]
mod tests {
    use crate::{
        engine::{Game, MatchPly, MoveDecision, Seat, TrainingSample, TurnChange},
        gomoku::{BoardState, GomokuCodec, GomokuMove},
    };

    use super::{MatchRecord, extract_training_game};

    #[test]
    fn extraction_expands_policy_and_uses_actor_perspective() {
        let state = BoardState::new();
        let action = GomokuMove::from_xy(0, 0);
        let mut legal_policy = vec![0.0; BoardState::N * BoardState::N];
        legal_policy[0] = 1.0;
        let terminal_state = state.make_move(&action);
        let record = MatchRecord {
            plies: vec![MatchPly {
                state,
                action,
                actor: Seat::First,
                turn_change: TurnChange::SwitchPlayer,
                decision: MoveDecision {
                    move_index: 0,
                    training_policy: Some(legal_policy),
                    diagnostics: Default::default(),
                },
            }],
            terminal_state,
            terminal_actor: Seat::Second,
            terminal_value: -1.0,
        };

        let samples = extract_training_game::<_, GomokuCodec>(record).unwrap();
        let [TrainingSample { policy, value, .. }] = samples.as_slice() else {
            panic!("expected one training sample");
        };
        assert_eq!(*value, 1.0);
        assert_eq!(policy[(0, 0)], 1.0);
        assert_eq!(policy[(0, 1)], 0.0);
    }
}
