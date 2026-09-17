//! Versioned analysis envelopes leave state and action semantics to the game.
use super::Game;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PositionEnvelope {
    pub game_type: String,
    pub state: Value,
}

pub trait GameAdapter {
    type Position: Game;
    const ID: &'static str;
    fn decode(state: Value) -> Result<Self::Position>;
    fn describe() -> Value;
}
