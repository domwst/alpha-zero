use serde::{Deserialize, Serialize};

pub const PROTOCOL_VERSION: u32 = 2;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StoneColor {
    Black,
    White,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GameOutcome {
    BlackWon,
    WhiteWon,
    Draw,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum ClientMessage {
    NewGame {
        human_color: StoneColor,
    },
    RestoreGame {
        human_color: StoneColor,
        moves: Vec<Cell>,
    },
    StartSearch {
        position_id: u64,
        simulations: u32,
    },
    StopSearch {
        position_id: u64,
    },
    Play {
        position_id: u64,
        row: u8,
        column: u8,
    },
    ChooseNetworkMove {
        position_id: u64,
        temperature: f32,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CheckpointInfo {
    pub architecture: String,
    pub epoch: usize,
    pub model_digest: String,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct Cell {
    pub row: u8,
    pub column: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct Stone {
    pub row: u8,
    pub column: u8,
    pub color: StoneColor,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct MoveStats {
    pub row: u8,
    pub column: u8,
    pub prior: f32,
    pub visits: u32,
    pub mean_value: Option<f32>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ServerMessage {
    Hello {
        protocol_version: u32,
        board_size: usize,
        compute_device: String,
        checkpoint: CheckpointInfo,
        max_search_simulations: u32,
        default_search_simulations: u32,
        c_puct: f32,
        snapshot_interval_ms: u64,
    },
    Position {
        position_id: u64,
        ply: usize,
        human_color: StoneColor,
        to_move: StoneColor,
        stones: Vec<Stone>,
        last_move: Option<Cell>,
        outcome: Option<GameOutcome>,
        carried_visits: u32,
    },
    SearchStatus {
        position_id: u64,
        analysis_id: u64,
        searched_simulations: u32,
        target_simulations: u32,
        running: bool,
    },
    SearchSnapshot {
        position_id: u64,
        analysis_id: u64,
        searched_simulations: u32,
        carried_visits: u32,
        total_visits: u32,
        target_simulations: u32,
        elapsed_ms: u64,
        simulations_per_second: f64,
        network_value: f32,
        search_value: Option<f32>,
        moves: Vec<MoveStats>,
        complete: bool,
    },
    Error {
        code: &'static str,
        message: String,
        recoverable: bool,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn protocol_uses_stable_discriminators_and_field_names() {
        let command = serde_json::from_str::<ClientMessage>(
            r#"{"type":"start_search","position_id":7,"simulations":2000}"#,
        )
        .unwrap();
        assert_eq!(
            command,
            ClientMessage::StartSearch {
                position_id: 7,
                simulations: 2_000,
            }
        );

        let restore = serde_json::from_str::<ClientMessage>(
            r#"{"type":"restore_game","human_color":"white","moves":[{"row":9,"column":9},{"row":9,"column":10}]}"#,
        )
        .unwrap();
        assert_eq!(
            restore,
            ClientMessage::RestoreGame {
                human_color: StoneColor::White,
                moves: vec![Cell { row: 9, column: 9 }, Cell { row: 9, column: 10 },],
            }
        );

        let message = ServerMessage::SearchStatus {
            position_id: 7,
            analysis_id: 3,
            searched_simulations: 512,
            target_simulations: 2_000,
            running: true,
        };
        assert_eq!(
            serde_json::to_value(message).unwrap(),
            serde_json::json!({
                "type": "search_status",
                "position_id": 7,
                "analysis_id": 3,
                "searched_simulations": 512,
                "target_simulations": 2_000,
                "running": true,
            })
        );
    }

    #[test]
    fn hello_reports_the_actual_compute_device() {
        let message = ServerMessage::Hello {
            protocol_version: PROTOCOL_VERSION,
            board_size: 19,
            compute_device: "CPU".to_owned(),
            checkpoint: CheckpointInfo {
                architecture: "kata_v1".to_owned(),
                epoch: 38,
                model_digest: "abcdef012345".to_owned(),
            },
            max_search_simulations: 10_000,
            default_search_simulations: 2_000,
            c_puct: 1.0,
            snapshot_interval_ms: 100,
        };
        let json = serde_json::to_value(message).unwrap();

        assert_eq!(json["compute_device"], "CPU");
    }
}
