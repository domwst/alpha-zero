use std::time::{Duration, Instant};

use alz::{
    engine::{
        Game, MonteCarloTree, NetworkPositionEvaluator, PositionEvaluator, RootNoise,
        TerminationState, apply_temperature, sample_policy,
    },
    gomoku::{BoardState, CellState, GomokuCodec, GomokuModel, GomokuMove},
};
use anyhow::{Context, Result, ensure};
use axum::extract::ws::{Message, WebSocket};
use rand::{SeedableRng, rngs::SmallRng};
use tokio::sync::OwnedSemaphorePermit;

use crate::protocol::{
    Cell, CheckpointInfo, ClientMessage, GameOutcome, MoveStats, PROTOCOL_VERSION, ServerMessage,
    Stone, StoneColor,
};

pub type DemoEvaluator = NetworkPositionEvaluator<GomokuModel, GomokuCodec>;

enum SearchEvent {
    Incoming(Option<Result<Message, axum::Error>>),
    ChunkFinished(Result<()>),
}

#[derive(Clone, Debug)]
pub struct SessionConfig {
    pub checkpoint: CheckpointInfo,
    pub compute_device: String,
    pub max_search_simulations: u32,
    pub default_search_simulations: u32,
    pub c_puct: f32,
    pub search_chunk_size: u32,
    pub snapshot_interval: Duration,
    /// Public address of the research dashboard, linked from the header.
    pub dashboard_url: Option<String>,
}

impl SessionConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.max_search_simulations > 0,
            "maximum search simulations must be positive"
        );
        ensure!(
            (1..=self.max_search_simulations).contains(&self.default_search_simulations),
            "default search simulations must be between one and the configured maximum"
        );
        ensure!(self.search_chunk_size > 0, "search chunk must be positive");
        ensure!(
            self.c_puct.is_finite() && self.c_puct >= 0.0,
            "c-puct must be finite and non-negative"
        );
        ensure!(
            !self.snapshot_interval.is_zero(),
            "snapshot interval must be positive"
        );
        Ok(())
    }
}

pub async fn run_websocket(
    socket: WebSocket,
    evaluator: DemoEvaluator,
    config: SessionConfig,
    seed: u64,
    _session_permit: OwnedSemaphorePermit,
) {
    if let Err(error) = run_websocket_inner(socket, evaluator, config, seed).await {
        tracing::warn!(error = ?error, "interactive session ended with an error");
    }
}

async fn run_websocket_inner<Evaluator>(
    mut socket: WebSocket,
    evaluator: Evaluator,
    config: SessionConfig,
    seed: u64,
) -> Result<()>
where
    Evaluator: PositionEvaluator<BoardState> + Clone + Send + Sync,
{
    config.validate()?;
    let mut session = GameSession::new(evaluator, config.clone(), seed);
    send(
        &mut socket,
        &ServerMessage::Hello {
            protocol_version: PROTOCOL_VERSION,
            board_size: BoardState::N,
            compute_device: config.compute_device,
            checkpoint: config.checkpoint,
            max_search_simulations: config.max_search_simulations,
            default_search_simulations: config.default_search_simulations,
            c_puct: config.c_puct,
            snapshot_interval_ms: config.snapshot_interval.as_millis() as u64,
            dashboard_url: config.dashboard_url,
        },
    )
    .await?;

    let mut last_snapshot_at: Option<Instant> = None;
    loop {
        if session.should_search() {
            let chunk = session.next_chunk_size();
            let event = {
                let search = session.run_search_chunk(chunk);
                tokio::pin!(search);
                tokio::select! {
                    incoming = socket.recv() => SearchEvent::Incoming(incoming),
                    result = &mut search => SearchEvent::ChunkFinished(result),
                }
            };

            match event {
                SearchEvent::Incoming(incoming) => {
                    let Some(incoming) = incoming else {
                        return Ok(());
                    };
                    match incoming.context("receiving websocket message")? {
                        Message::Text(text) => {
                            let messages =
                                match serde_json::from_str::<ClientMessage>(text.as_str()) {
                                    Ok(command) => session.handle_command(command),
                                    Err(error) => vec![ServerMessage::Error {
                                        code: "invalid_message",
                                        message: error.to_string(),
                                        recoverable: true,
                                    }],
                                };
                            send_many(&mut socket, messages).await?;
                        }
                        Message::Ping(payload) => socket.send(Message::Pong(payload)).await?,
                        Message::Close(_) => return Ok(()),
                        Message::Binary(_) | Message::Pong(_) => {}
                    }
                }
                SearchEvent::ChunkFinished(result) => {
                    if let Err(error) = result {
                        session.stop_after_error();
                        send(
                            &mut socket,
                            &ServerMessage::Error {
                                code: "search_failed",
                                message: format!("network search failed: {error:#}"),
                                recoverable: false,
                            },
                        )
                        .await?;
                        continue;
                    }

                    let complete = !session.should_search();
                    let due = last_snapshot_at
                        .is_none_or(|last| last.elapsed() >= config.snapshot_interval);
                    if (complete || due)
                        && let Some(snapshot) = session.search_snapshot()
                    {
                        send(&mut socket, &snapshot).await?;
                        last_snapshot_at = Some(Instant::now());
                    }
                    if complete {
                        send(&mut socket, &session.search_status()).await?;
                    }
                }
            }
        } else {
            let Some(incoming) = socket.recv().await else {
                return Ok(());
            };
            match incoming.context("receiving websocket message")? {
                Message::Text(text) => {
                    let messages = match serde_json::from_str::<ClientMessage>(text.as_str()) {
                        Ok(command) => session.handle_command(command),
                        Err(error) => vec![ServerMessage::Error {
                            code: "invalid_message",
                            message: error.to_string(),
                            recoverable: true,
                        }],
                    };
                    send_many(&mut socket, messages).await?;
                }
                Message::Ping(payload) => socket.send(Message::Pong(payload)).await?,
                Message::Close(_) => return Ok(()),
                Message::Binary(_) | Message::Pong(_) => {}
            }
        }
    }
}

async fn send(socket: &mut WebSocket, message: &ServerMessage) -> Result<()> {
    let json = serde_json::to_string(message).context("serializing websocket message")?;
    socket
        .send(Message::Text(json.into()))
        .await
        .context("sending websocket message")
}

async fn send_many(socket: &mut WebSocket, messages: Vec<ServerMessage>) -> Result<()> {
    for message in messages {
        send(socket, &message).await?;
    }
    Ok(())
}

struct GameSession<Evaluator: PositionEvaluator<BoardState>> {
    evaluator: Evaluator,
    config: SessionConfig,
    state: BoardState,
    tree: MonteCarloTree<BoardState, Evaluator>,
    rng: SmallRng,
    stones: Vec<Stone>,
    human_color: StoneColor,
    position_id: u64,
    analysis_id: u64,
    carried_visits: u32,
    target_simulations: u32,
    /// The client asked for this position's evaluation; with a target of zero
    /// that is all it gets: the network's priors and value, without search.
    evaluate_root: bool,
    search_elapsed: Duration,
}

impl<Evaluator> GameSession<Evaluator>
where
    Evaluator: PositionEvaluator<BoardState> + Clone + Send + Sync,
{
    fn new(evaluator: Evaluator, config: SessionConfig, seed: u64) -> Self {
        let state = BoardState::new();
        Self {
            tree: MonteCarloTree::new(state.clone(), evaluator.clone(), RootNoise::None),
            evaluator,
            config,
            state,
            rng: SmallRng::seed_from_u64(seed),
            stones: Vec::new(),
            human_color: StoneColor::Black,
            position_id: 1,
            analysis_id: 1,
            carried_visits: 0,
            target_simulations: 0,
            evaluate_root: false,
            search_elapsed: Duration::ZERO,
        }
    }

    fn to_move(&self) -> StoneColor {
        if self.stones.len().is_multiple_of(2) {
            StoneColor::Black
        } else {
            StoneColor::White
        }
    }

    fn outcome(&self) -> Option<GameOutcome> {
        if self.state.is_win() != CellState::Empty {
            Some(match self.stones.last()?.color {
                StoneColor::Black => GameOutcome::BlackWon,
                StoneColor::White => GameOutcome::WhiteWon,
            })
        } else if self.stones.len() == BoardState::N * BoardState::N {
            Some(GameOutcome::Draw)
        } else {
            None
        }
    }

    fn current_total_visits(&self) -> u32 {
        self.tree
            .get_total_descends()
            .unwrap_or(0)
            .min(u32::MAX as usize) as u32
    }

    fn searched_simulations(&self) -> u32 {
        self.current_total_visits()
            .saturating_sub(self.carried_visits)
    }

    fn root_evaluated(&self) -> bool {
        self.tree.get_network_state_estimation().is_some()
    }

    fn should_search(&self) -> bool {
        self.outcome().is_none()
            && (self.searched_simulations() < self.target_simulations
                || (self.evaluate_root && !self.root_evaluated()))
    }

    /// Zero when only the root's evaluation is outstanding.
    fn next_chunk_size(&self) -> u32 {
        self.target_simulations
            .saturating_sub(self.searched_simulations())
            .min(self.config.search_chunk_size)
    }

    async fn run_search_chunk(&mut self, chunk: u32) -> Result<()> {
        let started = Instant::now();
        if chunk == 0 {
            self.tree.expand_root().await?;
        } else {
            self.tree
                .do_simulations(chunk as usize, self.config.c_puct, &mut self.rng)
                .await?;
        }
        self.search_elapsed += started.elapsed();
        Ok(())
    }

    fn handle_command(&mut self, command: ClientMessage) -> Vec<ServerMessage> {
        match self.try_handle_command(command) {
            Ok(messages) => messages,
            Err(error) => vec![ServerMessage::Error {
                code: "invalid_command",
                message: format!("{error:#}"),
                recoverable: true,
            }],
        }
    }

    fn try_handle_command(&mut self, command: ClientMessage) -> Result<Vec<ServerMessage>> {
        match command {
            ClientMessage::NewGame { human_color } => {
                self.start_new_game(human_color);
                Ok(self.position_update_messages())
            }
            ClientMessage::RestoreGame { human_color, moves } => {
                self.restore_game(human_color, &moves)?;
                Ok(self.position_update_messages())
            }
            ClientMessage::StartSearch {
                position_id,
                simulations,
            } => {
                self.ensure_current_position(position_id)?;
                ensure!(
                    simulations <= self.config.max_search_simulations,
                    "requested {simulations} simulations; allowed range is 0..={}",
                    self.config.max_search_simulations
                );
                ensure!(self.outcome().is_none(), "the game is already over");
                self.analysis_id += 1;
                self.target_simulations = simulations;
                self.evaluate_root = true;
                let mut messages = vec![self.search_status()];
                if let Some(snapshot) = self.search_snapshot() {
                    messages.push(snapshot);
                }
                Ok(messages)
            }
            ClientMessage::StopSearch { position_id } => {
                self.ensure_current_position(position_id)?;
                self.analysis_id += 1;
                self.target_simulations = self.searched_simulations();
                let mut messages = vec![self.search_status()];
                if let Some(snapshot) = self.search_snapshot() {
                    messages.push(snapshot);
                }
                Ok(messages)
            }
            ClientMessage::Play {
                position_id,
                row,
                column,
            } => {
                self.ensure_current_position(position_id)?;
                self.apply_move(row, column)?;
                Ok(self.position_update_messages())
            }
            ClientMessage::ChooseNetworkMove {
                position_id,
                temperature,
                instant,
            } => {
                self.ensure_current_position(position_id)?;
                ensure!(
                    self.to_move() != self.human_color,
                    "it is the human player's turn"
                );
                ensure!(
                    temperature.is_finite() && (0.0..=5.0).contains(&temperature),
                    "temperature must be finite and between 0 and 5"
                );
                let policy = if instant {
                    self.tree
                        .root_snapshot()
                        .context("the position has not been evaluated yet")?
                        .moves
                        .iter()
                        .map(|r#move| r#move.prior)
                        .collect()
                } else {
                    ensure!(
                        self.searched_simulations() >= self.target_simulations
                            && self.target_simulations > 0,
                        "search has not reached its target"
                    );
                    self.tree.get_policy()
                };
                let sampling_policy = apply_temperature(&policy, temperature);
                let move_index = sample_policy(&sampling_policy, &mut self.rng);
                let moves = self.legal_moves()?;
                let action = moves
                    .get(move_index)
                    .context("sampled move is outside the legal move list")?;
                let (row, column) = action.to_xy();
                self.apply_move(row as u8, column as u8)?;
                Ok(self.position_update_messages())
            }
        }
    }

    fn ensure_current_position(&self, position_id: u64) -> Result<()> {
        ensure!(
            position_id == self.position_id,
            "stale position {position_id}; current position is {}",
            self.position_id
        );
        Ok(())
    }

    fn start_new_game(&mut self, human_color: StoneColor) {
        self.state = BoardState::new();
        self.tree =
            MonteCarloTree::new(self.state.clone(), self.evaluator.clone(), RootNoise::None);
        self.stones.clear();
        self.human_color = human_color;
        self.position_id += 1;
        self.analysis_id += 1;
        self.carried_visits = 0;
        self.target_simulations = 0;
        self.evaluate_root = false;
        self.search_elapsed = Duration::ZERO;
    }

    fn restore_game(&mut self, human_color: StoneColor, moves: &[Cell]) -> Result<()> {
        ensure!(
            moves.len() <= BoardState::N * BoardState::N,
            "restore contains too many moves"
        );

        let mut state = BoardState::new();
        let mut stones = Vec::with_capacity(moves.len());
        for (ply, cell) in moves.iter().enumerate() {
            ensure!(
                (cell.row as usize) < BoardState::N && (cell.column as usize) < BoardState::N,
                "restore move {} is outside the board",
                ply + 1
            );
            let legal_moves = match state.get_state() {
                TerminationState::Moves(legal_moves) => legal_moves,
                TerminationState::Terminal(_) => {
                    anyhow::bail!("restore contains moves after the game ended")
                }
            };
            let action = GomokuMove::from_xy(cell.row as usize, cell.column as usize);
            ensure!(
                legal_moves.contains(&action),
                "restore move {} targets an occupied cell",
                ply + 1
            );
            let color = if ply.is_multiple_of(2) {
                StoneColor::Black
            } else {
                StoneColor::White
            };
            state = state.make_move(&action);
            stones.push(Stone {
                row: cell.row,
                column: cell.column,
                color,
            });
        }

        self.tree = MonteCarloTree::new(state.clone(), self.evaluator.clone(), RootNoise::None);
        self.state = state;
        self.stones = stones;
        self.human_color = human_color;
        self.position_id += 1;
        self.analysis_id += 1;
        self.carried_visits = 0;
        self.target_simulations = 0;
        self.evaluate_root = false;
        self.search_elapsed = Duration::ZERO;
        Ok(())
    }

    fn legal_moves(&self) -> Result<Box<[GomokuMove]>> {
        match self.state.get_state() {
            TerminationState::Moves(moves) => Ok(moves),
            TerminationState::Terminal(_) => anyhow::bail!("the game is already over"),
        }
    }

    fn apply_move(&mut self, row: u8, column: u8) -> Result<()> {
        ensure!(
            (row as usize) < BoardState::N && (column as usize) < BoardState::N,
            "cell ({row}, {column}) is outside the board"
        );
        let legal_moves = self.legal_moves()?;
        let action = GomokuMove::from_xy(row as usize, column as usize);
        let move_index = legal_moves
            .iter()
            .position(|candidate| *candidate == action)
            .context("selected cell is occupied")?;
        let next_state = self.state.make_move(&action);
        self.tree.advance(move_index, &action, next_state.clone())?;
        self.stones.push(Stone {
            row,
            column,
            color: self.to_move(),
        });
        self.state = next_state;
        self.position_id += 1;
        self.analysis_id += 1;
        self.carried_visits = self.current_total_visits();
        self.target_simulations = 0;
        self.evaluate_root = false;
        self.search_elapsed = Duration::ZERO;
        Ok(())
    }

    fn position_update_messages(&self) -> Vec<ServerMessage> {
        let mut messages = vec![self.position_message(), self.search_status()];
        if let Some(snapshot) = self.search_snapshot() {
            messages.push(snapshot);
        }
        messages
    }

    fn position_message(&self) -> ServerMessage {
        ServerMessage::Position {
            position_id: self.position_id,
            ply: self.stones.len(),
            human_color: self.human_color,
            to_move: self.to_move(),
            stones: self.stones.clone(),
            last_move: self.stones.last().map(|stone| Cell {
                row: stone.row,
                column: stone.column,
            }),
            outcome: self.outcome(),
            carried_visits: self.carried_visits,
        }
    }

    fn search_status(&self) -> ServerMessage {
        ServerMessage::SearchStatus {
            position_id: self.position_id,
            analysis_id: self.analysis_id,
            searched_simulations: self.current_total_visits(),
            target_simulations: self.carried_visits + self.target_simulations,
            carried_visits: self.carried_visits,
            running: self.should_search(),
        }
    }

    fn search_snapshot(&self) -> Option<ServerMessage> {
        let root = self.tree.root_snapshot()?;
        let elapsed_seconds = self.search_elapsed.as_secs_f64();
        let searched = self.searched_simulations();
        let network_value = root.network_value;
        let search_value = root.search_value();
        let total_visits = root.total_visits;
        let moves = root
            .moves
            .into_iter()
            .map(|r#move| {
                let (row, column) = r#move.action.to_xy();
                MoveStats {
                    row: row as u8,
                    column: column as u8,
                    prior: r#move.prior,
                    visits: r#move.visits,
                    mean_value: r#move.mean_value(),
                }
            })
            .collect();
        Some(ServerMessage::SearchSnapshot {
            position_id: self.position_id,
            analysis_id: self.analysis_id,
            searched_simulations: searched + self.carried_visits,
            carried_visits: self.carried_visits,
            total_visits,
            target_simulations: self.carried_visits + self.target_simulations,
            elapsed_ms: self.search_elapsed.as_millis().min(u64::MAX as u128) as u64,
            simulations_per_second: if elapsed_seconds > 0.0 {
                searched as f64 / elapsed_seconds
            } else {
                0.0
            },
            network_value,
            search_value,
            moves,
            complete: !self.should_search(),
        })
    }

    fn stop_after_error(&mut self) {
        self.analysis_id += 1;
        self.target_simulations = self.searched_simulations();
    }
}

#[cfg(test)]
mod tests {
    use alz::engine::{PositionEvaluation, PositionEvaluator};

    use super::*;

    #[derive(Clone)]
    struct UniformEvaluator;

    impl PositionEvaluator<BoardState> for UniformEvaluator {
        async fn evaluate<'a>(
            &'a mut self,
            _state: &'a BoardState,
            moves: &'a [GomokuMove],
        ) -> anyhow::Result<PositionEvaluation> {
            Ok(PositionEvaluation {
                value: 0.0,
                legal_policy: vec![1.0 / moves.len() as f32; moves.len()],
            })
        }
    }

    /// Prefers one legal move by a wide margin, so a temperature of zero must pick it.
    #[derive(Clone)]
    struct FavoriteMoveEvaluator(usize);

    impl PositionEvaluator<BoardState> for FavoriteMoveEvaluator {
        async fn evaluate<'a>(
            &'a mut self,
            _state: &'a BoardState,
            moves: &'a [GomokuMove],
        ) -> anyhow::Result<PositionEvaluation> {
            let mut policy = vec![0.1 / moves.len() as f32; moves.len()];
            policy[self.0] += 0.9;
            Ok(PositionEvaluation {
                value: 0.25,
                legal_policy: policy,
            })
        }
    }

    fn config() -> SessionConfig {
        SessionConfig {
            checkpoint: CheckpointInfo {
                architecture: "test".to_owned(),
                epoch: 0,
                model_digest: "test".to_owned(),
            },
            compute_device: "CPU".to_owned(),
            max_search_simulations: 64,
            default_search_simulations: 32,
            c_puct: 1.0,
            search_chunk_size: 8,
            snapshot_interval: Duration::from_millis(100),
            dashboard_url: None,
        }
    }

    #[test]
    fn configured_search_limit_is_enforced() {
        let mut session = GameSession::new(UniformEvaluator, config(), 1);
        let response = session.handle_command(ClientMessage::StartSearch {
            position_id: 1,
            simulations: 65,
        });
        assert!(matches!(
            response.as_slice(),
            [ServerMessage::Error {
                code: "invalid_command",
                recoverable: true,
                ..
            }]
        ));
        assert!(!session.should_search());
    }

    #[tokio::test]
    async fn search_is_chunked_and_reports_complete_snapshots() {
        let mut session = GameSession::new(UniformEvaluator, config(), 1);
        session
            .try_handle_command(ClientMessage::StartSearch {
                position_id: 1,
                simulations: 12,
            })
            .unwrap();
        assert_eq!(session.next_chunk_size(), 8);
        session.run_search_chunk(8).await.unwrap();
        assert_eq!(session.searched_simulations(), 8);
        assert_eq!(session.next_chunk_size(), 4);
        session.run_search_chunk(4).await.unwrap();
        assert!(!session.should_search());

        let ServerMessage::SearchSnapshot {
            searched_simulations,
            total_visits,
            complete,
            moves,
            ..
        } = session.search_snapshot().unwrap()
        else {
            panic!("expected a search snapshot");
        };
        assert_eq!(searched_simulations, 12);
        assert_eq!(total_visits, 12);
        assert!(complete);
        assert_eq!(moves.len(), BoardState::N * BoardState::N);
    }

    #[tokio::test]
    async fn advancing_preserves_subtree_visits_as_carried_work() {
        let mut session = GameSession::new(UniformEvaluator, config(), 1);
        session
            .try_handle_command(ClientMessage::StartSearch {
                position_id: 1,
                simulations: 16,
            })
            .unwrap();
        session.run_search_chunk(16).await.unwrap();
        session
            .try_handle_command(ClientMessage::Play {
                position_id: 1,
                row: 0,
                column: 0,
            })
            .unwrap();

        assert_eq!(session.position_id, 2);
        assert_eq!(session.searched_simulations(), 0);
        assert_eq!(session.current_total_visits(), session.carried_visits);
    }

    #[tokio::test]
    async fn restore_replays_moves_and_discards_previous_analysis() {
        let mut session = GameSession::new(UniformEvaluator, config(), 1);
        session
            .try_handle_command(ClientMessage::StartSearch {
                position_id: 1,
                simulations: 8,
            })
            .unwrap();
        session.run_search_chunk(8).await.unwrap();
        assert_eq!(session.current_total_visits(), 8);

        session
            .try_handle_command(ClientMessage::RestoreGame {
                human_color: StoneColor::White,
                moves: vec![Cell { row: 9, column: 9 }, Cell { row: 9, column: 10 }],
            })
            .unwrap();

        assert_eq!(session.position_id, 2);
        assert_eq!(session.human_color, StoneColor::White);
        assert_eq!(session.stones.len(), 2);
        assert_eq!(session.stones[0].color, StoneColor::Black);
        assert_eq!(session.stones[1].color, StoneColor::White);
        assert_eq!(session.to_move(), StoneColor::Black);
        assert_eq!(session.current_total_visits(), 0);
        assert_eq!(session.carried_visits, 0);
        assert_eq!(session.target_simulations, 0);
    }

    #[test]
    fn invalid_restore_does_not_partially_replace_the_session() {
        let mut session = GameSession::new(UniformEvaluator, config(), 1);
        let result = session.try_handle_command(ClientMessage::RestoreGame {
            human_color: StoneColor::White,
            moves: vec![Cell { row: 9, column: 9 }, Cell { row: 9, column: 9 }],
        });

        assert!(result.is_err());
        assert_eq!(session.position_id, 1);
        assert_eq!(session.human_color, StoneColor::Black);
        assert!(session.stones.is_empty());
        assert_eq!(session.current_total_visits(), 0);
    }

    #[tokio::test]
    async fn evaluation_only_search_reports_priors_without_visits() {
        let mut session = GameSession::new(UniformEvaluator, config(), 1);
        let response = session
            .try_handle_command(ClientMessage::StartSearch {
                position_id: 1,
                simulations: 0,
            })
            .unwrap();
        // Nothing is evaluated yet, so only the status goes out at once.
        assert!(matches!(
            response.as_slice(),
            [ServerMessage::SearchStatus { running: true, .. }]
        ));
        assert!(session.should_search());
        assert_eq!(session.next_chunk_size(), 0);

        session.run_search_chunk(0).await.unwrap();
        assert!(!session.should_search());
        let ServerMessage::SearchSnapshot {
            total_visits,
            target_simulations,
            complete,
            moves,
            ..
        } = session.search_snapshot().unwrap()
        else {
            panic!("expected a search snapshot");
        };
        assert_eq!(total_visits, 0);
        assert_eq!(target_simulations, 0);
        assert!(complete);
        assert!(
            moves
                .iter()
                .all(|r#move| r#move.prior > 0.0 && r#move.visits == 0)
        );
    }

    #[tokio::test]
    async fn instant_reply_follows_the_network_policy() {
        let favorite = 5;
        let mut session = GameSession::new(FavoriteMoveEvaluator(favorite), config(), 1);
        session
            .try_handle_command(ClientMessage::Play {
                position_id: 1,
                row: 9,
                column: 9,
            })
            .unwrap();
        assert_ne!(session.to_move(), session.human_color);

        let early = session.try_handle_command(ClientMessage::ChooseNetworkMove {
            position_id: 2,
            temperature: 0.0,
            instant: true,
        });
        assert!(
            early.is_err(),
            "an instant reply needs the position's evaluation"
        );

        session
            .try_handle_command(ClientMessage::StartSearch {
                position_id: 2,
                simulations: 0,
            })
            .unwrap();
        session.run_search_chunk(0).await.unwrap();
        let searched = session.try_handle_command(ClientMessage::ChooseNetworkMove {
            position_id: 2,
            temperature: 0.0,
            instant: false,
        });
        assert!(
            searched.is_err(),
            "a search reply still needs a reached search target"
        );

        let (row, column) = session.legal_moves().unwrap()[favorite].to_xy();
        session
            .try_handle_command(ClientMessage::ChooseNetworkMove {
                position_id: 2,
                temperature: 0.0,
                instant: true,
            })
            .unwrap();
        let reply = session.stones.last().unwrap();
        assert_eq!((reply.row as usize, reply.column as usize), (row, column));
        assert_eq!(reply.color, StoneColor::White);
        assert!(
            !session.evaluate_root,
            "the next position waits for its own request"
        );
    }

    #[test]
    fn manual_play_can_override_the_network_turn() {
        let mut session = GameSession::new(UniformEvaluator, config(), 1);
        session
            .try_handle_command(ClientMessage::Play {
                position_id: 1,
                row: 0,
                column: 0,
            })
            .unwrap();
        assert_eq!(session.to_move(), StoneColor::White);
        assert_ne!(session.to_move(), session.human_color);

        session
            .try_handle_command(ClientMessage::Play {
                position_id: 2,
                row: 0,
                column: 1,
            })
            .unwrap();

        assert_eq!(session.position_id, 3);
        assert_eq!(session.stones.len(), 2);
        assert_eq!(session.stones[1].color, StoneColor::White);
        assert_eq!(session.to_move(), StoneColor::Black);
    }
}
