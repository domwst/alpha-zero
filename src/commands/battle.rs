use std::{
    fs,
    path::Path,
    time::{Duration, Instant},
};

use alz::{
    engine::{
        BattleSettings, ExecutorScope, MatchRecord, NetworkBatchStats, NetworkPositionEvaluator,
        Seat, do_battle,
    },
    gomoku::{BoardState, GomokuCodec, GomokuModel},
};
use anyhow::{Context, Result, ensure};
use futures::{StreamExt, stream};
use serde::Serialize;
use tch::Kind;

use crate::{cli::BattleArgs, training_snapshot::SnapshotDescriptor};

use super::common::{load_network, resolve_device};

const BATTLE_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum Competitor {
    FirstCheckpoint,
    SecondCheckpoint,
}

impl Competitor {
    fn other(self) -> Self {
        match self {
            Self::FirstCheckpoint => Self::SecondCheckpoint,
            Self::SecondCheckpoint => Self::FirstCheckpoint,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::FirstCheckpoint => "first_checkpoint",
            Self::SecondCheckpoint => "second_checkpoint",
        }
    }
}

#[derive(Debug, Serialize)]
struct MoveLog {
    ply: usize,
    checkpoint: Competitor,
    seat: &'static str,
    row: usize,
    column: usize,
    value_estimate: Option<f32>,
}

#[derive(Debug, Serialize)]
struct GameLog {
    game: usize,
    first_seat: Competitor,
    second_seat: Competitor,
    winner: Option<Competitor>,
    value_for_first_checkpoint: f32,
    plies: usize,
    moves: Vec<MoveLog>,
}

#[derive(Debug, Default, Serialize)]
struct SeatAggregate {
    games: usize,
    wins: usize,
    losses: usize,
    draws: usize,
}

#[derive(Debug, Serialize)]
struct CompetitorAggregate {
    wins: usize,
    losses: usize,
    draws: usize,
    score: f64,
    score_rate: f64,
    score_rate_95_percent_low: f64,
    score_rate_95_percent_high: f64,
    elo_difference: Option<f64>,
    as_first_seat: SeatAggregate,
    as_second_seat: SeatAggregate,
}

#[derive(Debug, Serialize)]
struct InferenceAggregate {
    evaluations_per_second: f64,
    batch: NetworkBatchStats,
}

#[derive(Debug, Serialize)]
struct BattleReport<'a> {
    schema_version: u32,
    config: &'a BattleArgs,
    first_checkpoint: SnapshotDescriptor,
    second_checkpoint: SnapshotDescriptor,
    first_temperature: f32,
    second_temperature: f32,
    duration_seconds: f64,
    games_per_second: f64,
    total_plies: usize,
    average_plies: f64,
    first_checkpoint_result: CompetitorAggregate,
    second_checkpoint_result: CompetitorAggregate,
    first_checkpoint_inference: InferenceAggregate,
    second_checkpoint_inference: InferenceAggregate,
    games: Vec<GameLog>,
}

pub async fn run(args: BattleArgs) -> Result<()> {
    validate_args(&args)?;
    let first_temperature = args.first_temperature.unwrap_or(args.temperature);
    let second_temperature = args.second_temperature.unwrap_or(args.temperature);

    let device = resolve_device(&args.device)?;
    let (first_var_store, first_network, first_snapshot) =
        load_network(&args.first_checkpoint_dir, None, device)?;
    let (second_var_store, second_network, second_snapshot) =
        load_network(&args.second_checkpoint_dir, None, device)?;

    let first_executor = ExecutorScope::<(), _>::new(
        first_network,
        args.games_parallelism,
        args.inference_batch_size,
        Duration::from_micros(args.batch_timeout_us),
        (Kind::Float, first_var_store.device()),
    );
    let second_executor = ExecutorScope::<(), _>::new(
        second_network,
        args.games_parallelism,
        args.inference_batch_size,
        Duration::from_micros(args.batch_timeout_us),
        (Kind::Float, second_var_store.device()),
    );
    let first_handle = first_executor.evaluator_handle();
    let second_handle = second_executor.evaluator_handle();

    let started = Instant::now();
    let mut matches = stream::iter(0..args.games)
        .map(|game| {
            let first_handle = first_handle.clone();
            let second_handle = second_handle.clone();
            async move {
                let first_evaluator =
                    NetworkPositionEvaluator::<GomokuModel, GomokuCodec>::new(first_handle);
                let second_evaluator =
                    NetworkPositionEvaluator::<GomokuModel, GomokuCodec>::new(second_handle);
                let first_seed = derive_seed(args.seed, game as u64, 0);
                let second_seed = derive_seed(args.seed, game as u64, 1);

                if game % 2 == 0 {
                    let record = do_battle(
                        BoardState::new(),
                        BattleSettings {
                            simulations: args.simulations,
                            c_puct: args.c_puct,
                            first_temperature,
                            second_temperature,
                            first_seed,
                            second_seed,
                        },
                        first_evaluator,
                        second_evaluator,
                    )
                    .await?;
                    Ok(game_log(game, Competitor::FirstCheckpoint, record))
                } else {
                    let record = do_battle(
                        BoardState::new(),
                        BattleSettings {
                            simulations: args.simulations,
                            c_puct: args.c_puct,
                            first_temperature: second_temperature,
                            second_temperature: first_temperature,
                            first_seed: second_seed,
                            second_seed: first_seed,
                        },
                        second_evaluator,
                        first_evaluator,
                    )
                    .await?;
                    Ok(game_log(game, Competitor::SecondCheckpoint, record))
                }
            }
        })
        .buffer_unordered(args.games_parallelism);

    let mut results = Vec::with_capacity(args.games);
    let mut heartbeat = (args.heartbeat_seconds > 0)
        .then(|| tokio::time::interval(Duration::from_secs(args.heartbeat_seconds)));
    if let Some(timer) = heartbeat.as_mut() {
        timer.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        timer.tick().await;
    }
    loop {
        let next = if let Some(timer) = heartbeat.as_mut() {
            tokio::select! {
                result = matches.next() => result,
                _ = timer.tick() => {
                    log_progress(
                        "heartbeat",
                        results.len(),
                        args.games,
                        started.elapsed(),
                        first_executor.completed_evaluations(),
                        second_executor.completed_evaluations(),
                    );
                    continue;
                }
            }
        } else {
            matches.next().await
        };
        let Some(result) = next else {
            break;
        };
        if let Ok(game) = &result {
            log_game_summary(game);
        }
        results.push(result);
        log_progress(
            "game_complete",
            results.len(),
            args.games,
            started.elapsed(),
            first_executor.completed_evaluations(),
            second_executor.completed_evaluations(),
        );
    }

    drop(matches);
    drop((first_handle, second_handle));
    let ((_, first_batch_stats), (_, second_batch_stats)) = tokio::join!(
        first_executor.join_with_stats(),
        second_executor.join_with_stats()
    );
    let duration = started.elapsed();
    let mut games = results.into_iter().collect::<Result<Vec<_>>>()?;
    games.sort_unstable_by_key(|game| game.game);

    let first_result = aggregate_for(Competitor::FirstCheckpoint, &games);
    let second_result = aggregate_for(Competitor::SecondCheckpoint, &games);
    let total_plies = games.iter().map(|game| game.plies).sum::<usize>();
    let duration_seconds = duration.as_secs_f64();
    let report = BattleReport {
        schema_version: BATTLE_SCHEMA_VERSION,
        config: &args,
        first_checkpoint: first_snapshot.descriptor(),
        second_checkpoint: second_snapshot.descriptor(),
        first_temperature,
        second_temperature,
        duration_seconds,
        games_per_second: games.len() as f64 / duration_seconds,
        total_plies,
        average_plies: total_plies as f64 / games.len() as f64,
        first_checkpoint_inference: InferenceAggregate {
            evaluations_per_second: first_batch_stats.requests as f64 / duration_seconds,
            batch: first_batch_stats,
        },
        second_checkpoint_inference: InferenceAggregate {
            evaluations_per_second: second_batch_stats.requests as f64 / duration_seconds,
            batch: second_batch_stats,
        },
        first_checkpoint_result: first_result,
        second_checkpoint_result: second_result,
        games,
    };

    tracing::info!(
        games = report.games.len(),
        first_wins = report.first_checkpoint_result.wins,
        second_wins = report.second_checkpoint_result.wins,
        draws = report.first_checkpoint_result.draws,
        first_score_rate = report.first_checkpoint_result.score_rate,
        first_elo_difference = report.first_checkpoint_result.elo_difference,
        average_plies = report.average_plies,
        duration_seconds,
        games_per_second = report.games_per_second,
        "battle series complete"
    );
    write_report(&report, args.output.as_deref())?;
    if !args.no_move_logs {
        log_moves(&report.games);
    }
    tracing::info!(result = %serde_json::to_string(&report)?, "BATTLE_RESULT");
    Ok(())
}

fn validate_args(args: &BattleArgs) -> Result<()> {
    ensure!(args.games > 0, "games must be greater than zero");
    ensure!(
        args.games_parallelism > 0,
        "games-parallelism must be greater than zero"
    );
    ensure!(
        args.inference_batch_size > 0,
        "inference-batch-size must be greater than zero"
    );
    ensure!(
        args.simulations > 0,
        "simulations must be greater than zero"
    );
    ensure!(
        args.c_puct.is_finite() && args.c_puct >= 0.0,
        "c-puct must be finite and non-negative"
    );
    for (name, temperature) in [
        ("temperature", Some(args.temperature)),
        ("first-temperature", args.first_temperature),
        ("second-temperature", args.second_temperature),
    ] {
        if let Some(temperature) = temperature {
            ensure!(
                temperature.is_finite() && temperature >= 0.0,
                "{name} must be finite and non-negative"
            );
        }
    }
    Ok(())
}

fn derive_seed(base: u64, game: u64, competitor: u64) -> u64 {
    let mut value = base
        ^ game.wrapping_mul(0x9e37_79b9_7f4a_7c15)
        ^ competitor.wrapping_mul(0xd1b5_4a32_d192_ed03);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn game_log(game: usize, first_seat: Competitor, record: MatchRecord<BoardState>) -> GameLog {
    let second_seat = first_seat.other();
    let first_checkpoint_seat = if first_seat == Competitor::FirstCheckpoint {
        Seat::First
    } else {
        Seat::Second
    };
    let value_for_first_checkpoint = record.value_for(first_checkpoint_seat);
    let winner = match value_for_first_checkpoint {
        value if value > 0.0 => Some(Competitor::FirstCheckpoint),
        value if value < 0.0 => Some(Competitor::SecondCheckpoint),
        _ => None,
    };
    let moves = record
        .plies
        .into_iter()
        .enumerate()
        .map(|(ply, turn)| {
            let (row, column) = turn.action.to_xy();
            let checkpoint = match turn.actor {
                Seat::First => first_seat,
                Seat::Second => second_seat,
            };
            MoveLog {
                ply: ply + 1,
                checkpoint,
                seat: seat_name(turn.actor),
                row: row + 1,
                column: column + 1,
                value_estimate: turn.decision.diagnostics.value_estimate,
            }
        })
        .collect::<Vec<_>>();
    GameLog {
        game: game + 1,
        first_seat,
        second_seat,
        winner,
        value_for_first_checkpoint,
        plies: moves.len(),
        moves,
    }
}

fn seat_name(seat: Seat) -> &'static str {
    match seat {
        Seat::First => "first",
        Seat::Second => "second",
    }
}

fn aggregate_for(competitor: Competitor, games: &[GameLog]) -> CompetitorAggregate {
    let mut wins = 0;
    let mut losses = 0;
    let mut draws = 0;
    let mut as_first_seat = SeatAggregate::default();
    let mut as_second_seat = SeatAggregate::default();
    for game in games {
        let seat = if game.first_seat == competitor {
            &mut as_first_seat
        } else {
            &mut as_second_seat
        };
        seat.games += 1;
        match game.winner {
            Some(winner) if winner == competitor => {
                wins += 1;
                seat.wins += 1;
            }
            Some(_) => {
                losses += 1;
                seat.losses += 1;
            }
            None => {
                draws += 1;
                seat.draws += 1;
            }
        }
    }

    let score = wins as f64 + draws as f64 * 0.5;
    let score_rate = score / games.len() as f64;
    let (score_rate_95_percent_low, score_rate_95_percent_high) =
        wilson_score_interval(score_rate, games.len());
    let elo_difference = if (0.0..1.0).contains(&score_rate) {
        Some(400.0 * (score_rate / (1.0 - score_rate)).log10())
    } else {
        None
    };
    CompetitorAggregate {
        wins,
        losses,
        draws,
        score,
        score_rate,
        score_rate_95_percent_low,
        score_rate_95_percent_high,
        elo_difference,
        as_first_seat,
        as_second_seat,
    }
}

fn wilson_score_interval(score_rate: f64, games: usize) -> (f64, f64) {
    let n = games as f64;
    let z = 1.959_963_984_540_054;
    let z_squared = z * z;
    let denominator = 1.0 + z_squared / n;
    let center = (score_rate + z_squared / (2.0 * n)) / denominator;
    let margin =
        z * ((score_rate * (1.0 - score_rate) + z_squared / (4.0 * n)) / n).sqrt() / denominator;
    ((center - margin).max(0.0), (center + margin).min(1.0))
}

fn log_game_summary(game: &GameLog) {
    tracing::info!(
        game = game.game,
        first_seat = game.first_seat.name(),
        second_seat = game.second_seat.name(),
        winner = game.winner.map(Competitor::name).unwrap_or("draw"),
        value_for_first_checkpoint = game.value_for_first_checkpoint,
        plies = game.plies,
        "battle game complete"
    );
}

fn log_progress(
    update: &'static str,
    games_completed: usize,
    games_total: usize,
    elapsed: Duration,
    first_evaluations: u64,
    second_evaluations: u64,
) {
    let elapsed_seconds = elapsed.as_secs_f64();
    let evaluations = first_evaluations + second_evaluations;
    tracing::info!(
        update,
        games_completed,
        games_total,
        evaluations,
        elapsed_seconds,
        games_per_second = games_completed as f64 / elapsed_seconds,
        evaluations_per_second = evaluations as f64 / elapsed_seconds,
        "battle progress"
    );
}

fn log_moves(games: &[GameLog]) {
    for game in games {
        for turn in &game.moves {
            tracing::info!(
                game = game.game,
                ply = turn.ply,
                checkpoint = turn.checkpoint.name(),
                seat = turn.seat,
                row = turn.row,
                column = turn.column,
                value_estimate = turn.value_estimate,
                "battle move"
            );
        }
    }
}

fn write_report(report: &BattleReport<'_>, output: Option<&Path>) -> Result<()> {
    let Some(path) = output else {
        return Ok(());
    };
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)
            .with_context(|| format!("creating battle output {}", parent.display()))?;
    }
    let mut file = fs::File::create(path)
        .with_context(|| format!("creating battle output {}", path.display()))?;
    serde_json::to_writer_pretty(&mut file, report)?;
    use std::io::Write as _;
    writeln!(file)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn game(game: usize, first_seat: Competitor, winner: Option<Competitor>) -> GameLog {
        GameLog {
            game,
            first_seat,
            second_seat: first_seat.other(),
            winner,
            value_for_first_checkpoint: match winner {
                Some(Competitor::FirstCheckpoint) => 1.0,
                Some(Competitor::SecondCheckpoint) => -1.0,
                None => 0.0,
            },
            plies: 1,
            moves: Vec::new(),
        }
    }

    #[test]
    fn aggregation_tracks_checkpoint_identity_across_seat_swaps() {
        let games = vec![
            game(
                0,
                Competitor::FirstCheckpoint,
                Some(Competitor::FirstCheckpoint),
            ),
            game(
                1,
                Competitor::SecondCheckpoint,
                Some(Competitor::FirstCheckpoint),
            ),
            game(2, Competitor::FirstCheckpoint, None),
            game(
                3,
                Competitor::SecondCheckpoint,
                Some(Competitor::SecondCheckpoint),
            ),
        ];

        let first = aggregate_for(Competitor::FirstCheckpoint, &games);
        assert_eq!((first.wins, first.losses, first.draws), (2, 1, 1));
        assert_eq!(first.score_rate, 0.625);
        assert_eq!(first.as_first_seat.games, 2);
        assert_eq!(first.as_second_seat.games, 2);

        let second = aggregate_for(Competitor::SecondCheckpoint, &games);
        assert_eq!((second.wins, second.losses, second.draws), (1, 2, 1));
        assert_eq!(second.score_rate, 0.375);
    }

    #[test]
    fn derived_seeds_depend_on_game_and_checkpoint() {
        assert_ne!(derive_seed(7, 0, 0), derive_seed(7, 0, 1));
        assert_ne!(derive_seed(7, 0, 0), derive_seed(7, 1, 0));
        assert_eq!(derive_seed(7, 3, 1), derive_seed(7, 3, 1));
    }

    #[test]
    fn confidence_interval_retains_uncertainty_for_a_sweep() {
        let (low, high) = wilson_score_interval(1.0, 20);
        assert!(low < 1.0);
        assert_eq!(high, 1.0);
    }
}
