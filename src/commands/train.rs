use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use alz::{
    alpha_zero::{
        AlphaZeroNet, ExecutorScope, NetworkBatchStats, NetworkPositionEvaluator, PositionCodec,
        Seat, TrainingCodec, extract_training_game, generate_self_played_game,
    },
    tictactoe::{BoardState, TicTacToeCodec, TicTacToeResNet, generate_game_image},
};
use anyhow::{Context, Result, ensure};
use rand::{
    SeedableRng,
    rngs::SmallRng,
    seq::{IteratorRandom, SliceRandom},
};
use serde::Serialize;
use tch::{
    Kind, Reduction, Tensor,
    nn::{self, Optimizer, OptimizerConfig},
};

use crate::{
    cli::TrainArgs,
    training_snapshot::{
        ReplayBuffer, ReplayGame, find_latest_snapshot, load_training_snapshot,
        save_training_snapshot,
    },
};

use super::common::resolve_device;

const DEFAULT_LEARNING_RATE: f64 = 1e-3;
const DEFAULT_WEIGHT_DECAY: f64 = 1e-4;
const METRICS_SCHEMA_VERSION: u32 = 2;

#[derive(Clone, Copy, Debug)]
pub(super) struct SelfPlaySettings {
    pub games: usize,
    pub simulations: usize,
    pub c_puct: f32,
    pub inference_batch_size: usize,
    pub games_parallelism: usize,
    pub batch_timeout: Duration,
    pub seed: u64,
    pub progress_every_games: usize,
    pub heartbeat_interval: Duration,
}

pub(super) struct EpochGames {
    pub games: Vec<ReplayGame>,
    pub total_score: f32,
    pub total_length: usize,
    pub batch_stats: NetworkBatchStats,
    pub duration: Duration,
}

#[derive(Clone, Debug, Serialize)]
struct TrainingStats {
    samples: usize,
    batches: usize,
    value_loss: f64,
    policy_loss: f64,
    total_loss: f64,
    samples_per_second: f64,
    duration_seconds: f64,
}

#[derive(Serialize)]
struct EpochStats<'a> {
    schema_version: u32,
    epoch: usize,
    config: &'a TrainArgs,
    games: usize,
    total_score: f32,
    average_score: f32,
    total_game_length: usize,
    average_game_length: f32,
    replay_games: usize,
    replay_positions: usize,
    evaluations_per_second: f64,
    moves_per_second: f64,
    games_per_second: f64,
    self_play_seconds: f64,
    network: &'a NetworkBatchStats,
    network_average_batch_size: f64,
    network_average_queue_wait_us: f64,
    network_average_request_latency_us: f64,
    network_average_service_us: f64,
    training: &'a TrainingStats,
    checkpoint_seconds: f64,
    rendering_seconds: f64,
    epoch_seconds: f64,
}

pub async fn run(args: TrainArgs) -> Result<()> {
    validate_args(&args)?;
    ensure!(
        !args.model.checkpoint_dir.join("metadata.json").is_file(),
        "training checkpoint-dir must be a run directory, not an individual snapshot"
    );

    fs::create_dir_all(&args.model.checkpoint_dir)
        .with_context(|| format!("creating {}", args.model.checkpoint_dir.display()))?;
    fs::create_dir_all(&args.games_dir)
        .with_context(|| format!("creating {}", args.games_dir.display()))?;
    fs::create_dir_all(&args.stats_dir)
        .with_context(|| format!("creating {}", args.stats_dir.display()))?;
    write_invocation_config(&args.stats_dir, &args)?;

    tch::manual_seed((args.seed & i64::MAX as u64) as i64);
    let device = resolve_device(&args.model.device)?;
    let mut var_store = nn::VarStore::new(device);
    let mut network = TicTacToeResNet::new(var_store.root());
    let learning_rate = args.learning_rate.unwrap_or(DEFAULT_LEARNING_RATE);
    let weight_decay = args.weight_decay.unwrap_or(DEFAULT_WEIGHT_DECAY);
    let mut optimizer = nn::Adam::default()
        .wd(weight_decay)
        .build(&var_store, learning_rate)?;

    let mut replay = ReplayBuffer::new();
    let mut start_epoch = 0;
    if let Some(snapshot) = find_latest_snapshot(&args.model.checkpoint_dir)? {
        let epoch = snapshot.epoch();
        tracing::info!(
            snapshot_epoch = epoch,
            "restoring complete training snapshot"
        );
        replay = load_training_snapshot(&snapshot, &mut var_store, &mut optimizer)?;
        if let Some(learning_rate) = args.learning_rate {
            optimizer.set_lr(learning_rate);
        }
        if let Some(weight_decay) = args.weight_decay {
            optimizer.set_weight_decay(weight_decay);
        }
        start_epoch = epoch + 1;
    }

    let target_epoch_count = args.epochs.unwrap_or(usize::MAX);
    if start_epoch >= target_epoch_count {
        tracing::info!(
            target_epoch_count,
            latest_completed_epoch = start_epoch.saturating_sub(1),
            "training already reached target epoch count"
        );
    }
    for epoch in start_epoch..target_epoch_count {
        let epoch_started = Instant::now();
        let settings = SelfPlaySettings {
            games: args.games_per_epoch,
            simulations: args.simulations,
            c_puct: args.c_puct,
            inference_batch_size: args.inference_batch_size,
            games_parallelism: args.games_parallelism,
            batch_timeout: Duration::from_micros(args.batch_timeout_us),
            seed: derive_seed(args.seed, epoch as u64),
            progress_every_games: args.progress_every_games,
            heartbeat_interval: Duration::from_secs(args.heartbeat_seconds),
        };
        let (returned_network, epoch_games) =
            collect_epoch_games(network, settings, var_store.device()).await?;
        network = returned_network;

        let games_in_epoch = epoch_games.games.len();
        let total_score = epoch_games.total_score;
        let total_length = epoch_games.total_length;
        let avg_score = total_score / games_in_epoch as f32;
        let avg_length = total_length as f32 / games_in_epoch as f32;
        let self_play_seconds = epoch_games.duration.as_secs_f64();
        let games_per_second = games_in_epoch as f64 / self_play_seconds;
        let moves_per_second = total_length as f64 / self_play_seconds;
        let evaluations_per_second = epoch_games.batch_stats.requests as f64 / self_play_seconds;
        tracing::info!(
            epoch,
            games = games_in_epoch,
            moves = total_length,
            evaluations = epoch_games.batch_stats.requests,
            average_score = avg_score,
            average_game_length = avg_length,
            self_play_seconds,
            games_per_second,
            moves_per_second,
            evaluations_per_second,
            average_batch_size = epoch_games.batch_stats.average_batch_size(),
            "self-play complete"
        );

        let mut render_rng =
            SmallRng::seed_from_u64(derive_seed(args.seed, epoch as u64 ^ 0x5245_4e44_4552));
        let sample_games = epoch_games
            .games
            .iter()
            .sample(
                &mut render_rng,
                args.rendered_games.min(epoch_games.games.len()),
            )
            .into_iter()
            .cloned()
            .collect::<Vec<_>>();

        replay.extend(epoch_games.games);
        while replay.len() > args.replay_games {
            replay.pop_front();
        }

        let training_stats = train_epoch(
            &network,
            &mut optimizer,
            &replay,
            args.training_batch_size,
            var_store.device(),
            derive_seed(args.seed, epoch as u64 ^ 0x0054_5241_494e),
        )?;
        tracing::info!(
            epoch,
            value_loss = training_stats.value_loss,
            policy_loss = training_stats.policy_loss,
            total_loss = training_stats.total_loss,
            samples = training_stats.samples,
            samples_per_second = training_stats.samples_per_second,
            duration_seconds = training_stats.duration_seconds,
            "training complete"
        );

        let checkpoint_started = Instant::now();
        save_training_snapshot(
            &args.model.checkpoint_dir,
            epoch,
            &var_store,
            &optimizer,
            &replay,
        )?;
        let checkpoint_duration = checkpoint_started.elapsed();

        let rendering_started = Instant::now();
        render_sample_games(&args.games_dir, epoch, sample_games)?;
        let rendering_duration = rendering_started.elapsed();

        let epoch_duration = epoch_started.elapsed();
        let replay_positions = replay.iter().map(Vec::len).sum::<usize>();
        let stats = EpochStats {
            schema_version: METRICS_SCHEMA_VERSION,
            epoch,
            config: &args,
            games: games_in_epoch,
            total_score,
            average_score: avg_score,
            total_game_length: total_length,
            average_game_length: avg_length,
            replay_games: replay.len(),
            replay_positions,
            evaluations_per_second,
            moves_per_second,
            games_per_second,
            self_play_seconds,
            network: &epoch_games.batch_stats,
            network_average_batch_size: epoch_games.batch_stats.average_batch_size(),
            network_average_queue_wait_us: epoch_games.batch_stats.average_queue_wait_us(),
            network_average_request_latency_us: epoch_games
                .batch_stats
                .average_request_latency_us(),
            network_average_service_us: epoch_games.batch_stats.average_service_us(),
            training: &training_stats,
            checkpoint_seconds: checkpoint_duration.as_secs_f64(),
            rendering_seconds: rendering_duration.as_secs_f64(),
            epoch_seconds: epoch_duration.as_secs_f64(),
        };
        write_stats(&args.stats_dir, epoch, &stats)?;
        tracing::info!(
            epoch,
            epoch_seconds = stats.epoch_seconds,
            checkpoint_seconds = stats.checkpoint_seconds,
            rendering_seconds = stats.rendering_seconds,
            "epoch complete"
        );
        tracing::debug!(metrics = %serde_json::to_string(&stats)?, "epoch metrics");
    }

    Ok(())
}

pub(super) async fn collect_epoch_games(
    network: TicTacToeResNet,
    settings: SelfPlaySettings,
    device: tch::Device,
) -> Result<(TicTacToeResNet, EpochGames)> {
    ensure!(settings.games > 0, "games must be greater than zero");
    ensure!(
        settings.simulations > 0,
        "simulations must be greater than zero"
    );
    ensure!(
        settings.inference_batch_size > 0,
        "inference batch size must be greater than zero"
    );
    ensure!(
        settings.games_parallelism > 0,
        "games parallelism must be greater than zero"
    );

    let started = Instant::now();
    let mut executor = ExecutorScope::new(
        network,
        settings.games_parallelism,
        settings.inference_batch_size,
        settings.batch_timeout,
        (Kind::Float, device),
    );

    for game_index in 0..settings.games {
        let simulations = settings.simulations;
        let c_puct = settings.c_puct;
        let game_seed = derive_seed(settings.seed, game_index as u64);
        std::mem::drop(executor.spawn(move |handle| async move {
            let evaluator =
                NetworkPositionEvaluator::<TicTacToeResNet, TicTacToeCodec>::new(handle);
            generate_self_played_game(
                BoardState::new(),
                simulations,
                c_puct,
                self_play_temperature,
                evaluator,
                SmallRng::seed_from_u64(game_seed),
            )
            .await
        }));
    }

    let mut games = Vec::with_capacity(settings.games);
    let mut total_score = 0.0;
    let mut total_length = 0;
    let mut heartbeat = (settings.heartbeat_interval > Duration::ZERO)
        .then(|| tokio::time::interval(settings.heartbeat_interval));
    if let Some(timer) = heartbeat.as_mut() {
        timer.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        timer.tick().await;
    }

    loop {
        let maybe_record = if let Some(timer) = heartbeat.as_mut() {
            tokio::select! {
                record = executor.next() => record,
                _ = timer.tick() => {
                    let elapsed_seconds = started.elapsed().as_secs_f64();
                    let completed_evaluations = executor.completed_evaluations();
                    tracing::info!(
                        update = "heartbeat",
                        games_completed = games.len(),
                        games_total = settings.games,
                        unfinished_games = executor.len(),
                        completed_moves = total_length,
                        completed_evaluations,
                        elapsed_seconds,
                        games_per_second = games.len() as f64 / elapsed_seconds,
                        moves_per_second = total_length as f64 / elapsed_seconds,
                        evaluations_per_second = completed_evaluations as f64 / elapsed_seconds,
                        "self-play progress"
                    );
                    continue;
                }
            }
        } else {
            executor.next().await
        };
        let Some(record) = maybe_record else {
            break;
        };
        let record = record?;
        total_score += record.value_for(Seat::First);
        total_length += record.plies.len();
        games.push(extract_training_game::<_, TicTacToeCodec>(record)?);
        if settings.progress_every_games > 0
            && (games.len() % settings.progress_every_games == 0 || games.len() == settings.games)
        {
            let elapsed_seconds = started.elapsed().as_secs_f64();
            let completed_evaluations = executor.completed_evaluations();
            tracing::info!(
                update = "completion",
                games_completed = games.len(),
                games_total = settings.games,
                unfinished_games = executor.len(),
                completed_moves = total_length,
                completed_evaluations,
                elapsed_seconds,
                games_per_second = games.len() as f64 / elapsed_seconds,
                moves_per_second = total_length as f64 / elapsed_seconds,
                evaluations_per_second = completed_evaluations as f64 / elapsed_seconds,
                "self-play progress"
            );
        }
    }

    let (network, batch_stats) = executor.join_with_stats().await;
    Ok((
        network,
        EpochGames {
            games,
            total_score,
            total_length,
            batch_stats,
            duration: started.elapsed(),
        },
    ))
}

fn train_epoch(
    network: &TicTacToeResNet,
    optimizer: &mut Optimizer,
    replay: &ReplayBuffer,
    batch_size: usize,
    device: tch::Device,
    seed: u64,
) -> Result<TrainingStats> {
    let started = Instant::now();
    let augmentation_count = TicTacToeCodec::augmentation_count();
    let mut training_samples = replay
        .iter()
        .flatten()
        .flat_map(|sample| (0..augmentation_count).map(move |augmentation| (sample, augmentation)))
        .collect::<Vec<_>>();
    training_samples.shuffle(&mut SmallRng::seed_from_u64(seed));

    let mut total_value_loss = 0.0f64;
    let mut total_policy_loss = 0.0f64;
    let mut batches = 0;
    for chunk in training_samples.chunks(batch_size) {
        let mut states = Vec::with_capacity(chunk.len());
        let mut policies = Vec::with_capacity(chunk.len());
        let mut values = Vec::with_capacity(chunk.len());
        for &(sample, augmentation) in chunk {
            let state = TicTacToeCodec::encode_position(&sample.state);
            let policy = TicTacToeCodec::policy_to_tensor(&sample.policy);
            let (state, policy) = TicTacToeCodec::augment(&state, &policy, augmentation);
            states.push(state);
            policies.push(policy);
            values.push(sample.value);
        }

        let states = Tensor::stack(&states, 0).to_kind(Kind::Float).to(device);
        let policies = Tensor::stack(&policies, 0).to_kind(Kind::Float).to(device);
        let values = Tensor::from_slice(&values).to_kind(Kind::Float).to(device);

        let (expected_values, expected_policies) = network.forward_t(&states, true);
        let value_loss = expected_values.mse_loss(&values, Reduction::Mean);
        let policy_loss = -(policies * expected_policies).sum(None) / chunk.len() as f64;
        optimizer.backward_step(&(&value_loss + &policy_loss));

        let chunk_len = chunk.len() as f64;
        total_value_loss +=
            f32::try_from(&value_loss).context("reading value loss")? as f64 * chunk_len;
        total_policy_loss +=
            f32::try_from(&policy_loss).context("reading policy loss")? as f64 * chunk_len;
        batches += 1;
    }

    let samples = training_samples.len();
    let duration = started.elapsed();
    let value_loss = total_value_loss / samples as f64;
    let policy_loss = total_policy_loss / samples as f64;
    Ok(TrainingStats {
        samples,
        batches,
        value_loss,
        policy_loss,
        total_loss: value_loss + policy_loss,
        samples_per_second: samples as f64 / duration.as_secs_f64(),
        duration_seconds: duration.as_secs_f64(),
    })
}

fn render_sample_games(games_dir: &Path, epoch: usize, games: Vec<ReplayGame>) -> Result<()> {
    for (id, game) in games.into_iter().enumerate() {
        generate_game_image(&game).save(game_path(games_dir, epoch, id))?;
    }
    Ok(())
}

fn write_invocation_config(stats_dir: &Path, args: &TrainArgs) -> Result<()> {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let path = stats_dir.join(format!(
        "invocation-{timestamp}-{}.json",
        std::process::id()
    ));
    let mut file = fs::File::create(&path)
        .with_context(|| format!("creating invocation config {}", path.display()))?;
    serde_json::to_writer_pretty(&mut file, args)?;
    writeln!(file)?;
    Ok(())
}

fn write_stats(stats_dir: &Path, epoch: usize, stats: &EpochStats<'_>) -> Result<()> {
    let path = stats_path(stats_dir, epoch);
    let mut file =
        fs::File::create(&path).with_context(|| format!("creating stats {}", path.display()))?;
    serde_json::to_writer_pretty(&mut file, stats)?;
    writeln!(file)?;
    file.flush()?;

    let jsonl_path = stats_dir.join("epochs.jsonl");
    let mut jsonl = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&jsonl_path)
        .with_context(|| format!("opening stats stream {}", jsonl_path.display()))?;
    serde_json::to_writer(&mut jsonl, stats)?;
    writeln!(jsonl)?;
    jsonl.flush()?;
    Ok(())
}

fn game_path(root: &Path, epoch: usize, id: usize) -> PathBuf {
    root.join(format!("{epoch:08}.{id:02}.png"))
}

fn stats_path(root: &Path, epoch: usize) -> PathBuf {
    root.join(format!("{epoch:08}.json"))
}

fn self_play_temperature(turn: usize) -> f32 {
    match turn {
        0..8 => 1.0,
        value @ 8..13 => 1.0 - (value - 7) as f32 * 0.1,
        13.. => 0.5,
    }
}

pub(super) fn derive_seed(base: u64, stream: u64) -> u64 {
    let mut value = base ^ stream.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn validate_args(args: &TrainArgs) -> Result<()> {
    ensure!(
        args.games_per_epoch > 0,
        "games-per-epoch must be greater than zero"
    );
    ensure!(
        args.simulations > 0,
        "simulations must be greater than zero"
    );
    ensure!(
        args.c_puct.is_finite() && args.c_puct >= 0.0,
        "c-puct must be finite and non-negative"
    );
    ensure!(
        args.replay_games > 0,
        "replay-games must be greater than zero"
    );
    ensure!(
        args.inference_batch_size > 0,
        "inference-batch-size must be greater than zero"
    );
    ensure!(
        args.games_parallelism > 0,
        "games-parallelism must be greater than zero"
    );
    ensure!(
        args.games_parallelism <= tokio::sync::Semaphore::MAX_PERMITS,
        "games-parallelism exceeds Tokio's semaphore limit"
    );
    ensure!(
        args.training_batch_size > 0,
        "training-batch-size must be greater than zero"
    );
    ensure!(
        args.learning_rate
            .is_none_or(|learning_rate| learning_rate.is_finite() && learning_rate > 0.0),
        "learning-rate must be finite and positive"
    );
    ensure!(
        args.weight_decay
            .is_none_or(|weight_decay| weight_decay.is_finite() && weight_decay >= 0.0),
        "weight-decay must be finite and non-negative"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::derive_seed;

    #[test]
    fn derived_seeds_are_stable_and_stream_specific() {
        assert_eq!(derive_seed(7, 11), derive_seed(7, 11));
        assert_ne!(derive_seed(7, 11), derive_seed(7, 12));
        assert_ne!(derive_seed(7, 11), derive_seed(8, 11));
    }
}
