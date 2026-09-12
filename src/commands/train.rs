use std::{
    collections::VecDeque,
    fs::{self, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use alz::{
    engine::{
        AlphaZeroNet, ExecutorScope, NetworkBatchStats, NetworkPositionEvaluator, Seat,
        apply_temperature, extract_training_game, generate_self_played_game_with_top_p,
        policy_log_probabilities,
    },
    gomoku::{BoardState, GomokuCodec, GomokuModel, ModelSpec, generate_game_image},
    training_batches::TrainingBatches,
    training_snapshot::{
        ReplayBuffer, ReplayGame, find_latest_snapshot, load_replay_checkpoint,
        load_training_snapshot, save_training_snapshot,
    },
};
use anyhow::{Context, Result, ensure};
use rand::{
    SeedableRng,
    rngs::SmallRng,
    seq::{IteratorRandom, SliceRandom},
};
use serde::Serialize;
use tch::{
    Kind, Reduction,
    nn::{self, Optimizer},
};

use crate::cli::{InferenceSymmetryChoice, TrainArgs};

use super::common::{resolve_device, resolve_training_architecture};

const DEFAULT_LEARNING_RATE: f64 = 1e-3;
const DEFAULT_WEIGHT_DECAY: f64 = 1e-4;
const METRICS_SCHEMA_VERSION: u32 = 7;
// Describes newly generated samples; resumed replay may still contain legacy targets.
const NEW_GAME_POLICY_TARGET: &str = "normalized_search_visits_v1";
const SELF_PLAY_TEMPERATURE_SCHEDULE: &str = "paired_moves_1_to_6_t1_19_to_20_t0.7_v1";
const REPLAY_SHUFFLE_STREAM: u64 = 0x0000_5245_504c_4159;
const INFERENCE_SYMMETRY_STREAM: u64 = 0x5359_4d4d_4554_5259;

#[derive(Clone, Copy, Debug)]
pub(super) struct SelfPlaySettings {
    pub top_p: f64,
    pub games: usize,
    pub simulations: usize,
    pub c_puct: f32,
    pub inference_batch_size: usize,
    pub inference_symmetry: InferenceSymmetryChoice,
    pub games_parallelism: usize,
    pub batch_timeout: Duration,
    pub seed: u64,
    pub progress_every_games: usize,
    pub heartbeat_interval: Duration,
}

pub(super) struct EpochGames {
    pub sampling: SamplingStats,
    pub games: Vec<ReplayGame>,
    pub total_score: f32,
    pub total_length: usize,
    pub batch_stats: NetworkBatchStats,
    pub duration: Duration,
}

#[derive(Default, Serialize)]
pub(super) struct SamplingStats {
    positions: usize,
    nonzero_before: usize,
    nonzero_retained: usize,
    deterministic_positions: usize,
    removed_probability_mass: f64,
    sampling_entropy_sum: f64,
}

#[derive(Clone, Debug, Serialize)]
pub(super) struct TrainingStats {
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
    model: &'a ModelSpec,
    config: &'a TrainArgs,
    games: usize,
    total_score: f32,
    average_score: f32,
    total_game_length: usize,
    average_game_length: f32,
    replay_games: usize,
    replay_positions: usize,
    replay_position_capacity: Option<usize>,
    new_game_policy_target: &'static str,
    self_play_temperature_schedule: &'static str,
    sampling: &'a SamplingStats,
    scheduled_learning_rate: Option<f64>,
    evaluations_per_second: f64,
    moves_per_second: f64,
    games_per_second: f64,
    self_play_seconds: f64,
    network: &'a NetworkBatchStats,
    network_average_batch_size: f64,
    network_average_queue_wait_us: f64,
    network_average_request_latency_us: f64,
    network_average_service_us: f64,
    network_average_position_encoding_us: f64,
    network_average_policy_mask_construction_us: f64,
    network_average_policy_mask_batch_construction_us: f64,
    network_average_policy_mask_transfer_submission_us: f64,
    network_average_policy_postprocess_submission_us: f64,
    network_average_policy_decode_us: f64,
    training: &'a TrainingStats,
    checkpoint_seconds: f64,
    rendering_seconds: f64,
    epoch_seconds: f64,
}

pub async fn run(args: TrainArgs) -> Result<()> {
    args.performance.validate()?;
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
    let snapshot = find_latest_snapshot(&args.model.checkpoint_dir)?;
    let model_spec = resolve_training_architecture(
        args.model.architecture,
        snapshot.as_ref().map(|snapshot| snapshot.model_spec()),
    )?;
    validate_replay_history(&args)?;
    write_invocation_config(&args.stats_dir, &args, &model_spec)?;

    tch::manual_seed((args.seed & i64::MAX as u64) as i64);
    let device = resolve_device(&args.model.device)?;
    let mut var_store = nn::VarStore::new(device);
    let mut network = GomokuModel::new(var_store.root(), &model_spec);
    if snapshot.is_none() && args.bn_gamma_one {
        super::train_replay::initialize_bn_gamma_one(&var_store)?;
    }
    let learning_rate = args.learning_rate.unwrap_or(DEFAULT_LEARNING_RATE);
    let weight_decay = args.weight_decay.unwrap_or(DEFAULT_WEIGHT_DECAY);
    let mut optimizer = super::common::build_adam(
        args.performance.adam_backend,
        &var_store,
        learning_rate,
        weight_decay,
    )?;

    let mut replay = ReplayBuffer::new();
    let mut start_epoch = 0;
    if let Some(snapshot) = snapshot {
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
        if epoch < args.replay_history_epochs.unwrap_or(0) {
            let source = args
                .replay_history_dir
                .as_ref()
                .context("missing history directory")?;
            let source_checkpoint = source.join("checkpoints").join(format!("{epoch:08}"));
            let (descriptor, saved_replay) = load_replay_checkpoint(&source_checkpoint)?;
            let source_stats = source.join("stats").join(format!("{epoch:08}.json"));
            let original: serde_json::Value =
                serde_json::from_reader(fs::File::open(&source_stats)?)?;
            replay = saved_replay; // Exact saved order, no pooling, splitting, reshuffling, or trimming.
            let rate = scheduled_learning_rate(&args, epoch)?.unwrap_or(learning_rate);
            optimizer.set_lr(rate);
            tracing::info!(epoch, learning_rate=rate, source=%source_checkpoint.display(), "starting replay-history epoch");
            let data = TrainingBatches::new(&replay, args.performance.replay_cache, device)?;
            ensure!(
                original["training"]["samples"].as_u64() == Some(data.len() as u64),
                "history sample count differs at epoch {epoch}"
            );
            let training = train_epoch(
                &network,
                &mut optimizer,
                &data,
                args.training_batch_size,
                device,
                derive_seed(args.seed, epoch as u64 ^ 0x0054_5241_494e),
                args.performance.prefetch_batches,
            )?;
            drop(data);
            let mut stats = serde_json::json!({
                "schema_version": METRICS_SCHEMA_VERSION, "epoch":epoch, "model":model_spec,
                "config":args, "training":training, "self_play_seconds":null,
                "epoch_seconds":epoch_started.elapsed().as_secs_f64(),
                "history_source":descriptor, "history_stats_sha256":file_sha256(&source_stats)?,
                "history_replay_sha256":file_sha256(&source_checkpoint.join("replay.bin.zst"))?,
                "scheduled_learning_rate":rate,
                "provenance":"retrained on another model's saved epoch buffer; no new self-play games"
            });
            for key in [
                "games",
                "total_score",
                "average_score",
                "total_game_length",
                "average_game_length",
                "replay_games",
                "replay_positions",
                "replay_position_capacity",
                "new_game_policy_target",
                "self_play_temperature_schedule",
            ] {
                stats[key] = original[key].clone();
            }
            // Publish metrics first, then the atomic snapshot used as the completion marker.
            fs::write(
                stats_path(&args.stats_dir, epoch),
                serde_json::to_vec_pretty(&stats)?,
            )?;
            save_training_snapshot(
                &args.model.checkpoint_dir,
                epoch,
                &model_spec,
                &var_store,
                &optimizer,
                &replay,
            )?;
            tracing::info!(
                epoch,
                samples = training.samples,
                duration_seconds = training.duration_seconds,
                value_loss = training.value_loss,
                policy_loss = training.policy_loss,
                "replay-history epoch complete"
            );
            continue;
        }
        let settings = SelfPlaySettings {
            top_p: args.top_p,
            games: args.games_per_epoch,
            simulations: args.simulations,
            c_puct: args.c_puct,
            inference_batch_size: args.inference_batch_size,
            inference_symmetry: args.inference_symmetry,
            games_parallelism: args.games_parallelism,
            batch_timeout: Duration::from_micros(args.batch_timeout_us),
            seed: derive_seed(args.seed, epoch as u64),
            progress_every_games: args.progress_every_games,
            heartbeat_interval: Duration::from_secs(args.heartbeat_seconds),
        };
        let (returned_network, mut epoch_games) =
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
            average_position_encoding_us = epoch_games.batch_stats.average_position_encoding_us(),
            average_policy_mask_construction_us = epoch_games
                .batch_stats
                .average_policy_mask_construction_us(),
            average_policy_mask_batch_construction_us = epoch_games
                .batch_stats
                .average_policy_mask_batch_construction_us(),
            average_policy_mask_transfer_submission_us = epoch_games
                .batch_stats
                .average_policy_mask_transfer_submission_us(),
            average_policy_decode_us = epoch_games.batch_stats.average_policy_decode_us(),
            average_policy_postprocess_submission_us = epoch_games
                .batch_stats
                .average_policy_postprocess_submission_us(),
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

        let replay_position_capacity = if args.replay_games.is_some() {
            None
        } else {
            Some(replay_position_capacity(&args, epoch)?)
        };
        let new_games = std::mem::take(&mut epoch_games.games);
        let replay_seed = derive_seed(args.seed, epoch as u64 ^ REPLAY_SHUFFLE_STREAM);
        if let Some(capacity) = args.replay_games {
            extend_replay_shuffled(&mut replay, new_games, capacity, replay_seed);
        } else {
            extend_replay_positions_shuffled(
                &mut replay,
                new_games,
                replay_position_capacity.expect("position capacity was calculated"),
                replay_seed,
            );
        }

        let scheduled_learning_rate = scheduled_learning_rate(&args, epoch)?;
        if let Some(rate) = scheduled_learning_rate {
            optimizer.set_lr(rate);
            tracing::info!(epoch, learning_rate = rate, "replay-linked learning rate");
        }
        let training_data = TrainingBatches::new(&replay, args.performance.replay_cache, device)?;
        let training_stats = train_epoch(
            &network,
            &mut optimizer,
            &training_data,
            args.training_batch_size,
            var_store.device(),
            derive_seed(args.seed, epoch as u64 ^ 0x0054_5241_494e),
            args.performance.prefetch_batches,
        )?;
        drop(training_data);
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
            &model_spec,
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
            model: &model_spec,
            config: &args,
            games: games_in_epoch,
            total_score,
            average_score: avg_score,
            total_game_length: total_length,
            average_game_length: avg_length,
            replay_games: replay.len(),
            replay_positions,
            replay_position_capacity,
            new_game_policy_target: NEW_GAME_POLICY_TARGET,
            self_play_temperature_schedule: SELF_PLAY_TEMPERATURE_SCHEDULE,
            sampling: &epoch_games.sampling,
            scheduled_learning_rate,
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
            network_average_position_encoding_us: epoch_games
                .batch_stats
                .average_position_encoding_us(),
            network_average_policy_mask_construction_us: epoch_games
                .batch_stats
                .average_policy_mask_construction_us(),
            network_average_policy_mask_batch_construction_us: epoch_games
                .batch_stats
                .average_policy_mask_batch_construction_us(),
            network_average_policy_mask_transfer_submission_us: epoch_games
                .batch_stats
                .average_policy_mask_transfer_submission_us(),
            network_average_policy_postprocess_submission_us: epoch_games
                .batch_stats
                .average_policy_postprocess_submission_us(),
            network_average_policy_decode_us: epoch_games.batch_stats.average_policy_decode_us(),
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
    network: GomokuModel,
    settings: SelfPlaySettings,
    device: tch::Device,
) -> Result<(GomokuModel, EpochGames)> {
    ensure!(settings.games > 0, "games must be greater than zero");
    ensure!(
        settings.top_p.is_finite() && settings.top_p > 0.0 && settings.top_p <= 1.0,
        "top-p must be in (0, 1]"
    );
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
        let top_p = settings.top_p;
        let inference_symmetry = settings.inference_symmetry;
        let game_seed = derive_seed(settings.seed, game_index as u64);
        std::mem::drop(executor.spawn(move |handle| async move {
            let evaluator = match inference_symmetry {
                InferenceSymmetryChoice::None => {
                    NetworkPositionEvaluator::<GomokuModel, GomokuCodec>::new(handle)
                }
                InferenceSymmetryChoice::Random => {
                    NetworkPositionEvaluator::<GomokuModel, GomokuCodec>::with_random_symmetry(
                        handle,
                        derive_seed(game_seed, INFERENCE_SYMMETRY_STREAM),
                    )
                }
            };
            generate_self_played_game_with_top_p(
                BoardState::new(),
                simulations,
                c_puct,
                self_play_temperature,
                evaluator,
                SmallRng::seed_from_u64(game_seed),
                top_p,
            )
            .await
        }));
    }

    let mut games = Vec::with_capacity(settings.games);
    let mut total_score = 0.0;
    let mut total_length = 0;
    let mut sampling = SamplingStats::default();
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
        for (ply, entry) in record.plies.iter().enumerate() {
            if let (Some(search), Some(sampled)) = (
                &entry.decision.training_policy,
                &entry.decision.diagnostics.sampling_policy,
            ) {
                let before = apply_temperature(search, self_play_temperature(ply));
                sampling.positions += 1;
                sampling.nonzero_before += before.iter().filter(|&&p| p > 0.0).count();
                let retained = sampled.iter().filter(|&&p| p > 0.0).count();
                sampling.nonzero_retained += retained;
                sampling.deterministic_positions += usize::from(retained == 1);
                sampling.removed_probability_mass += before
                    .iter()
                    .zip(sampled)
                    .filter(|(_, q)| **q == 0.0)
                    .map(|(p, _)| f64::from(*p))
                    .sum::<f64>();
                sampling.sampling_entropy_sum += sampled
                    .iter()
                    .filter(|&&p| p > 0.0)
                    .map(|&p| -f64::from(p) * f64::from(p).ln())
                    .sum::<f64>();
            }
        }
        total_score += record.value_for(Seat::First);
        total_length += record.plies.len();
        games.push(extract_training_game::<_, GomokuCodec>(record)?);
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
            sampling,
            games,
            total_score,
            total_length,
            batch_stats,
            duration: started.elapsed(),
        },
    ))
}

pub(super) fn train_epoch(
    network: &GomokuModel,
    optimizer: &mut Optimizer,
    data: &TrainingBatches<'_>,
    batch_size: usize,
    device: tch::Device,
    seed: u64,
    prefetch: usize,
) -> Result<TrainingStats> {
    let started = Instant::now();
    let mut total_value_loss = 0.0f64;
    let mut total_policy_loss = 0.0f64;
    let mut batches = 0;
    for batch in data.batches(batch_size, Some(seed), prefetch)? {
        let batch = batch?.to_device(device)?;
        let output = network.forward_t(&batch.states, true);
        let predicted_policy_log_probabilities = policy_log_probabilities(&output.policy_logits);
        let value_loss = output.values.mse_loss(&batch.values, Reduction::Mean);
        let policy_loss =
            -(&batch.policies * predicted_policy_log_probabilities).sum(None) / batch.len() as f64;
        let value_scalar = f32::try_from(&value_loss).context("reading value loss")? as f64;
        let policy_scalar = f32::try_from(&policy_loss).context("reading policy loss")? as f64;
        ensure!(
            value_scalar.is_finite() && policy_scalar.is_finite(),
            "non-finite loss in training batch {batches}; refusing optimizer update"
        );
        optimizer.backward_step(&(&value_loss + &policy_loss));

        let chunk_len = batch.len() as f64;
        total_value_loss += value_scalar * chunk_len;
        total_policy_loss += policy_scalar * chunk_len;
        batches += 1;
    }

    let samples = data.len();
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

fn file_sha256(path: &Path) -> Result<String> {
    use sha2::{Digest, Sha256};
    use std::io::Read;
    let mut file = fs::File::open(path)?;
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 65536];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

fn validate_replay_history(args: &TrainArgs) -> Result<()> {
    let Some(source) = &args.replay_history_dir else {
        return Ok(());
    };
    let epochs = args
        .replay_history_epochs
        .context("missing history epoch count")?;
    ensure!(epochs > 0, "history epoch count must be positive");
    let source = source.canonicalize()?;
    ensure!(
        args.stats_dir.canonicalize()? != source.join("stats").canonicalize()?,
        "history output stats must differ from source"
    );
    ensure!(
        args.model.checkpoint_dir.canonicalize()? != source.join("checkpoints").canonicalize()?,
        "history output must differ from source checkpoints"
    );
    let config = serde_json::to_value(args)?;
    let mut sources = Vec::new();
    for epoch in 0..epochs {
        let checkpoint = source.join("checkpoints").join(format!("{epoch:08}"));
        let stats_path = source.join("stats").join(format!("{epoch:08}.json"));
        let stats: serde_json::Value = serde_json::from_reader(fs::File::open(&stats_path)?)?;
        ensure!(
            stats["epoch"].as_u64() == Some(epoch as u64),
            "history epoch mismatch"
        );
        for key in [
            "seed",
            "training_batch_size",
            "learning_rate",
            "weight_decay",
            "bn_gamma_one",
            "replay_positions",
            "replay_position_growth",
            "replay_growth_start_epoch",
            "replay_lr_exponent",
            "replay_games",
            "adam_backend",
        ] {
            ensure!(
                stats["config"][key] == config[key],
                "history training setting {key} differs at epoch {epoch}"
            );
        }
        let rate = scheduled_learning_rate(args, epoch)?
            .unwrap_or(args.learning_rate.unwrap_or(DEFAULT_LEARNING_RATE));
        ensure!(
            (stats["scheduled_learning_rate"]
                .as_f64()
                .context("missing historical LR")?
                - rate)
                .abs()
                < 1e-12,
            "history learning rate differs at epoch {epoch}"
        );
        sources.push(serde_json::json!({"epoch":epoch, "metadata_sha256":file_sha256(&checkpoint.join("metadata.json"))?,
            "replay_sha256":file_sha256(&checkpoint.join("replay.bin.zst"))?, "stats_sha256":file_sha256(&stats_path)?}));
    }
    let manifest = serde_json::json!({"source":source,"epochs":epochs,"sources":sources});
    let path = args.stats_dir.join("replay-history.json");
    if path.exists() {
        let previous: serde_json::Value = serde_json::from_reader(fs::File::open(&path)?)?;
        ensure!(
            previous == manifest,
            "replay history changed since the previous invocation"
        );
    } else {
        fs::write(path, serde_json::to_vec_pretty(&manifest)?)?;
    }
    Ok(())
}

fn extend_replay_shuffled<T>(
    replay: &mut VecDeque<T>,
    mut epoch_items: Vec<T>,
    capacity: usize,
    seed: u64,
) {
    debug_assert!(capacity > 0);
    // Games arrive in completion order. Randomize each epoch so a FIFO capacity
    // cut retains a representative subset instead of the slowest finishers.
    epoch_items.shuffle(&mut SmallRng::seed_from_u64(seed));
    replay.extend(epoch_items);
    let overflow = replay.len().saturating_sub(capacity);
    replay.drain(..overflow);
}

fn extend_replay_positions_shuffled<T>(
    replay: &mut VecDeque<Vec<T>>,
    mut games: Vec<Vec<T>>,
    capacity: usize,
    seed: u64,
) {
    debug_assert!(capacity > 0);
    games.shuffle(&mut SmallRng::seed_from_u64(seed));
    replay.extend(games);
    let mut positions: usize = replay.iter().map(Vec::len).sum();
    // Keep trajectories intact, including one oversized game if necessary.
    while positions > capacity && replay.len() > 1 {
        positions -= replay.pop_front().expect("nonempty replay").len();
    }
}

fn replay_position_capacity(args: &TrainArgs, epoch: usize) -> Result<usize> {
    // Snapshot epochs are zero-based. The first 15 training cycles use the plateau.
    let completed_cycles = epoch.checked_add(1).context("epoch count overflow")?;
    let growth_cycles = completed_cycles.saturating_sub(args.replay_growth_start_epoch);
    args.replay_position_growth
        .checked_mul(growth_cycles)
        .and_then(|growth| args.replay_positions.checked_add(growth))
        .context("replay position capacity overflow")
}

fn render_sample_games(games_dir: &Path, epoch: usize, games: Vec<ReplayGame>) -> Result<()> {
    for (id, game) in games.into_iter().enumerate() {
        generate_game_image(&game).save(game_path(games_dir, epoch, id))?;
    }
    Ok(())
}

fn write_invocation_config(
    stats_dir: &Path,
    args: &TrainArgs,
    model_spec: &ModelSpec,
) -> Result<()> {
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

    #[derive(Serialize)]
    struct InvocationConfig<'a> {
        schema_version: u32,
        model: &'a ModelSpec,
        config: &'a TrainArgs,
        new_game_policy_target: &'static str,
        self_play_temperature_schedule: &'static str,
    }

    serde_json::to_writer_pretty(
        &mut file,
        &InvocationConfig {
            schema_version: METRICS_SCHEMA_VERSION,
            model: model_spec,
            config: args,
            new_game_policy_target: NEW_GAME_POLICY_TARGET,
            self_play_temperature_schedule: SELF_PLAY_TEMPERATURE_SCHEDULE,
        },
    )?;
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
    // Both seats receive the same temperature within each consecutive move pair.
    let progress = (turn / 2).saturating_sub(2).min(7) as f32 / 7.0;
    1.0 - 0.3 * progress
}

fn scheduled_learning_rate(args: &TrainArgs, epoch: usize) -> Result<Option<f64>> {
    let Some(exponent) = args.replay_lr_exponent else {
        return Ok(None);
    };
    let base = args
        .learning_rate
        .context("replay-linked LR requires an explicit initial learning-rate")?;
    let capacity = replay_position_capacity(args, epoch)?;
    let rate = base * (args.replay_positions as f64 / capacity as f64).powf(exponent);
    ensure!(
        rate.is_finite() && rate > 0.0,
        "scheduled learning rate must be finite and positive"
    );
    Ok(Some(rate))
}

pub(super) fn derive_seed(base: u64, stream: u64) -> u64 {
    let mut value = base ^ stream.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn validate_args(args: &TrainArgs) -> Result<()> {
    ensure!(
        args.top_p.is_finite() && args.top_p > 0.0 && args.top_p <= 1.0,
        "top-p must be in (0, 1]"
    );
    ensure!(
        args.replay_lr_exponent
            .is_none_or(|x| x.is_finite() && x >= 0.0),
        "replay-lr-exponent must be finite and non-negative"
    );
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
        args.replay_games.is_none_or(|capacity| capacity > 0),
        "replay-games must be greater than zero"
    );
    ensure!(
        args.replay_positions > 0,
        "replay-positions must be greater than zero"
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
    use std::collections::{BTreeMap, VecDeque};

    use super::{
        REPLAY_SHUFFLE_STREAM, derive_seed, extend_replay_positions_shuffled,
        extend_replay_shuffled, replay_position_capacity, self_play_temperature,
    };
    use crate::cli::{Cli, Command};
    use clap::Parser;

    async fn verify_history_reconstruction(device: &str) {
        use crate::cli::{Cli, Command};
        use clap::Parser;
        use std::collections::BTreeMap;
        let root = std::env::temp_dir().join(format!(
            "alz-history-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let make_args = |name: &str, epochs: usize, history: bool| {
            let cli = Cli::try_parse_from([
                "alz",
                "train",
                "--device",
                device,
                "--architecture",
                "kata-gelu-boardmask-value64x2-v1",
                "--learning-rate",
                "0.001",
                "--replay-lr-exponent",
                "1.1",
                "--bn-gamma-one",
                "--replay-cache",
                "cpu",
                "--prefetch-batches",
                "2",
            ])
            .unwrap();
            let Command::Train(mut args) = cli.command else {
                unreachable!()
            };
            args.model.checkpoint_dir = root.join(name).join("checkpoints");
            args.stats_dir = root.join(name).join("stats");
            args.games_dir = root.join(name).join("games");
            args.epochs = Some(epochs);
            args.games_per_epoch = 2;
            args.simulations = 2;
            args.games_parallelism = 2;
            args.inference_batch_size = 2;
            args.batch_timeout_us = 100;
            args.training_batch_size = 32;
            args.replay_positions = 64;
            args.replay_position_growth = 8;
            args.replay_growth_start_epoch = 1;
            args.rendered_games = 0;
            args.seed = 20260909;
            if history {
                args.replay_history_dir = Some(root.join("source"));
                args.replay_history_epochs = Some(2);
            }
            args
        };
        super::run(make_args("source", 2, false)).await.unwrap();
        super::run(make_args("reconstructed", 2, true))
            .await
            .unwrap();
        super::run(make_args("resumed", 1, true)).await.unwrap();
        super::run(make_args("resumed", 2, true)).await.unwrap();
        let tensors = |name: &str, epoch: usize| -> BTreeMap<String, tch::Tensor> {
            tch::Tensor::read_safetensors(
                root.join(name)
                    .join("checkpoints")
                    .join(format!("{epoch:08}"))
                    .join("model.safetensors"),
            )
            .unwrap()
            .into_iter()
            .collect()
        };
        for epoch in 0..2 {
            let expected = tensors("source", epoch);
            for name in ["reconstructed", "resumed"] {
                for (key, tensor) in tensors(name, epoch) {
                    let delta = (&tensor - &expected[&key]).abs().max().double_value(&[]);
                    assert!(
                        delta < 1e-6,
                        "{device} {name} epoch {epoch} tensor {key}: {delta}"
                    );
                }
            }
        }
        // The reconstructed optimizer and replay must also continue like the source.
        super::run(make_args("source", 3, false)).await.unwrap();
        super::run(make_args("reconstructed", 3, true))
            .await
            .unwrap();
        let expected = tensors("source", 2);
        for (key, tensor) in tensors("reconstructed", 2) {
            let delta = (&tensor - &expected[&key]).abs().max().double_value(&[]);
            assert!(delta < 1e-6, "continuation tensor {key}: {delta}");
        }
        std::fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    #[ignore = "integration test; run serially"]
    async fn replay_history_cpu_reconstructs_resumes_and_continues() {
        verify_history_reconstruction("cpu").await;
    }

    #[tokio::test]
    #[ignore = "requires deterministic CUDA; use scripts/check_replay_history_runpod.sh"]
    async fn replay_history_cuda_reconstructs_resumes_and_continues() {
        assert!(tch::Cuda::is_available());
        verify_history_reconstruction("cuda").await;
    }

    #[test]
    fn replay_linked_lr_is_absolute_and_uses_initial_rate() {
        let Command::Train(args) = Cli::try_parse_from([
            "alz",
            "train",
            "--learning-rate",
            "0.001",
            "--replay-lr-exponent",
            "1.1",
            "--replay-positions",
            "62500",
            "--replay-position-growth",
            "3750",
        ])
        .unwrap()
        .command
        else {
            panic!("train command")
        };
        assert_eq!(
            super::scheduled_learning_rate(&args, 0).unwrap(),
            Some(0.001)
        );
        assert_eq!(
            super::scheduled_learning_rate(&args, 14).unwrap(),
            Some(0.001)
        );
        let final_rate = super::scheduled_learning_rate(&args, 99).unwrap().unwrap();
        assert!((final_rate - 0.001 * (62500.0_f64 / 381250.0).powf(1.1)).abs() < 1e-12);
        assert!(super::scheduled_learning_rate(&args, 15).unwrap().unwrap() < 0.001);
        for bad in ["0", "-0.1", "1.1", "NaN"] {
            if let Ok(cli) = Cli::try_parse_from(["alz", "train", "--top-p", bad]) {
                let Command::Train(invalid) = cli.command else {
                    panic!("train command")
                };
                assert!(super::validate_args(&invalid).is_err());
            }
        }
    }

    #[test]
    fn position_capacity_schedule_uses_absolute_epoch_on_resume() {
        let Command::Train(mut args) = Cli::try_parse_from(["alz", "train"]).unwrap().command
        else {
            panic!("expected train command");
        };
        for (cycle, expected) in [
            (1, 37_500),
            (15, 37_500),
            (16, 39_750),
            (25, 60_000),
            (50, 116_250),
            (100, 228_750),
        ] {
            assert_eq!(
                replay_position_capacity(&args, cycle - 1).unwrap(),
                expected
            );
        }
        args.replay_position_growth = 0;
        assert_eq!(replay_position_capacity(&args, 1000).unwrap(), 37_500);
        args.replay_position_growth = usize::MAX;
        assert!(replay_position_capacity(&args, 16).is_err());
    }

    #[test]
    fn position_capacity_evicts_whole_old_games_by_size() {
        let mut replay = VecDeque::from([vec![0; 2], vec![1; 8]]);
        extend_replay_positions_shuffled(&mut replay, vec![vec![2; 3]], 11, 7);
        assert_eq!(replay, VecDeque::from([vec![1; 8], vec![2; 3]]));
        extend_replay_positions_shuffled(&mut replay, vec![vec![3; 4]], 11, 7);
        assert_eq!(replay, VecDeque::from([vec![2; 3], vec![3; 4]]));
        extend_replay_positions_shuffled(&mut replay, vec![vec![4; 20]], 11, 7);
        assert_eq!(replay, VecDeque::from([vec![4; 20]]));
    }

    #[test]
    fn temperature_is_paired_monotone_and_clamped() {
        for ply in 0..6 {
            assert_eq!(self_play_temperature(ply), 1.0);
        }
        for round in 0..200 {
            assert_eq!(
                self_play_temperature(round * 2),
                self_play_temperature(round * 2 + 1)
            );
            assert!(self_play_temperature(round * 2) >= self_play_temperature(round * 2 + 2));
        }
        assert!((self_play_temperature(6) - (1.0 - 0.3 / 7.0)).abs() < 1e-6);
        assert_eq!(self_play_temperature(18), 0.7);
        assert_eq!(self_play_temperature(19), 0.7);
        assert_eq!(self_play_temperature(usize::MAX), 0.7);
    }

    #[test]
    fn derived_seeds_are_stable_and_stream_specific() {
        assert_eq!(derive_seed(7, 11), derive_seed(7, 11));
        assert_ne!(derive_seed(7, 11), derive_seed(7, 12));
        assert_ne!(derive_seed(7, 11), derive_seed(8, 11));
    }

    #[test]
    fn replay_capacity_keeps_a_shuffled_subset_of_the_oldest_epoch() {
        fn build_replay() -> VecDeque<(usize, usize)> {
            let mut replay = VecDeque::new();
            for epoch in 0..3 {
                let games = (0..700)
                    .map(|completion_rank| (epoch, completion_rank))
                    .collect();
                extend_replay_shuffled(
                    &mut replay,
                    games,
                    1_800,
                    derive_seed(7, epoch as u64 ^ REPLAY_SHUFFLE_STREAM),
                );
            }
            replay
        }

        let replay = build_replay();
        assert_eq!(replay, build_replay());

        let mut ranks_by_epoch = BTreeMap::<_, Vec<_>>::new();
        for (epoch, completion_rank) in replay {
            ranks_by_epoch
                .entry(epoch)
                .or_default()
                .push(completion_rank);
        }
        assert_eq!(ranks_by_epoch[&0].len(), 400);
        assert_eq!(ranks_by_epoch[&1].len(), 700);
        assert_eq!(ranks_by_epoch[&2].len(), 700);
        assert!(ranks_by_epoch[&0].iter().any(|rank| *rank < 300));
        assert_ne!(ranks_by_epoch[&0], (300..700).collect::<Vec<_>>());
    }
}
