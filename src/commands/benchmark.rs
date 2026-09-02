use std::{fs, path::Path, time::Instant};

use alz::{
    engine::{
        AlphaZeroNet, NetworkBatchStats, masked_policy_probabilities, policy_log_probabilities,
    },
    gomoku::{GomokuModel, ModelSpec},
};
use anyhow::{Context, Result, ensure};
use serde::Serialize;
use tch::{
    Device, Kind, Reduction, Tensor,
    nn::{self, Optimizer, OptimizerConfig},
};

use crate::cli::{
    BenchmarkMode, InferenceBenchmarkArgs, SelfPlayBenchmarkArgs, TrainingBenchmarkArgs,
};

use super::{
    common::resolve_device,
    train::{SelfPlaySettings, collect_epoch_games},
};

const BENCHMARK_SCHEMA_VERSION: u32 = 5;

#[derive(Serialize)]
struct InferenceResult<'a> {
    schema_version: u32,
    benchmark: &'static str,
    config: &'a InferenceBenchmarkArgs,
    device: String,
    examples: usize,
    duration_seconds: f64,
    milliseconds_per_iteration: f64,
    examples_per_second: f64,
    checksum: f64,
}

#[derive(Serialize)]
struct TrainingResult<'a> {
    schema_version: u32,
    benchmark: &'static str,
    config: &'a TrainingBenchmarkArgs,
    device: String,
    examples: usize,
    duration_seconds: f64,
    milliseconds_per_iteration: f64,
    examples_per_second: f64,
    final_loss: f64,
}

#[derive(Serialize)]
struct SelfPlayResult<'a> {
    schema_version: u32,
    benchmark: &'static str,
    config: &'a SelfPlayBenchmarkArgs,
    device: String,
    duration_seconds: f64,
    games: usize,
    games_per_second: f64,
    total_game_length: usize,
    moves_per_second: f64,
    average_game_length: f64,
    total_score: f32,
    evaluations: u64,
    evaluations_per_second: f64,
    average_batch_size: f64,
    batch_fill_ratio: f64,
    average_queue_wait_us: f64,
    average_request_latency_us: f64,
    average_service_us: f64,
    average_position_encoding_us: f64,
    average_policy_mask_construction_us: f64,
    average_policy_mask_batch_construction_us: f64,
    average_policy_mask_transfer_submission_us: f64,
    average_policy_postprocess_submission_us: f64,
    average_policy_decode_us: f64,
    network: &'a NetworkBatchStats,
}

pub async fn run(mode: BenchmarkMode) -> Result<()> {
    match mode {
        BenchmarkMode::Inference(args) => run_inference(args),
        BenchmarkMode::Training(args) => run_training(args),
        BenchmarkMode::SelfPlay(args) => run_self_play(args).await,
    }
}

fn run_inference(args: InferenceBenchmarkArgs) -> Result<()> {
    ensure!(args.batch_size > 0, "batch-size must be greater than zero");
    ensure!(args.iterations > 0, "iterations must be greater than zero");
    tch::manual_seed((args.seed & i64::MAX as u64) as i64);
    let device = resolve_device(&args.device)?;
    let var_store = nn::VarStore::new(device);
    let model_spec = ModelSpec::from(args.architecture);
    let network = GomokuModel::new(var_store.root(), &model_spec);
    let input = Tensor::zeros(
        [args.batch_size as i64, 2, 19, 19],
        (Kind::Float, Device::Cpu),
    );
    let policy_mask = Tensor::ones([args.batch_size as i64, 19, 19], (Kind::Bool, Device::Cpu));

    let mut checksum = 0.0;
    for _ in 0..args.warmup_iterations {
        checksum += inference_iteration(&network, &input, &policy_mask, device)?;
    }
    let started = Instant::now();
    for _ in 0..args.iterations {
        checksum += inference_iteration(&network, &input, &policy_mask, device)?;
    }
    let duration = started.elapsed();
    let examples = args.batch_size * args.iterations;
    let result = InferenceResult {
        schema_version: BENCHMARK_SCHEMA_VERSION,
        benchmark: "inference",
        config: &args,
        device: format!("{device:?}"),
        examples,
        duration_seconds: duration.as_secs_f64(),
        milliseconds_per_iteration: duration.as_secs_f64() * 1_000.0 / args.iterations as f64,
        examples_per_second: examples as f64 / duration.as_secs_f64(),
        checksum,
    };
    emit_result(&result, args.output.as_deref())
}

fn inference_iteration(
    network: &GomokuModel,
    cpu_input: &Tensor,
    cpu_policy_mask: &Tensor,
    device: Device,
) -> Result<f64> {
    let input = cpu_input.to_device(device);
    let output = tch::no_grad(|| network.forward_t(&input, false));
    let policy_mask = if matches!(device, Device::Cuda(_)) {
        cpu_policy_mask.to_device_(device, Kind::Bool, true, false)
    } else {
        cpu_policy_mask.to_device(device)
    };
    let policies = masked_policy_probabilities(&output.policy_logits, &policy_mask);
    let values = output.values.to(Device::Cpu);
    let policies = policies.to(Device::Cpu);
    let checksum = f32::try_from(&values.sum(None)).context("reading inference values")?
        + f32::try_from(&policies.sum(None)).context("reading inference policies")?;
    Ok(checksum as f64)
}

fn run_training(args: TrainingBenchmarkArgs) -> Result<()> {
    ensure!(args.batch_size > 0, "batch-size must be greater than zero");
    ensure!(args.iterations > 0, "iterations must be greater than zero");
    tch::manual_seed((args.seed & i64::MAX as u64) as i64);
    let device = resolve_device(&args.device)?;
    let var_store = nn::VarStore::new(device);
    let model_spec = ModelSpec::from(args.architecture);
    let network = GomokuModel::new(var_store.root(), &model_spec);
    let mut optimizer = nn::Adam::default().build(&var_store, 1e-3)?;
    let states = Tensor::zeros(
        [args.batch_size as i64, 2, 19, 19],
        (Kind::Float, Device::Cpu),
    );
    let policies = Tensor::ones([args.batch_size as i64, 19, 19], (Kind::Float, Device::Cpu))
        / (19 * 19) as f64;
    let values = Tensor::zeros([args.batch_size as i64], (Kind::Float, Device::Cpu));

    let mut final_loss = 0.0;
    for _ in 0..args.warmup_iterations {
        final_loss = training_iteration(
            &network,
            &mut optimizer,
            &states,
            &policies,
            &values,
            device,
        )?;
    }
    let started = Instant::now();
    for _ in 0..args.iterations {
        final_loss = training_iteration(
            &network,
            &mut optimizer,
            &states,
            &policies,
            &values,
            device,
        )?;
    }
    let duration = started.elapsed();
    let examples = args.batch_size * args.iterations;
    let result = TrainingResult {
        schema_version: BENCHMARK_SCHEMA_VERSION,
        benchmark: "training",
        config: &args,
        device: format!("{device:?}"),
        examples,
        duration_seconds: duration.as_secs_f64(),
        milliseconds_per_iteration: duration.as_secs_f64() * 1_000.0 / args.iterations as f64,
        examples_per_second: examples as f64 / duration.as_secs_f64(),
        final_loss,
    };
    emit_result(&result, args.output.as_deref())
}

fn training_iteration(
    network: &GomokuModel,
    optimizer: &mut Optimizer,
    cpu_states: &Tensor,
    cpu_policies: &Tensor,
    cpu_values: &Tensor,
    device: Device,
) -> Result<f64> {
    let states = cpu_states.to_device(device);
    let policies = cpu_policies.to_device(device);
    let values = cpu_values.to_device(device);
    let output = network.forward_t(&states, true);
    let predicted_policy_log_probabilities = policy_log_probabilities(&output.policy_logits);
    let value_loss = output.values.mse_loss(&values, Reduction::Mean);
    let policy_loss =
        -(policies * predicted_policy_log_probabilities).sum(None) / cpu_states.size()[0] as f64;
    let loss = &value_loss + &policy_loss;
    optimizer.backward_step(&loss);
    Ok(f32::try_from(&loss).context("reading training loss")? as f64)
}

async fn run_self_play(args: SelfPlayBenchmarkArgs) -> Result<()> {
    ensure!(args.games > 0, "games must be greater than zero");
    ensure!(
        args.simulations > 0,
        "simulations must be greater than zero"
    );
    ensure!(
        args.c_puct.is_finite() && args.c_puct >= 0.0,
        "c-puct must be finite and non-negative"
    );
    ensure!(
        args.inference_batch_size > 0,
        "inference-batch-size must be greater than zero"
    );
    ensure!(
        args.games_parallelism > 0,
        "games-parallelism must be greater than zero"
    );
    tch::manual_seed((args.seed & i64::MAX as u64) as i64);
    let device = resolve_device(&args.device)?;
    let var_store = nn::VarStore::new(device);
    let model_spec = ModelSpec::from(args.architecture);
    let network = GomokuModel::new(var_store.root(), &model_spec);

    if args.warmup_batches > 0 {
        let input = Tensor::zeros(
            [args.inference_batch_size as i64, 2, 19, 19],
            (Kind::Float, Device::Cpu),
        );
        let policy_mask = Tensor::ones(
            [args.inference_batch_size as i64, 19, 19],
            (Kind::Bool, Device::Cpu),
        );
        for _ in 0..args.warmup_batches {
            let _ = inference_iteration(&network, &input, &policy_mask, device)?;
        }
    }

    let settings = SelfPlaySettings {
        games: args.games,
        simulations: args.simulations,
        c_puct: args.c_puct,
        inference_batch_size: args.inference_batch_size,
        inference_symmetry: args.inference_symmetry,
        games_parallelism: args.games_parallelism,
        batch_timeout: std::time::Duration::from_micros(args.batch_timeout_us),
        seed: args.seed,
        progress_every_games: 0,
        heartbeat_interval: std::time::Duration::ZERO,
    };
    let (_network, games) = collect_epoch_games(network, settings, device).await?;
    let duration_seconds = games.duration.as_secs_f64();
    let game_count = games.games.len();
    let average_batch_size = games.batch_stats.average_batch_size();
    let result = SelfPlayResult {
        schema_version: BENCHMARK_SCHEMA_VERSION,
        benchmark: "self_play",
        config: &args,
        device: format!("{device:?}"),
        duration_seconds,
        games: game_count,
        games_per_second: game_count as f64 / duration_seconds,
        total_game_length: games.total_length,
        moves_per_second: games.total_length as f64 / duration_seconds,
        average_game_length: games.total_length as f64 / game_count as f64,
        total_score: games.total_score,
        evaluations: games.batch_stats.requests,
        evaluations_per_second: games.batch_stats.requests as f64 / duration_seconds,
        average_batch_size,
        batch_fill_ratio: average_batch_size / args.inference_batch_size as f64,
        average_queue_wait_us: games.batch_stats.average_queue_wait_us(),
        average_request_latency_us: games.batch_stats.average_request_latency_us(),
        average_service_us: games.batch_stats.average_service_us(),
        average_position_encoding_us: games.batch_stats.average_position_encoding_us(),
        average_policy_mask_construction_us: games
            .batch_stats
            .average_policy_mask_construction_us(),
        average_policy_mask_batch_construction_us: games
            .batch_stats
            .average_policy_mask_batch_construction_us(),
        average_policy_mask_transfer_submission_us: games
            .batch_stats
            .average_policy_mask_transfer_submission_us(),
        average_policy_postprocess_submission_us: games
            .batch_stats
            .average_policy_postprocess_submission_us(),
        average_policy_decode_us: games.batch_stats.average_policy_decode_us(),
        network: &games.batch_stats,
    };
    emit_result(&result, args.output.as_deref())
}

fn emit_result<T: Serialize>(result: &T, output: Option<&Path>) -> Result<()> {
    if let Some(path) = output {
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            fs::create_dir_all(parent)
                .with_context(|| format!("creating benchmark output {}", parent.display()))?;
        }
        let mut file = fs::File::create(path)
            .with_context(|| format!("creating benchmark output {}", path.display()))?;
        serde_json::to_writer_pretty(&mut file, result)?;
        use std::io::Write as _;
        writeln!(file)?;
    }
    tracing::info!(result = %serde_json::to_string(result)?, "BENCHMARK_RESULT");
    Ok(())
}
