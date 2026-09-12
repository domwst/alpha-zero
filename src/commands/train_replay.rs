use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::{BufReader, Read, Write},
    path::{Path, PathBuf},
};

use alz::{
    engine::{AlphaZeroNet, TrainingCodec, policy_log_probabilities},
    gomoku::{GomokuCodec, GomokuModel, ModelSpec},
    training_batches::TrainingBatches,
    training_snapshot::{
        ReplayBuffer, ReplayGame, SnapshotDescriptor, find_latest_snapshot, load_replay_checkpoint,
        load_training_snapshot, save_training_snapshot,
    },
};
use anyhow::{Context, Result, ensure};
use rand::{SeedableRng, rngs::SmallRng, seq::SliceRandom};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tch::{Reduction, nn};

use crate::cli::{AdamBackendChoice, ReplayLrSchedule, TrainReplayArgs};

use super::{
    common::resolve_device,
    train::{derive_seed, train_epoch},
};

const SPLIT_STREAM: u64 = 0x0053_504c_4954;
const TRAIN_STREAM: u64 = 0x0054_5241_494e;

fn epoch_learning_rate(config: &RunConfig, epoch: usize) -> f64 {
    match config.lr_schedule {
        ReplayLrSchedule::Constant => config.learning_rate,
        ReplayLrSchedule::Cosine => {
            let last = config.schedule_epochs.expect("validated schedule horizon") - 1;
            let end = config.final_learning_rate.expect("validated final rate");
            let progress = epoch.min(last) as f64 / last as f64;
            end + (config.learning_rate - end) * (1.0 + (std::f64::consts::PI * progress).cos())
                / 2.0
        }
    }
}

fn bn_gamma_names(variables: &std::collections::HashMap<String, tch::Tensor>) -> Vec<String> {
    let mut names: Vec<_> = variables
        .keys()
        .filter_map(|name| {
            name.strip_suffix(".running_var")
                .map(|prefix| format!("{prefix}.weight"))
        })
        .collect();
    names.sort();
    names
}

pub(super) fn initialize_bn_gamma_one(store: &nn::VarStore) -> Result<()> {
    let variables = store.variables();
    let names = bn_gamma_names(&variables);
    ensure!(!names.is_empty(), "model has no BatchNorm affine weights");
    tch::no_grad(|| -> Result<()> {
        for name in names {
            let mut tensor = variables
                .get(&name)
                .context("BatchNorm gamma missing")?
                .shallow_clone();
            let _ = tensor.f_fill_(1.0)?;
        }
        Ok(())
    })
}

fn tensor_fingerprints(store: &nn::VarStore) -> Result<serde_json::Value> {
    let variables = store.variables();
    let gamma = bn_gamma_names(&variables);
    let mut hashes = BTreeMap::new();
    for (name, tensor) in &variables {
        let flat = tensor
            .to_device(tch::Device::Cpu)
            .to_kind(tch::Kind::Float)
            .view([-1]);
        let values: Vec<f32> = Vec::try_from(&flat)?;
        let mut digest = Sha256::new();
        for value in values {
            digest.update(value.to_le_bytes());
        }
        hashes.insert(name.clone(), format!("{:x}", digest.finalize()));
    }
    Ok(serde_json::json!({"sha256": hashes, "bn_gamma_names": gamma}))
}

/// Device, source path, and target epoch count may change when moving/resuming a run.
#[derive(Debug, Deserialize, PartialEq, Serialize)]
struct RunConfig {
    #[serde(default)]
    lr_schedule: ReplayLrSchedule,
    #[serde(default)]
    final_learning_rate: Option<f64>,
    #[serde(default)]
    schedule_epochs: Option<usize>,
    #[serde(default)]
    bn_gamma_one: bool,
    #[serde(default)]
    split_seed: Option<u64>,
    #[serde(default)]
    adam_backend: AdamBackendChoice,
    schema_version: u32,
    model: ModelSpec,
    dataset_sha256: String,
    seed: u64,
    batch_size: usize,
    learning_rate: f64,
    weight_decay: f64,
    validation_fraction: f64,
}

#[derive(Debug, Serialize)]
struct ValidationStats {
    samples: usize,
    value_loss: f64,
    policy_loss: f64,
    total_loss: f64,
}

pub fn run(args: TrainReplayArgs) -> Result<()> {
    args.performance.validate()?;
    ensure!(args.epochs > 0, "epochs must be greater than zero");
    ensure!(
        args.training_batch_size > 0,
        "training-batch-size must be greater than zero"
    );
    ensure!(
        args.learning_rate.is_finite() && args.learning_rate > 0.0,
        "invalid learning rate"
    );
    match args.lr_schedule {
        ReplayLrSchedule::Constant => ensure!(
            args.final_learning_rate.is_none(),
            "final-learning-rate requires cosine scheduling"
        ),
        ReplayLrSchedule::Cosine => ensure!(
            args.epochs >= 2
                && args.final_learning_rate.is_some_and(|rate| rate.is_finite()
                    && rate > 0.0
                    && rate <= args.learning_rate),
            "cosine needs at least two passes and a positive final rate no larger than the initial rate"
        ),
    }
    ensure!(
        args.weight_decay.is_finite() && args.weight_decay >= 0.0,
        "invalid weight decay"
    );
    ensure!(
        args.validation_fraction.is_finite()
            && args.validation_fraction > 0.0
            && args.validation_fraction < 1.0,
        "validation-fraction must be between zero and one"
    );

    let (sources, replay, dataset_sha256, duplicate_games) =
        load_dataset(&args.replay_checkpoint_dir)?;
    let source_games = replay.len();
    let (training, validation) = split_replay(
        replay,
        args.validation_fraction,
        args.split_seed.unwrap_or(args.seed),
    )?;
    let training_positions = training.iter().map(Vec::len).sum::<usize>();
    let validation_positions = validation.iter().map(Vec::len).sum::<usize>();
    tracing::info!(
        source_games,
        duplicate_games,
        source_snapshots = sources.len(),
        training_games = training.len(),
        validation_games = validation.len(),
        training_positions,
        validation_positions,
        dataset_sha256,
        "validated fixed replay dataset"
    );

    if args.inspect_only {
        return Ok(());
    }

    let config = RunConfig {
        lr_schedule: args.lr_schedule,
        final_learning_rate: args.final_learning_rate,
        schedule_epochs: (args.lr_schedule == ReplayLrSchedule::Cosine).then_some(args.epochs),
        bn_gamma_one: args.bn_gamma_one,
        split_seed: args.split_seed,
        adam_backend: args.performance.adam_backend,
        schema_version: 1,
        model: args.architecture.into(),
        dataset_sha256,
        seed: args.seed,
        batch_size: args.training_batch_size,
        learning_rate: args.learning_rate,
        weight_decay: args.weight_decay,
        validation_fraction: args.validation_fraction,
    };
    let config_path = args.run_dir.join("replay-config.json");
    let checkpoint_dir = args.run_dir.join("checkpoints");
    if config_path.exists() {
        let previous: RunConfig =
            serde_json::from_reader(BufReader::new(File::open(&config_path)?))?;
        ensure!(
            previous == config,
            "replay run configuration changed; use a new run-dir"
        );
    } else {
        ensure!(
            !args.run_dir.exists() || fs::read_dir(&args.run_dir)?.next().is_none(),
            "new replay run-dir must be empty"
        );
        fs::create_dir_all(&args.run_dir)?;
        write_json(&config_path, &config)?;
        write_json(
            &args.run_dir.join("dataset.json"),
            &serde_json::json!({
                "sources": sources,
                "duplicate_games_removed": duplicate_games,
                "dataset_sha256": config.dataset_sha256,
                "training_games": training.len(), "training_positions": training_positions,
                "validation_games": validation.len(), "validation_positions": validation_positions,
                "augmentation_count": GomokuCodec::augmentation_count(),
                "split": "exact game deduplication, then seeded shuffle of trajectory groups; validation excluded from training"
            }),
        )?;
    }
    fs::create_dir_all(args.run_dir.join("epochs"))?;

    let device = resolve_device(&args.device)?;
    tch::manual_seed((args.seed & i64::MAX as u64) as i64);
    let mut var_store = nn::VarStore::new(device);
    let network = GomokuModel::new(var_store.root(), &config.model);
    if args.bn_gamma_one {
        initialize_bn_gamma_one(&var_store)?;
    }
    // This small manifest verifies paired initial tensors without retaining another checkpoint.
    // On resume, retain the original receipt; snapshot loading below restores learned parameters.
    let initial_manifest = args.run_dir.join("initial-tensors.json");
    if !initial_manifest.exists() {
        write_json(&initial_manifest, &tensor_fingerprints(&var_store)?)?;
    }
    let mut optimizer = super::common::build_adam(
        args.performance.adam_backend,
        &var_store,
        args.learning_rate,
        args.weight_decay,
    )?;
    let cache_started = std::time::Instant::now();
    let training_data = TrainingBatches::new(&training, args.performance.replay_cache, device)?;
    let validation_data = TrainingBatches::new(&validation, args.performance.replay_cache, device)?;
    write_json(
        &args.run_dir.join("batch-cache.json"),
        &serde_json::json!({
            "mode": args.performance.replay_cache, "device": format!("{device:?}"),
            "prefetch_batches": args.performance.prefetch_batches,
            "initialization_seconds": cache_started.elapsed().as_secs_f64(),
            "encoded_bytes": if args.performance.replay_cache == alz::training_batches::ReplayCacheMode::None { 0 } else { training_data.cache_bytes()? + validation_data.cache_bytes()? }
        }),
    )?;
    let mut start_epoch = 0;
    if let Some(snapshot) = find_latest_snapshot(&checkpoint_dir)? {
        ensure!(
            snapshot.model_spec() == &config.model,
            "checkpoint architecture differs from run config"
        );
        let restored_replay = load_training_snapshot(&snapshot, &mut var_store, &mut optimizer)?;
        ensure!(
            restored_replay == training,
            "checkpoint replay differs from the fixed training split"
        );
        start_epoch = snapshot.epoch() + 1;
        tracing::info!(start_epoch, "restored replay training model and optimizer");
    } else {
        let initial = validate(
            &network,
            &validation_data,
            args.training_batch_size,
            device,
            args.performance.prefetch_batches,
        )?;
        write_json(&args.run_dir.join("initial-validation.json"), &initial)?;
    }

    ensure!(
        start_epoch <= args.epochs,
        "run already exceeds requested epoch budget; use a new run-dir"
    );
    if args.initialize_only {
        ensure!(
            start_epoch == 0,
            "initialize-only requires an untrained run"
        );
        return Ok(());
    }
    for epoch in start_epoch..args.epochs {
        let learning_rate = epoch_learning_rate(&config, epoch);
        optimizer.set_lr(learning_rate);
        tracing::info!(
            epoch,
            learning_rate,
            epochs = args.epochs,
            "starting fixed replay training pass"
        );
        let training_stats = train_epoch(
            &network,
            &mut optimizer,
            &training_data,
            args.training_batch_size,
            device,
            derive_seed(args.seed, epoch as u64 ^ TRAIN_STREAM),
            args.performance.prefetch_batches,
        )?;
        let validation_stats = validate(
            &network,
            &validation_data,
            args.training_batch_size,
            device,
            args.performance.prefetch_batches,
        )?;
        tracing::info!(
            epoch,
            validation_value_loss = validation_stats.value_loss,
            validation_policy_loss = validation_stats.policy_loss,
            "completed replay training pass"
        );
        // Write metrics before publishing the checkpoint, so a completed snapshot always has metrics.
        write_json(
            &args.run_dir.join("epochs").join(format!("{epoch:08}.json")),
            &serde_json::json!({"epoch": epoch, "learning_rate": learning_rate, "training": training_stats, "validation": validation_stats}),
        )?;
        save_training_snapshot(
            &checkpoint_dir,
            epoch,
            &config.model,
            &var_store,
            &optimizer,
            &training,
        )?;
    }
    let latest =
        find_latest_snapshot(&checkpoint_dir)?.context("no completed replay checkpoint")?;
    write_json(
        &args.run_dir.join("result.json"),
        &serde_json::json!({
            "schema_version": 1, "model": config.model, "dataset_sha256": config.dataset_sha256,
            "completed_epochs": latest.epoch() + 1, "checkpoint": latest.descriptor(),
            "selection": "final checkpoint after the requested training budget; no match-based selection"
        }),
    )?;
    Ok(())
}

#[derive(Serialize)]
struct ReplaySource {
    checkpoint: SnapshotDescriptor,
    replay_sha256: String,
    games: usize,
    positions: usize,
}

fn load_dataset(paths: &[PathBuf]) -> Result<(Vec<ReplaySource>, ReplayBuffer, String, usize)> {
    let mut unique = BTreeMap::<[u8; 32], ReplayGame>::new();
    let mut sources = Vec::new();
    let mut duplicates = 0;
    for path in paths {
        let replay_sha256 = file_sha256(&path.join("replay.bin.zst"))?;
        let (checkpoint, replay) = load_replay_checkpoint(path)?;
        sources.push(ReplaySource {
            checkpoint,
            replay_sha256,
            games: replay.len(),
            positions: replay.iter().map(Vec::len).sum(),
        });
        for game in replay {
            ensure!(!game.is_empty(), "source contains an empty replay game");
            let hash: [u8; 32] = Sha256::digest(bincode::serialize(&game)?).into();
            match unique.entry(hash) {
                std::collections::btree_map::Entry::Vacant(entry) => {
                    entry.insert(game);
                }
                std::collections::btree_map::Entry::Occupied(_) => {
                    duplicates += 1;
                }
            }
        }
        tracing::info!(path = %path.display(), unique_games = unique.len(), duplicates,
            "loaded replay source");
    }
    // Sort by content, so source order, duplicate buffers, and absolute paths cannot change the split.
    let mut hash = Sha256::new();
    hash.update(b"alz-fixed-replay-dataset-v1");
    for key in unique.keys() {
        hash.update(key);
    }
    let dataset_sha256 = format!("{:x}", hash.finalize());
    Ok((
        sources,
        unique.into_values().collect(),
        dataset_sha256,
        duplicates,
    ))
}

fn trajectory_hash(game: &ReplayGame) -> Result<[u8; 32]> {
    // Ignore policy/value targets: copies of one trajectory with different targets stay together.
    let states = game.iter().map(|sample| &sample.state).collect::<Vec<_>>();
    Ok(Sha256::digest(bincode::serialize(&states)?).into())
}

fn split_replay(
    replay: ReplayBuffer,
    fraction: f64,
    seed: u64,
) -> Result<(ReplayBuffer, ReplayBuffer)> {
    let mut groups = BTreeMap::<[u8; 32], Vec<ReplayGame>>::new();
    for game in replay {
        groups
            .entry(trajectory_hash(&game)?)
            .or_default()
            .push(game);
    }
    ensure!(
        groups.len() >= 2,
        "at least two distinct game trajectories are required"
    );
    let mut groups = groups.into_values().collect::<Vec<_>>();
    groups.shuffle(&mut SmallRng::seed_from_u64(derive_seed(
        seed,
        SPLIT_STREAM,
    )));
    let count = ((groups.len() as f64 * fraction).round() as usize).clamp(1, groups.len() - 1);
    let training = groups.split_off(count).into_iter().flatten().collect();
    Ok((training, groups.into_iter().flatten().collect()))
}

fn validate(
    network: &GomokuModel,
    data: &TrainingBatches<'_>,
    batch_size: usize,
    device: tch::Device,
    prefetch: usize,
) -> Result<ValidationStats> {
    let _guard = tch::no_grad_guard();
    // Evaluate all D4 orientations just as in training, but use inference-mode BatchNorm.
    let mut value_sum = 0.0;
    let mut policy_sum = 0.0;
    for batch in data.batches(batch_size, None, prefetch)? {
        let batch = batch?.to_device(device)?;
        let output = network.forward_t(&batch.states, false);
        let value_loss = f64::try_from(output.values.mse_loss(&batch.values, Reduction::Sum))?;
        let policy_loss = f64::try_from(
            -(&batch.policies * policy_log_probabilities(&output.policy_logits)).sum(None),
        )?;
        ensure!(
            value_loss.is_finite() && policy_loss.is_finite(),
            "non-finite validation loss"
        );
        value_sum += value_loss;
        policy_sum += policy_loss;
    }
    let count = data.len();
    ensure!(count > 0, "validation split has no positions");
    Ok(ValidationStats {
        samples: count,
        value_loss: value_sum / count as f64,
        policy_loss: policy_sum / count as f64,
        total_loss: (value_sum + policy_sum) / count as f64,
    })
}

fn file_sha256(path: &Path) -> Result<String> {
    let mut reader =
        BufReader::new(File::open(path).with_context(|| format!("opening {}", path.display()))?);
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let size = reader.read(&mut buffer)?;
        if size == 0 {
            break;
        }
        hash.update(&buffer[..size]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

fn write_json(path: &Path, value: &impl Serialize) -> Result<()> {
    let pending = path.with_extension("json.tmp");
    let mut writer = File::create(&pending)?;
    serde_json::to_writer_pretty(&mut writer, value)?;
    writeln!(writer)?;
    writer.sync_all()?;
    fs::rename(pending, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use alz::{
        engine::TrainingSample,
        gomoku::{BoardState, GomokuMove, GomokuPolicy},
    };
    use tch::nn::OptimizerConfig;

    #[test]
    fn old_replay_configs_select_standard_adam() {
        let old = serde_json::json!({
            "schema_version": 1, "model": {"architecture": "kata_v1"},
            "dataset_sha256": "test", "seed": 7, "batch_size": 256,
            "learning_rate": 0.001, "weight_decay": 0.0001, "validation_fraction": 0.1
        });
        let standard: RunConfig = serde_json::from_value(old.clone()).unwrap();
        assert_eq!(standard.adam_backend, AdamBackendChoice::Standard);
        assert_eq!(standard.lr_schedule, ReplayLrSchedule::Constant);
        assert!(!standard.bn_gamma_one);
        assert_eq!(epoch_learning_rate(&standard, 19), 0.001);
        let mut fused = old;
        fused["adam_backend"] = serde_json::json!("fused");
        assert_ne!(
            standard,
            serde_json::from_value::<RunConfig>(fused).unwrap()
        );
    }

    #[test]
    fn cosine_schedule_has_fixed_endpoints_and_resume_horizon() {
        let mut config: RunConfig = serde_json::from_value(serde_json::json!({
            "schema_version": 1, "model": {"architecture": "kata_v1"},
            "dataset_sha256": "test", "seed": 7, "batch_size": 256,
            "learning_rate": 0.001, "weight_decay": 0.0001, "validation_fraction": 0.1,
            "lr_schedule": "cosine", "final_learning_rate": 0.0001, "schedule_epochs": 20
        }))
        .unwrap();
        assert!((epoch_learning_rate(&config, 0) - 0.001).abs() < 1e-15);
        assert!((epoch_learning_rate(&config, 19) - 0.0001).abs() < 1e-15);
        for epoch in 1..20 {
            assert!(epoch_learning_rate(&config, epoch) < epoch_learning_rate(&config, epoch - 1));
        }
        let restored: RunConfig =
            serde_json::from_slice(&serde_json::to_vec(&config).unwrap()).unwrap();
        assert_eq!(
            epoch_learning_rate(&config, 12),
            epoch_learning_rate(&restored, 12)
        );
        config.schedule_epochs = Some(21);
        assert_ne!(config, restored);
    }

    #[test]
    fn gamma_one_changes_only_batch_norm_scales() {
        let store = nn::VarStore::new(tch::Device::Cpu);
        let spec: ModelSpec =
            serde_json::from_value(serde_json::json!({"architecture": "kata_gelu_value64x2_v1"}))
                .unwrap();
        let _model = GomokuModel::new(store.root(), &spec);
        let before = tensor_fingerprints(&store).unwrap();
        initialize_bn_gamma_one(&store).unwrap();
        let after = tensor_fingerprints(&store).unwrap();
        let variables = store.variables();
        let gamma = bn_gamma_names(&variables);
        assert!(!gamma.is_empty());
        for (name, tensor) in &variables {
            if gamma.contains(name) {
                assert_eq!(tensor.eq(1.0).all().int64_value(&[]), 1);
                assert_ne!(before["sha256"][name], after["sha256"][name]);
            } else {
                assert_eq!(before["sha256"][name], after["sha256"][name], "{name}");
            }
        }
    }

    #[test]
    fn split_is_reproducible_and_keeps_whole_games_separate() {
        let replay: ReplayBuffer = (0..10)
            .map(|i| {
                vec![
                    TrainingSample {
                        state: BoardState::new(),
                        policy: GomokuPolicy::one_hot(GomokuMove::from_xy(0, i)),
                        value: 1.0,
                    };
                    i + 1
                ]
            })
            .collect();
        let (train, validation) = split_replay(replay.clone(), 0.2, 7).unwrap();
        assert_eq!(train.len(), 8);
        assert_eq!(validation.len(), 2);
        assert_eq!(
            train.iter().chain(&validation).map(Vec::len).sum::<usize>(),
            55
        );
        assert!(train.iter().all(|game| !validation.contains(game)));
        assert!(
            (train.clone(), validation.clone()) == split_replay(replay.clone(), 0.2, 7).unwrap()
        );
        assert!((train, validation) != split_replay(replay, 0.2, 8).unwrap());
        assert!(split_replay(ReplayBuffer::new(), 0.2, 7).is_err());
    }

    #[test]
    fn target_variants_of_the_same_trajectory_stay_in_one_split() {
        let sample = TrainingSample {
            state: BoardState::new(),
            policy: GomokuPolicy::one_hot(GomokuMove::from_xy(0, 0)),
            value: 1.0,
        };
        let mut variant = sample.clone();
        variant.value = -1.0;
        variant.policy = GomokuPolicy::one_hot(GomokuMove::from_xy(1, 1));
        let replay = ReplayBuffer::from([vec![sample.clone()], vec![variant], vec![sample; 2]]);
        for seed in 0..8 {
            let (train, validation) = split_replay(replay.clone(), 0.5, seed).unwrap();
            assert!(train.iter().all(|a| {
                validation
                    .iter()
                    .all(|b| trajectory_hash(a).unwrap() != trajectory_hash(b).unwrap())
            }));
            assert_eq!(train.len() + validation.len(), 3);
        }
    }

    #[test]
    fn overlapping_sources_deduplicate_and_do_not_require_source_model_files() {
        let root = std::env::temp_dir().join(format!(
            "alz-replay-pool-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let vs = nn::VarStore::new(tch::Device::Cpu);
        let _model = GomokuModel::new(vs.root(), &ModelSpec::KataV1);
        let optimizer = nn::Adam::default().build(&vs, 1e-3).unwrap();
        let games = (0..3)
            .map(|i| {
                vec![TrainingSample {
                    state: BoardState::new(),
                    policy: GomokuPolicy::one_hot(GomokuMove::from_xy(0, i)),
                    value: 1.0,
                }]
            })
            .collect::<Vec<_>>();
        for epoch in 0..2 {
            save_training_snapshot(
                &root,
                epoch,
                &ModelSpec::KataV1,
                &vs,
                &optimizer,
                &games[epoch..epoch + 2].to_vec().into(),
            )
            .unwrap();
            fs::remove_file(root.join(format!("{epoch:08}/model.safetensors"))).unwrap();
            fs::remove_file(root.join(format!("{epoch:08}/optimizer.ot"))).unwrap();
        }
        let first = root.join("00000000");
        let second = root.join("00000001");
        let (_, pooled, digest, duplicates) =
            load_dataset(&[first.clone(), second.clone()]).unwrap();
        assert_eq!(pooled.len(), 3);
        assert_eq!(duplicates, 1);
        let (_, reversed, reversed_digest, duplicates) =
            load_dataset(&[second, first.clone(), first.clone()]).unwrap();
        assert!(pooled == reversed);
        assert_eq!(digest, reversed_digest);
        assert_eq!(duplicates, 3);
        fs::write(first.join("replay.bin.zst"), b"corrupted").unwrap();
        assert!(load_dataset(&[first]).is_err());
        fs::remove_dir_all(root).unwrap();
    }
}
