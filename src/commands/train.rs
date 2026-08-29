use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
    time::Duration,
};

use alz::{
    alpha_zero::{
        extract_training_game, generate_self_played_game, AlphaZeroNet, ExecutorScope,
        NetworkPositionEvaluator, PositionCodec, Seat, TrainingCodec,
    },
    tictactoe::{generate_game_image, BoardState, TicTacToeCodec, TicTacToeResNet},
    util::Timer,
};
use anyhow::{ensure, Context, Result};
use rand::seq::{IteratorRandom, SliceRandom};
use tch::{
    nn::{self, Optimizer, OptimizerConfig},
    Kind, Reduction, Tensor,
};

use crate::{
    cli::TrainArgs,
    training_snapshot::{
        find_latest_snapshot, load_training_snapshot, save_training_snapshot, ReplayBuffer,
        ReplayGame,
    },
};

use super::common::resolve_device;

const DEFAULT_LEARNING_RATE: f64 = 1e-3;
const DEFAULT_WEIGHT_DECAY: f64 = 1e-4;

struct EpochGames {
    games: Vec<ReplayGame>,
    total_score: f32,
    total_length: usize,
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
        println!("Restoring complete training snapshot {epoch}");
        replay = load_training_snapshot(&snapshot, &mut var_store, &mut optimizer)?;
        if let Some(learning_rate) = args.learning_rate {
            optimizer.set_lr(learning_rate);
        }
        if let Some(weight_decay) = args.weight_decay {
            optimizer.set_weight_decay(weight_decay);
        }
        start_epoch = epoch + 1;
    }

    let epochs = args.epochs.unwrap_or(usize::MAX);
    for epoch in (start_epoch..).take(epochs) {
        let timer = Timer::new();
        let (returned_network, epoch_games) =
            collect_epoch_games(network, &args, var_store.device()).await?;
        network = returned_network;

        let games_in_epoch = epoch_games.games.len();
        let total_score = epoch_games.total_score;
        let total_length = epoch_games.total_length;
        let avg_score = total_score / games_in_epoch as f32;
        let avg_length = total_length as f32 / games_in_epoch as f32;
        println!("Average score is {avg_score}");
        println!("Average length is {avg_length}");

        let sample_games = epoch_games
            .games
            .iter()
            .sample(
                &mut rand::rng(),
                args.rendered_games.min(epoch_games.games.len()),
            )
            .into_iter()
            .cloned()
            .collect::<Vec<_>>();

        replay.extend(epoch_games.games);
        while replay.len() > args.replay_games {
            replay.pop_front();
        }

        let (value_loss, policy_loss) = train_epoch(
            &network,
            &mut optimizer,
            &replay,
            args.training_batch_size,
            var_store.device(),
        );
        println!("Total value and policy loss: ({value_loss}, {policy_loss})");

        save_training_snapshot(
            &args.model.checkpoint_dir,
            epoch,
            &var_store,
            &optimizer,
            &replay,
        )?;
        render_sample_games(&args.games_dir, epoch, sample_games)?;
        write_stats(
            &args.stats_dir,
            epoch,
            games_in_epoch,
            total_score,
            total_length,
            avg_score,
            avg_length,
            value_loss,
            policy_loss,
            timer.passed(),
        )?;
    }

    Ok(())
}

async fn collect_epoch_games(
    network: TicTacToeResNet,
    args: &TrainArgs,
    device: tch::Device,
) -> Result<(TicTacToeResNet, EpochGames)> {
    let mut batch_size = args.inference_batch_size;
    let parallelism = batch_size
        .checked_add(args.parallelism_padding)
        .context("initial parallelism overflowed usize")?;
    let mut executor = ExecutorScope::new(
        network,
        parallelism,
        batch_size,
        Duration::from_millis(args.batch_accumulation_ms),
        (Kind::Float, device),
    );

    for _ in 0..args.games_per_epoch {
        let simulations = args.simulations;
        let c_puct = args.c_puct;
        std::mem::drop(executor.spawn(move |handle| async move {
            let evaluator =
                NetworkPositionEvaluator::<TicTacToeResNet, TicTacToeCodec>::new(handle);
            generate_self_played_game(
                BoardState::new(),
                simulations,
                c_puct,
                self_play_temperature,
                evaluator,
            )
            .await
        }));
    }

    let ramp_interval = Duration::from_secs(args.parallelism_step_seconds);
    let ramp_sleep = tokio::time::sleep(ramp_interval);
    tokio::pin!(ramp_sleep);
    let mut completed_ramps = 0;

    let mut games = Vec::with_capacity(args.games_per_epoch);
    let mut total_score = 0.0;
    let mut total_length = 0;
    loop {
        tokio::select! {
            () = &mut ramp_sleep, if completed_ramps < args.parallelism_steps => {
                println!("Increasing parallelism by {}", args.parallelism_step);
                executor.increase_parallelism(args.parallelism_step).await;
                batch_size = batch_size
                    .checked_add(args.parallelism_step)
                    .context("inference batch size overflowed usize")?;
                executor.set_batch_size(batch_size).await;
                completed_ramps += 1;
                ramp_sleep.as_mut().reset(tokio::time::Instant::now() + ramp_interval);
            }
            task_result = executor.next() => {
                let Some(record) = task_result else {
                    break;
                };
                let record = record?;
                total_score += record.value_for(Seat::First);
                total_length += record.plies.len();
                games.push(extract_training_game::<_, TicTacToeCodec>(record)?);
                println!("Game finished, {} more to go", executor.len());
            }
        }
    }

    let network = executor.join().await;
    Ok((
        network,
        EpochGames {
            games,
            total_score,
            total_length,
        },
    ))
}

fn train_epoch(
    network: &TicTacToeResNet,
    optimizer: &mut Optimizer,
    replay: &ReplayBuffer,
    batch_size: usize,
    device: tch::Device,
) -> (f32, f32) {
    let augmentation_count = TicTacToeCodec::augmentation_count();
    let mut training_samples = replay
        .iter()
        .flatten()
        .flat_map(|sample| (0..augmentation_count).map(move |augmentation| (sample, augmentation)))
        .collect::<Vec<_>>();
    training_samples.shuffle(&mut rand::rng());

    let mut total_value_loss = 0.0;
    let mut total_policy_loss = 0.0;
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
        let policy_loss = (policies * expected_policies).sum(None) / chunk.len() as f64;
        optimizer.backward_step(&(&value_loss - &policy_loss));

        total_value_loss += f32::try_from(&value_loss).unwrap();
        total_policy_loss += f32::try_from(&policy_loss).unwrap();
    }
    (total_value_loss, total_policy_loss)
}

fn render_sample_games(games_dir: &Path, epoch: usize, games: Vec<ReplayGame>) -> Result<()> {
    for (id, game) in games.into_iter().enumerate() {
        generate_game_image(&game).save(game_path(games_dir, epoch, id))?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn write_stats(
    stats_dir: &Path,
    epoch: usize,
    games_in_epoch: usize,
    total_score: f32,
    total_length: usize,
    avg_score: f32,
    avg_length: f32,
    value_loss: f32,
    policy_loss: f32,
    duration: Duration,
) -> Result<()> {
    let mut file = fs::File::create(stats_path(stats_dir, epoch))?;
    writeln!(file, "Total games: {games_in_epoch}")?;
    writeln!(file, "Average game length: {avg_length}")?;
    writeln!(file, "Average game score: {avg_score}")?;
    writeln!(file, "Total game length: {total_length}")?;
    writeln!(file, "Total game score: {total_score}")?;
    writeln!(file, "Value loss: {value_loss}")?;
    writeln!(file, "Policy loss: {policy_loss}")?;
    writeln!(file, "Epoch duration: {duration:?}")?;
    Ok(())
}

fn game_path(root: &Path, epoch: usize, id: usize) -> PathBuf {
    root.join(format!("{epoch:02}.{id:02}.png"))
}

fn stats_path(root: &Path, epoch: usize) -> PathBuf {
    root.join(format!("{epoch:02}.stats"))
}

fn self_play_temperature(turn: usize) -> f32 {
    match turn {
        0..8 => 1.0,
        value @ 8..13 => 1.0 - (value - 7) as f32 * 0.1,
        13.. => 0.5,
    }
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

    let growth = args
        .parallelism_step
        .checked_mul(args.parallelism_steps)
        .context("parallelism growth overflowed usize")?;
    let max_parallelism = args
        .inference_batch_size
        .checked_add(args.parallelism_padding)
        .and_then(|parallelism| parallelism.checked_add(growth))
        .context("maximum parallelism overflowed usize")?;
    ensure!(
        max_parallelism <= tokio::sync::Semaphore::MAX_PERMITS,
        "maximum parallelism exceeds Tokio's semaphore limit"
    );
    Ok(())
}
