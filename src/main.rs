mod training_snapshot;

use std::{
    fs,
    io::{self, Write},
    path::{Path, PathBuf},
    str::FromStr,
    time::Duration,
};

use alz::{
    alpha_zero::{
        extract_training_game, generate_self_played_game, run_match, Agent, AlphaZeroNet,
        ExecutorScope, MctsAgent, MoveDecision, NetworkBatchedExecutorHandle,
        NetworkPositionEvaluator, PolicyAgent, PositionCodec, RootNoise, Seat, Shared,
        TrainingCodec, Turn, Versus,
    },
    tictactoe::{
        generate_game_image, BoardState, CellState, TicTacToeCodec, TicTacToeMove, TicTacToeResNet,
    },
    util::Timer,
};
use rand::{
    rngs::SmallRng,
    seq::{IteratorRandom, SliceRandom},
    SeedableRng,
};
use tch::{
    nn::{self, OptimizerConfig},
    Device, Kind, Reduction, Tensor,
};

use training_snapshot::{
    find_latest_snapshot, load_training_snapshot, save_training_snapshot, ReplayBuffer,
};

const GAMES_IN_HISTORY: usize = 1800;

fn get_game_pic_file(epoch: usize, id: usize) -> PathBuf {
    PathBuf::from(format!("games/{epoch:02}.{id:02}.png"))
}

fn get_stats_file(epoch: usize) -> PathBuf {
    PathBuf::from(format!("stats/{epoch:02}.stats"))
}

fn detect_device() -> Device {
    let device = cfg_select! {
        target_os = "macos" => Device::Mps,
        _ => Device::cuda_if_available(),
    };
    eprintln!("Using device {device:?}");
    device
}

fn print_state(state: &BoardState) {
    for row in 0..BoardState::N {
        for column in 0..BoardState::N {
            let symbol = match state[(row, column)] {
                CellState::X => 'X',
                CellState::O => 'O',
                CellState::Empty => ' ',
            };
            print!("{symbol}");
        }
        println!();
    }
}

struct HumanAgent;

impl Agent<BoardState> for HumanAgent {
    fn select_move<'a>(
        &'a mut self,
        turn: Turn<'a, BoardState>,
    ) -> impl std::future::Future<Output = anyhow::Result<MoveDecision>> + Send + 'a {
        let state = turn.state.clone();
        let legal_moves = turn.legal_moves.to_vec();
        async move {
            tokio::task::spawn_blocking(move || {
                print_state(&state);
                let [row, column] = {
                    let mut input = String::new();
                    io::stdin().read_line(&mut input)?;
                    input
                        .split_whitespace()
                        .map(FromStr::from_str)
                        .collect::<Result<Vec<usize>, _>>()?
                        .try_into()
                        .map_err(|_| anyhow::anyhow!("expected a row and column"))?
                };
                if row == 0 || column == 0 {
                    anyhow::bail!("row and column are one-based");
                }
                let selected = TicTacToeMove(row - 1, column - 1);
                let move_index = legal_moves
                    .iter()
                    .position(|&candidate| candidate == selected)
                    .ok_or_else(|| anyhow::anyhow!("selected move is not legal"))?;
                Ok(MoveDecision::new(move_index))
            })
            .await?
        }
    }
}

#[allow(unused)]
async fn play_nn_only_game() -> anyhow::Result<()> {
    let mut vs = nn::VarStore::new(detect_device());

    let net = TicTacToeResNet::new(vs.root());
    // let mut opt = nn::Adam::default().wd(1e-4).build(&vs, 1e-3)?;

    find_latest_snapshot(Path::new("checkpoints"))?
        .ok_or_else(|| anyhow::anyhow!("No complete training snapshot found"))?
        .load_model(&mut vs)?;

    let mut executor = ExecutorScope::new(
        net,
        1,
        1,
        Duration::from_secs(100),
        (Kind::Float, vs.device()),
    );
    {
        std::mem::drop(executor.spawn(|handle| async move {
            let evaluator =
                NetworkPositionEvaluator::<TicTacToeResNet, TicTacToeCodec>::new(handle);
            let agent = PolicyAgent::<BoardState, _, _, _>::new(
                evaluator,
                SmallRng::from_rng(&mut rand::rng()),
                |_| 1.0,
            );
            let mut controller = Shared::new(agent);
            run_match(BoardState::new(), &mut controller).await
        }));
    }
    let record = executor.next().await.unwrap()?;
    for ply in &record.plies {
        if let Some(value) = ply.decision.diagnostics.value_estimate {
            println!("Predicted value is {value}");
        }
    }
    executor.join().await;
    Ok(())
}

#[allow(unused)]
async fn play_with_human(
    handle: NetworkBatchedExecutorHandle<TicTacToeResNet>,
    human_first: bool,
    sims: usize,
) -> anyhow::Result<()> {
    let start = BoardState::new();
    let evaluator = NetworkPositionEvaluator::<TicTacToeResNet, TicTacToeCodec>::new(handle);
    let network = MctsAgent::new(
        start.clone(),
        evaluator,
        RootNoise::None,
        sims,
        1.0,
        SmallRng::from_rng(&mut rand::rng()),
        |_| 0.33,
    );
    let (record, human_seat) = if human_first {
        let mut controller = Versus::new(HumanAgent, network);
        (run_match(start, &mut controller).await?, Seat::First)
    } else {
        let mut controller = Versus::new(network, HumanAgent);
        (run_match(start, &mut controller).await?, Seat::Second)
    };
    print_state(&record.terminal_state);
    match record.value_for(human_seat) {
        value if value > 0.0 => println!("Congrats"),
        value if value < 0.0 => println!("You lost"),
        _ => println!("Draw"),
    }

    Ok(())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // let mut vs = nn::VarStore::new(detect_device());
    //
    // let net = TicTacToeResNet::new(&vs.root());
    // // let mut opt = nn::Adam::default().wd(1e-4).build(&vs, 1e-3)?;
    //
    // find_latest_snapshot(Path::new("checkpoints"))?
    //     .unwrap()
    //     .load_model(&mut vs)?;
    //
    // let mut executor = ExecutorScope::new(
    //     net,
    //     1,
    //     1,
    //     Duration::from_secs(100),
    //     (Kind::Float, vs.device()),
    // );
    // let _ = executor.spawn(|handle| async move {
    //     play_with_human(handle, true, 4096).await.unwrap();
    // });
    // executor.next().await;
    // executor.join().await;
    // return Ok(());

    // play_nn_only_game().await?;
    // let mut vs = nn::VarStore::new(Device::Mps);
    // return Ok(());
    let mut vs = nn::VarStore::new(detect_device());

    let mut net = TicTacToeResNet::new(vs.root());
    let mut opt = nn::Adam::default().wd(1e-4).build(&vs, 1e-3)?;

    let mut start_epoch = 0;
    let checkpoint_dir = PathBuf::from("checkpoints");
    if !checkpoint_dir.exists() {
        println!("Creating checkpoints folder");
        fs::create_dir_all(&checkpoint_dir)?;
    }

    let mut games_hist = ReplayBuffer::new();
    if let Some(snapshot) = find_latest_snapshot(&checkpoint_dir)? {
        let epoch = snapshot.epoch();
        println!("Restoring complete training snapshot {epoch}");
        games_hist = load_training_snapshot(&snapshot, &mut vs, &mut opt)?;
        start_epoch = epoch + 1;
    }

    if let Some(p) = get_game_pic_file(0, 0).parent() {
        if !p.exists() {
            println!("Creating games directory");
            fs::create_dir_all(p)?;
        }
    }

    if let Some(p) = get_stats_file(0).parent() {
        if !p.exists() {
            println!("Creating stats directory");
            fs::create_dir_all(p)?;
        }
    }

    for epoch in start_epoch.. {
        let mut batch_size = 128;
        let timer = Timer::new();
        let mut executor = ExecutorScope::new(
            net,
            batch_size + 32,
            batch_size,
            Duration::from_millis(100),
            (Kind::Float, vs.device()),
        );

        let games_in_epoch = 600;
        // let total_games = 1;
        for _ in 0..games_in_epoch {
            std::mem::drop(executor.spawn(|handle| async {
                let evaluator =
                    NetworkPositionEvaluator::<TicTacToeResNet, TicTacToeCodec>::new(handle);
                generate_self_played_game(
                    BoardState::new(),
                    // 128,
                    // 512,
                    2048,
                    // 256,
                    // 32,
                    1.0,
                    |turn: usize| match turn {
                        0..8 => 1.0,
                        v @ 8..13 => 1.0 - (v - 7) as f32 * 0.1,
                        13.. => 0.5,
                    },
                    evaluator,
                )
                .await
            }));
        }

        let (lim_tx, mut lim_rx) = tokio::sync::mpsc::channel(1);
        tokio::spawn({
            async move {
                for _ in 0..8 {
                    tokio::time::sleep(Duration::from_secs(60)).await;
                    if lim_tx.send(()).await.is_err() {
                        break;
                    }
                }
            }
        });

        let mut history = vec![];

        let mut total_score = 0.0;
        let mut total_length = 0;
        loop {
            tokio::select! {
                Some(()) = lim_rx.recv() => {
                    let delta = 16;
                    println!("Increasing parallelism by {delta}");
                    executor.increase_parallelism(delta).await;
                    batch_size += delta;
                    executor.set_batch_size(batch_size).await;
                }
                task_result = executor.next() => {
                    let record = match task_result {
                        Some(v) => v?,
                        None => break,
                    };
                    total_score += record.value_for(Seat::First);
                    total_length += record.plies.len();
                    history.push(extract_training_game::<_, TicTacToeCodec>(record)?);
                    println!("Game finished, {} more to go", executor.len());
                }
            }
        }

        let avg_score = total_score / games_in_epoch as f32;
        let avg_length = total_length as f32 / games_in_epoch as f32;
        println!("Average score is {}", avg_score);
        println!("Average length is {}", avg_length);

        net = executor.join().await;

        let sample_games = history
            .iter()
            .sample(&mut rand::rng(), 20)
            .into_iter()
            .cloned()
            .collect::<Vec<_>>();

        games_hist.extend(history.into_iter());
        while games_hist.len() > GAMES_IN_HISTORY {
            games_hist.pop_front();
        }

        let augmentation_count = TicTacToeCodec::augmentation_count();
        let mut training_samples = games_hist
            .iter()
            .flatten()
            .flat_map(|sample| {
                (0..augmentation_count).map(move |augmentation| (sample, augmentation))
            })
            .collect::<Vec<_>>();
        training_samples.shuffle(&mut rand::rng());

        let mut total_values_loss = 0.0;
        let mut total_policies_loss = 0.0;
        for chunk in training_samples.chunks(1024) {
            let mut states = Vec::with_capacity(chunk.len());
            let mut policies = Vec::with_capacity(chunk.len());
            let mut values = Vec::with_capacity(chunk.len());
            for &(sample, augmentation) in chunk {
                let state_tensor = TicTacToeCodec::encode_position(&sample.state);
                let policy_tensor = TicTacToeCodec::policy_to_tensor(&sample.policy);
                let (state_tensor, policy_tensor) =
                    TicTacToeCodec::augment(&state_tensor, &policy_tensor, augmentation);
                states.push(state_tensor);
                policies.push(policy_tensor);
                values.push(sample.value);
            }

            let states = Tensor::stack(&states, 0)
                .to_kind(Kind::Float)
                .to(vs.device());
            let policies = Tensor::stack(&policies, 0)
                .to_kind(Kind::Float)
                .to(vs.device());
            let values = Tensor::from_slice(&values)
                .to_kind(Kind::Float)
                .to(vs.device());

            let (exp_values, exp_policies) = net.forward_t(&states, true);
            // let val_loss = (exp_values - values)
            //     .pow(&Tensor::from(2.).to_kind(Kind::Float).to(vs.device()))
            //     .sum(None);
            let val_loss = exp_values.mse_loss(&values, Reduction::Mean);
            let pol_loss = (policies * exp_policies).sum(None) / chunk.len() as f64;
            opt.backward_step(&(&val_loss - &pol_loss));

            total_values_loss += f32::try_from(&val_loss).unwrap();
            total_policies_loss += f32::try_from(&pol_loss).unwrap();
        }

        println!("Total value and policy loss: ({total_values_loss}, {total_policies_loss})");

        save_training_snapshot(&checkpoint_dir, epoch, &vs, &opt, &games_hist)?;
        for (i, sample_game) in sample_games.into_iter().enumerate() {
            generate_game_image(&sample_game).save(get_game_pic_file(epoch, i))?;
        }

        let mut f = fs::File::create(get_stats_file(epoch))?;
        writeln!(f, "Total games: {games_in_epoch}")?;
        writeln!(f, "Average game length: {avg_length}")?;
        writeln!(f, "Average game score: {avg_score}")?;
        writeln!(f, "Total game length: {total_length}")?;
        writeln!(f, "Total game score: {total_score}")?;
        writeln!(f, "Value loss: {total_values_loss}")?;
        writeln!(f, "Policy loss: {total_policies_loss}")?;
        writeln!(f, "Epoch duration: {:?}", timer.passed())?;
    }

    Ok(())
}
