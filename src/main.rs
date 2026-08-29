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
        apply_temperature, generate_self_played_game, sample_policy, AlphaZeroAdapter,
        AlphaZeroNet, ExecutorScope, Game, MonteCarloTree, MoveParameters,
        NetworkBatchedExecutorHandle, RootNoise, TerminationState,
    },
    tictactoe::{
        generate_game_image, BoardState, CellState, TicTacToeMove, TicTacToeResAlphaZeroAdapter,
        TicTacToeResNet,
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

#[allow(unused)]
async fn play_nn_only_game() -> anyhow::Result<()> {
    let mut vs = nn::VarStore::new(detect_device());

    let net = TicTacToeResNet::new(&vs.root());
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
        let _ = executor.spawn(|mut handle| async move {
            let mut state = BoardState::new();
            let mut history = vec![];
            let _outcome = loop {
                let moves = match state.get_state() {
                    TerminationState::Terminal(v) => break v,
                    TerminationState::Moves(moves) => moves,
                };

                let (value, policy) = handle
                    .execute(TicTacToeResAlphaZeroAdapter::convert_game_to_nn_input(
                        &state,
                    ))
                    .await;
                let value = f32::try_from(value).unwrap();
                let policy = TicTacToeResAlphaZeroAdapter::get_estimated_policy(&policy, &moves);
                let r#move = sample_policy(&policy, &mut rand::rng());
                let new_state = state.make_move(&moves[r#move]);
                history.push((std::mem::replace(&mut state, new_state), policy, value));
            };
            history
        });
    }
    let hist = executor.next().await.unwrap();
    generate_game_image(&hist).save("out.png")?;
    for (_, _, value) in &hist {
        println!("Predicted value is {value}");
    }
    Ok(())
}

#[allow(unused)]
async fn play_with_human(
    handle: NetworkBatchedExecutorHandle<TicTacToeResNet>,
    human_first: bool,
    sims: usize,
) -> anyhow::Result<()> {
    let mut rng = SmallRng::from_rng(&mut rand::rng());
    let mut current_player = human_first;
    let mut state = BoardState::new();

    fn print_state(state: &BoardState) {
        for i in 0..BoardState::N {
            for j in 0..BoardState::N {
                let sym = match state[(i, j)] {
                    CellState::X => 'X',
                    CellState::O => 'O',
                    CellState::Empty => ' ',
                };
                print!("{sym}");
            }
            println!();
        }
    }

    let mut tree = MonteCarloTree::<BoardState, TicTacToeResNet, TicTacToeResAlphaZeroAdapter>::new(
        state.clone(),
        handle,
        RootNoise::None,
    );

    let v = loop {
        let moves = match state.get_state() {
            TerminationState::Terminal(v) => break v,
            TerminationState::Moves(moves) => moves,
        };
        let r#move = if current_player {
            print_state(&state);
            println!(
                "NN's value prediction: {:?}",
                tree.get_network_state_estimation()
            );
            // tree.do_simulations(sims, 1.0, &mut rng).await;
            tree.do_simulations(2, 1.0, &mut rng).await;
            let [i, j] = {
                let mut s = String::new();
                io::stdin().read_line(&mut s)?;
                s.trim()
                    .split(' ')
                    .map(FromStr::from_str)
                    .collect::<Result<Vec<usize>, _>>()?
                    .try_into()
                    .unwrap()
            };
            let m = TicTacToeMove(i - 1, j - 1);
            moves.iter().position(|&v| v == m).unwrap()
        } else {
            tree.do_simulations(sims, 1.0, &mut rng).await;
            let policy = apply_temperature(&tree.get_policy(), 0.33);
            let r#move = sample_policy(&policy, &mut rng);
            println!(
                "Network's move {} {}",
                moves[r#move].0 + 1,
                moves[r#move].1 + 1
            );
            r#move
        };
        {
            let (stat_info, dyn_info) = tree.get_move_stats(r#move).unwrap();
            println!(
                "Move: descends={}, avg_score={}, net_policy={}, total_descends={}, most_descends={}",
                dyn_info.descends,
                dyn_info.get_avg_score(),
                stat_info.priority,
                tree.get_total_descends().unwrap(),
                tree.most_descends().unwrap(),
            );
        }
        current_player ^= moves[r#move].is_player_switch();
        tree.do_move(r#move);
        state = state.make_move(&moves[r#move]);
    };
    if (v > 0.0) ^ current_player {
        println!("Looser!");
    } else {
        println!("Congrats");
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

    let mut net = TicTacToeResNet::new(&vs.root());
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
            let _ = executor.spawn(|handle| async {
                generate_self_played_game::<
                    BoardState,
                    TicTacToeResNet,
                    TicTacToeResAlphaZeroAdapter,
                    _,
                >(
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
                    handle,
                )
                .await
            });
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
                    let res = match task_result {
                        Some(v) => v,
                        None => break,
                    };
                    total_score += res[0].2;
                    total_length += res.len();
                    history.push(res);
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
            .map(Vec::clone)
            .collect::<Vec<_>>();

        games_hist.extend(history.into_iter());
        while games_hist.len() > GAMES_IN_HISTORY {
            games_hist.pop_front();
        }

        let augmentation_count = TicTacToeResAlphaZeroAdapter::augmentation_count();
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
                let (state, policy, value) = sample;
                let state_tensor = TicTacToeResAlphaZeroAdapter::convert_game_to_nn_input(state);
                let policy_tensor = TicTacToeResAlphaZeroAdapter::convert_policy_to_nn(
                    policy,
                    state.get_state().get_moves().unwrap(),
                );
                let (state_tensor, policy_tensor) = TicTacToeResAlphaZeroAdapter::augment(
                    &state_tensor,
                    &policy_tensor,
                    augmentation,
                );
                states.push(state_tensor);
                policies.push(policy_tensor);
                values.push(*value);
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
