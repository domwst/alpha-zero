use std::{path::PathBuf, time::Duration};

use pytorch::{
    alpha_zero::{generate_self_played_game, AlphaZeroAdapter, AlphaZeroNet, ExecutorScope, Game},
    tictactoe::{generate_game_image, BoardState, TicTacToeAlphaZeroAdapter, TicTacToeNet},
};
use rand::{
    seq::{IteratorRandom, SliceRandom},
    thread_rng,
};
use tap::{tap, Tap};
use tch::{
    nn::{self, OptimizerConfig},
    Device, Kind, Tensor,
};
use unzip3::Unzip3;

fn get_checkpoint_file(epoch: usize) -> PathBuf {
    PathBuf::from(format!("checkpoints/{epoch:02}.safetensors"))
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut vs = nn::VarStore::new(Device::Mps);
    println!("Going to use device {:?}", vs.device());

    let mut net = TicTacToeNet::new(&vs.root());
    let mut opt = nn::Adam::default().build(&vs, 1e-4)?;

    let mut start_epoch = 0;
    if get_checkpoint_file(0)
        .parent()
        .map(|p| p.exists())
        .unwrap_or(false)
    {
        println!("Checkpoint folder found");
        let mut cp = 0;
        while get_checkpoint_file(cp).exists() {
            cp += 1;
        }
        if cp > 0 {
            cp -= 1;
            println!("Restoring from checkpoint {cp}");
            vs.load(get_checkpoint_file(cp))?;
            start_epoch = cp + 1;
        }
    }

    // let executor = NetworkBatchedExecutor::new(net);
    //
    // let mut worker_handles = FuturesUnordered::new();

    for epoch in start_epoch.. {
        let mut executor = ExecutorScope::new(
            net,
            192,
            128,
            Duration::from_millis(100),
            (Kind::Float, vs.device()),
        );

        let total_games = 600;
        // let total_games = 1;
        for _ in 0..total_games {
            executor.spawn(|handle| async {
                generate_self_played_game::<BoardState, TicTacToeNet, TicTacToeAlphaZeroAdapter, _>(
                    BoardState::new(),
                    // 128,
                    // 512,
                    // 2048,
                    32,
                    1.0 / 32.0,
                    |_| 1.0,
                    handle,
                )
                .await
            });
        }

        let mut batch_size = 128;

        let (lim_tx, mut lim_rx) = tokio::sync::mpsc::channel(1);
        tokio::spawn({
            async move {
                for _ in 0..24 {
                    tokio::time::sleep(Duration::from_secs(6)).await;
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
                    println!("Increasing parallelism by 16");
                    executor.increase_parallelism(16).await;
                    batch_size += 16;
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

        println!("Average score is {}", total_score / total_games as f32);
        println!(
            "Average length is {}",
            total_length as f32 / total_games as f32
        );

        net = executor.join().await;

        let sample_games = history
            .iter()
            .choose_multiple(&mut thread_rng(), 20)
            .into_iter()
            .map(Vec::clone)
            .collect::<Vec<_>>();

        let history = history
            .into_iter()
            .flatten()
            .map(|(state, policy, value)| {
                (
                    TicTacToeAlphaZeroAdapter::convert_game_to_nn_input(&state),
                    TicTacToeAlphaZeroAdapter::convert_policy_to_nn(
                        &policy,
                        &state.get_state().get_moves().unwrap(),
                    ),
                    value,
                )
            })
            .map(|(state, policy, value)| {
                TicTacToeAlphaZeroAdapter::reflect_and_augment(&state, &policy)
                    .into_iter()
                    .map(move |(state, policy)| (state, policy, Tensor::from(value)))
            })
            .flatten()
            .collect::<Vec<_>>()
            .tap_mut(|h| h.shuffle(&mut thread_rng()));

        let mut total_values_loss = 0.0;
        let mut total_policies_loss = 0.0;
        for chunk in history.chunks(1024) {
            let (states, policies, values): (Vec<_>, Vec<_>, Vec<_>) = chunk
                .into_iter()
                .map(|(state, policy, value)| (state.copy(), policy.copy(), value.copy()))
                .unzip3();

            let states = Tensor::stack(&states, 0)
                .to_kind(Kind::Float)
                .to(vs.device());
            let policies = Tensor::stack(&policies, 0)
                .to_kind(Kind::Float)
                .to(vs.device());
            let values = Tensor::stack(&values, 0)
                .to_kind(Kind::Float)
                .to(vs.device());

            let (exp_values, exp_policies) = net.forward_t(&states, true);
            let val_loss = (exp_values - values)
                .pow(&Tensor::from(2.).to_kind(Kind::Float).to(vs.device()))
                .sum(None);
            let pol_loss = (policies * exp_policies).sum(None);
            total_values_loss += f32::try_from(&val_loss).unwrap();
            total_policies_loss += f32::try_from(&pol_loss).unwrap();
            opt.backward_step(&(val_loss - pol_loss));
        }

        println!("Total value and policy loss: ({total_values_loss}, {total_policies_loss})");

        vs.save(format!("checkpoints/{epoch:02}.safetensors"))
            .unwrap();
        for (i, sample_game) in sample_games.into_iter().enumerate() {
            generate_game_image(&sample_game)
                .save(format!("games/{epoch:02}.{i:02}.png"))
                .unwrap();
        }
    }

    Ok(())
}
