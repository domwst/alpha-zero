use std::{fs, sync::Arc, time::Duration};

use futures::{stream::FuturesUnordered, StreamExt};
use pytorch::{
    alpha_zero::{generate_self_played_game, NetworkBatchedExecutor},
    tictactoe::{BoardState, TicTacToeAlphaZeroAdapter, TicTacToeNet},
};
use tch::{nn, Device, Kind};

#[tokio::main(flavor = "multi_thread", worker_threads = 12)]
async fn main() -> anyhow::Result<()> {
    let vs = nn::VarStore::new(Device::Mps);

    let net = TicTacToeNet::new(&vs.root());
    // let net_cp = net.clone();

    let executor = NetworkBatchedExecutor::new(net);

    let mut worker_handles = FuturesUnordered::new();

    let parallel_games = Arc::new(tokio::sync::Semaphore::new(1536));
    let mut total_games = 5000;
    for _ in 0..total_games {
        let h = tokio::spawn({
            // let net = net_cp.clone();
            let handle = executor.mint_handle();
            let parallel_games = parallel_games.clone();

            async move {
                let _permit = parallel_games.acquire().await.unwrap();
                generate_self_played_game::<BoardState, TicTacToeNet, TicTacToeAlphaZeroAdapter, _>(
                    BoardState::new(),
                    2048,
                    0.1,
                    |_| 1.0,
                    handle,
                )
                .await
            }
        });
        worker_handles.push(h);
    }

    let executor = tokio::spawn({
        let device = vs.device();
        async move {
            executor
                .serve(1024, Duration::from_millis(100), (Kind::Float, device))
                .await;
        }
    });

    let mut history = vec![];

    while let Some(res) = worker_handles.next().await {
        let mut res = res.unwrap();
        total_games -= 1;
        history.append(&mut res);
        println!(
            "Game finished, {total_games} more to go, {} pending",
            worker_handles.len()
        );
    }

    executor.await.unwrap();

    fs::write("./history.hist", serde_json::to_vec(&history).unwrap())?;

    Ok(())
}
