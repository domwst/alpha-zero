use std::{sync::Arc, time::Duration};

use futures::{stream::FuturesUnordered, StreamExt};
use image::{math::Rect, ImageBuffer, Pixel, Rgb};
use pytorch::{
    alpha_zero::{generate_self_played_game, BatcherCommand, NetworkBatchedExecutor},
    tictactoe::{BoardState, CellState, TicTacToeAlphaZeroAdapter, TicTacToeNet},
};
use tch::{
    nn::{self, OptimizerConfig},
    Device, Kind,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let vs = nn::VarStore::new(Device::Mps);

    let net = TicTacToeNet::new(&vs.root());
    let _opt = nn::Adam::default().build(&vs, 1e-4)?;

    let executor = NetworkBatchedExecutor::new(net);

    let mut worker_handles = FuturesUnordered::new();

    let parallel_games = Arc::new(tokio::sync::Semaphore::new(192));
    let total_games = 2000;
    // let total_games = 1;
    for _ in 0..total_games {
        let h = tokio::spawn({
            // let handle1 = executor.mint_handle();
            // let handle2 = executor.mint_handle();
            let handle = executor.mint_handle();
            let parallel_games = parallel_games.clone();

            async move {
                let _permit = parallel_games.acquire().await.unwrap();
                // do_battle(BoardState::new(), 2048, 1. / 64., 0.5, handle1, handle2).await
                generate_self_played_game::<BoardState, TicTacToeNet, TicTacToeAlphaZeroAdapter, _>(
                    BoardState::new(),
                    128,
                    1.0 / 16.0,
                    |_| 1.0,
                    handle,
                )
                .await
            }
        });
        worker_handles.push(h);
    }

    let mut batch_size = 128;
    let (cmd_tx, cmd_rx) = tokio::sync::mpsc::channel(1);
    let executor = tokio::spawn({
        let device = vs.device();
        async move {
            executor
                .serve(
                    batch_size,
                    Duration::from_millis(100),
                    cmd_rx,
                    (Kind::Float, device),
                )
                .await;
        }
    });

    let (games_tx, mut games_rx) = tokio::sync::watch::channel(worker_handles.len());

    let limit_increaser = tokio::spawn({
        let parallel_games = parallel_games.clone();

        async move {
            for _ in 0..24 {
                tokio::time::sleep(Duration::from_secs(6)).await;
                println!("Increasing limits by 24 games");
                parallel_games.add_permits(24);
                batch_size += 16;
                cmd_tx
                    .send(BatcherCommand::SetBatchSize(batch_size))
                    .await
                    .unwrap();
            }
            while let Ok(()) = games_rx.changed().await {
                let v = *games_rx.borrow_and_update();
                if v > 0 && v < batch_size {
                    batch_size = v;
                    if let Err(_) = cmd_tx.send(BatcherCommand::SetBatchSize(v)).await {
                        break;
                    }
                }
            }
        }
    });

    // let mut history = vec![];

    let mut total_score = 0.0;
    let mut total_length = 0;
    while let Some(res) = worker_handles.next().await {
        let res = res.unwrap();
        total_score += res[0].2;
        total_length += res.len();
        // history.append(&mut res);
        println!("Game finished, {} more to go", worker_handles.len());
        games_tx.send(worker_handles.len()).unwrap();
    }

    drop(games_tx);
    limit_increaser.await?;
    executor.await?;

    println!("Average score is {}", total_score / total_games as f32);
    println!(
        "Average length is {}",
        total_length as f32 / total_games as f32
    );

    // println!("Game score is {}", history[0].2);
    // let square = 10;
    // let fld = 19 * square;
    // let line = 5;
    // let mut img = image::RgbImage::new(
    //     fld * history.len() as u32 + line * (history.len() as u32 - 1),
    //     fld,
    // );
    //
    // let width = img.width();
    // let height = img.height();
    // fn draw_rect(img: &mut ImageBuffer<Rgb<u8>, Vec<u8>>, r: Rect, pixel: Rgb<f32>) {
    //     for i in r.x..r.x + r.width {
    //         for j in r.y..r.y + r.height {
    //             img.put_pixel(i, j, Rgb(pixel.0.map(|v| v as u8)));
    //         }
    //     }
    // }
    //
    // draw_rect(
    //     &mut img,
    //     Rect {
    //         x: 0,
    //         y: 0,
    //         width,
    //         height,
    //     },
    //     Rgb([255., 255., 255.]),
    // );
    //
    // for (i, (state, pol, _)) in history.iter().enumerate() {
    //     let x = i as u32 * (fld + line);
    //
    //     let mut pol = pol.iter().copied();
    //     let x_clr = Rgb([255., 0., 0.]);
    //     let o_clr = Rgb([0., 0., 255.]);
    //     let policy = Rgb([0., 255., 0.]);
    //
    //     for i in 0..19 {
    //         for j in 0..19 {
    //             let clr = match state[(i as usize, j as usize)] {
    //                 CellState::X => x_clr,
    //                 CellState::O => o_clr,
    //                 CellState::Empty => {
    //                     let p = pol.next().unwrap();
    //                     policy.map(|x| p * x)
    //                 }
    //             };
    //             draw_rect(
    //                 &mut img,
    //                 Rect {
    //                     x: x + i * square,
    //                     y: 0 + j * square,
    //                     width: square,
    //                     height: square,
    //                 },
    //                 clr,
    //             );
    //         }
    //     }
    //
    //     if i + 1 != history.len() {
    //         draw_rect(
    //             &mut img,
    //             Rect {
    //                 x: x + square,
    //                 y: 0,
    //                 width: line,
    //                 height,
    //             },
    //             Rgb([0., 0., 0.]),
    //         );
    //     }
    // }
    //
    // img.save("game.png")?;

    // fs::write("./history.hist", serde_json::to_vec(&history).unwrap())?;

    Ok(())
}
