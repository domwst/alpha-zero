use std::{future::Future, path::PathBuf, sync::Arc, time::Duration};

use futures::{stream::FuturesUnordered, StreamExt};
use image::{math::Rect, ImageBuffer, Pixel, Rgb};
use pytorch::{
    alpha_zero::{
        generate_self_played_game, AlphaZeroAdapter, AlphaZeroNet, BatcherCommand, Game,
        NetworkBatchedExecutor, NetworkBatchedExecutorHandle,
    },
    tictactoe::{BoardState, CellState, TicTacToeAlphaZeroAdapter, TicTacToeNet},
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
use tokio::{
    sync::{mpsc::Sender, Semaphore},
    task::JoinHandle,
};
use unzip3::Unzip3;

struct BatchSizeManager {
    current_batch_size: usize,
    max_batch_size: usize,
    change_ratio: (usize, usize),
}

impl BatchSizeManager {
    fn new(max_batch_size: usize, change_ratio: (usize, usize)) -> Self {
        Self {
            current_batch_size: max_batch_size,
            max_batch_size,
            change_ratio,
        }
    }

    fn on_task_count_change(&mut self, tasks: usize) -> Option<usize> {
        let (num, denom) = self.change_ratio;

        let upper_bound =
            ((self.current_batch_size * denom - 1) / num + 1).min(self.max_batch_size);
        if tasks >= upper_bound {
            let new_batch_size = tasks.min(self.max_batch_size);
            if new_batch_size != self.current_batch_size {
                self.current_batch_size = new_batch_size;
                Some(self.current_batch_size)
            } else {
                None
            }
        } else if tasks < self.current_batch_size {
            self.current_batch_size = ((self.current_batch_size * num) / denom).max(1);
            Some(self.current_batch_size)
        } else {
            None
        }
    }

    fn change_max_batch_size(&mut self, max_batch_size: usize) -> Option<usize> {
        self.max_batch_size = max_batch_size;
        if self.current_batch_size > self.max_batch_size {
            self.current_batch_size = self.max_batch_size;
            Some(self.current_batch_size)
        } else {
            None
        }
    }
}

struct ExecutorScope<T, TNet: AlphaZeroNet> {
    results: FuturesUnordered<JoinHandle<T>>,
    parallelism: Arc<Semaphore>,
    parallelism_tokens: usize,
    batch_size_manager: BatchSizeManager,
    executor_cmd: Sender<BatcherCommand>,
    executor_handle: NetworkBatchedExecutorHandle<TNet>,
    executor: JoinHandle<TNet>,
}

impl<T, TNet: AlphaZeroNet + Send + 'static> ExecutorScope<T, TNet> {
    fn new(
        nn: TNet,
        parallelism: usize,
        batch_size: usize,
        batch_acc_time: Duration,
        options: (Kind, Device),
    ) -> Self {
        let executor = NetworkBatchedExecutor::new(nn);
        let handle = executor.mint_handle();
        let (cmd_tx, cmd_rx) = tokio::sync::mpsc::channel(1);
        let executor = tokio::spawn(async move {
            executor
                .serve(batch_size, batch_acc_time, cmd_rx, options)
                .await
        });

        Self {
            results: FuturesUnordered::new(),
            parallelism: Arc::new(Semaphore::new(parallelism)),
            parallelism_tokens: parallelism,
            batch_size_manager: BatchSizeManager::new(batch_size, (5, 6)),
            executor_cmd: cmd_tx,
            executor_handle: handle,
            executor,
        }
    }

    fn spawn<
        F: FnOnce(NetworkBatchedExecutorHandle<TNet>) -> Fut,
        Fut: Future<Output = T> + 'static + Send,
    >(
        &self,
        f: F,
    ) where
        T: Send + 'static,
    {
        self.results.push({
            let f = f(self.executor_handle.clone());
            let par = self.parallelism.clone();
            tokio::spawn(async move {
                let _perm = par.acquire().await.unwrap();
                f.await
            })
        });
    }

    async fn increase_parallelism(&mut self, delta: usize) {
        self.parallelism_tokens += delta;
        self.parallelism.add_permits(delta);
        self.on_tasks_count_change().await;
    }

    async fn set_batch_size(&mut self, batch_size: usize) {
        if let Some(v) = self.batch_size_manager.change_max_batch_size(batch_size) {
            self.executor_cmd
                .send(BatcherCommand::SetBatchSize(v))
                .await
                .unwrap();
        }
        self.on_tasks_count_change().await;
    }

    async fn on_tasks_count_change(&mut self) {
        let tasks = self.len().min(self.parallelism_tokens);
        if let Some(batch) = self.batch_size_manager.on_task_count_change(tasks) {
            self.executor_cmd
                .send(BatcherCommand::SetBatchSize(batch))
                .await
                .unwrap();
        }
    }

    async fn next(&mut self) -> Option<T> {
        let res = self.results.next().await.map(Result::unwrap);
        self.on_tasks_count_change().await;
        res
    }

    async fn join(self) -> TNet {
        assert_eq!(self.len(), 0);

        let Self {
            executor,
            executor_handle,
            executor_cmd,
            ..
        } = self;
        drop((executor_handle, executor_cmd));
        executor.await.unwrap()
    }

    fn len(&self) -> usize {
        self.results.len()
    }
}

fn generate_game_image(history: &[(BoardState, Vec<f32>, f32)]) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    let square = 10;
    let fld = 19 * square;
    let line = 5;
    let mut img = image::RgbImage::new(
        fld * history.len() as u32 + line * (history.len() as u32 - 1),
        fld,
    );

    let width = img.width();
    let height = img.height();
    fn draw_rect(img: &mut ImageBuffer<Rgb<u8>, Vec<u8>>, r: Rect, pixel: Rgb<f32>) {
        for i in r.x..r.x + r.width {
            for j in r.y..r.y + r.height {
                img.put_pixel(i, j, Rgb(pixel.0.map(|v| v as u8)));
            }
        }
    }

    draw_rect(
        &mut img,
        Rect {
            x: 0,
            y: 0,
            width,
            height,
        },
        Rgb([255., 255., 255.]),
    );

    for (i, (state, pol, _)) in history.iter().enumerate() {
        let x = i as u32 * (fld + line);

        let mut pol = pol.iter().copied();
        let x_clr = Rgb([255., 0., 0.]);
        let o_clr = Rgb([0., 0., 255.]);
        let policy = Rgb([0., 255., 0.]);

        for i in 0..19 {
            for j in 0..19 {
                let clr = match state[(i as usize, j as usize)] {
                    CellState::X => x_clr,
                    CellState::O => o_clr,
                    CellState::Empty => {
                        let p = pol.next().unwrap();
                        policy.map(|x| p * x)
                    }
                };
                draw_rect(
                    &mut img,
                    Rect {
                        x: x + i * square,
                        y: 0 + j * square,
                        width: square,
                        height: square,
                    },
                    clr,
                );
            }
        }
    }

    img
}

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
            (val_loss - pol_loss / 19).backward();
            opt.step();
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
