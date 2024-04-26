use std::{future::Future, sync::Arc, time::Duration};

use futures::{stream::FuturesUnordered, StreamExt};
use tch::{Device, Kind};
use tokio::{
    sync::{mpsc::Sender, Semaphore},
    task::JoinHandle,
};

use super::{AlphaZeroNet, BatcherCommand, NetworkBatchedExecutor, NetworkBatchedExecutorHandle};

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

pub struct ExecutorScope<T, TNet: AlphaZeroNet> {
    results: FuturesUnordered<JoinHandle<T>>,
    parallelism: Arc<Semaphore>,
    parallelism_tokens: usize,
    batch_size_manager: BatchSizeManager,
    executor_cmd: Sender<BatcherCommand>,
    executor_handle: NetworkBatchedExecutorHandle<TNet>,
    executor: JoinHandle<TNet>,
}

impl<T, TNet: AlphaZeroNet + Send + 'static> ExecutorScope<T, TNet> {
    pub fn new(
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

    pub fn spawn<
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

    pub async fn increase_parallelism(&mut self, delta: usize) {
        self.parallelism_tokens += delta;
        self.parallelism.add_permits(delta);
        self.on_tasks_count_change().await;
    }

    pub async fn set_batch_size(&mut self, batch_size: usize) {
        if let Some(v) = self.batch_size_manager.change_max_batch_size(batch_size) {
            self.executor_cmd
                .send(BatcherCommand::SetBatchSize(v))
                .await
                .unwrap();
        }
        self.on_tasks_count_change().await;
    }

    pub async fn on_tasks_count_change(&mut self) {
        let tasks = self.len().min(self.parallelism_tokens);
        if let Some(batch) = self.batch_size_manager.on_task_count_change(tasks) {
            self.executor_cmd
                .send(BatcherCommand::SetBatchSize(batch))
                .await
                .unwrap();
        }
    }

    pub async fn next(&mut self) -> Option<T> {
        let res = self.results.next().await.map(Result::unwrap);
        self.on_tasks_count_change().await;
        res
    }

    pub async fn join(self) -> TNet {
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

    pub fn len(&self) -> usize {
        self.results.len()
    }
}
