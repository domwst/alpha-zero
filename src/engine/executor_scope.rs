//! Owns an inference service. Game admission belongs to GameController.
use super::{
    AlphaZeroNet, NetworkBatchStats, NetworkBatchedExecutor, NetworkBatchedExecutorHandle,
};
use std::time::Duration;
use tch::{Device, Kind};
use tokio::task::JoinHandle;

pub struct ExecutorScope<Net: AlphaZeroNet> {
    handle: Option<NetworkBatchedExecutorHandle<Net>>,
    executor: Option<JoinHandle<(Net, NetworkBatchStats)>>,
}
impl<Net: AlphaZeroNet + Send + 'static> ExecutorScope<Net> {
    pub fn new(
        nn: Net,
        queue_capacity: usize,
        batch_size: usize,
        deadline: Duration,
        options: (Kind, Device),
        batch_grid: &[usize],
    ) -> anyhow::Result<Self> {
        let executor = NetworkBatchedExecutor::new(nn, queue_capacity.max(batch_size))
            .with_batch_grid(batch_grid.to_vec())?;
        let handle = executor.mint_handle();
        let (_, receiver) = tokio::sync::mpsc::channel(1);
        let task = tokio::spawn(executor.serve(batch_size, deadline, receiver, options));
        Ok(Self {
            handle: Some(handle),
            executor: Some(task),
        })
    }
    pub fn evaluator_handle(&self) -> NetworkBatchedExecutorHandle<Net> {
        self.handle.as_ref().unwrap().clone()
    }
    pub fn live_stats(&self) -> super::LiveNetworkStats {
        self.handle.as_ref().unwrap().live_stats()
    }
    pub fn completed_evaluations(&self) -> u64 {
        self.handle.as_ref().unwrap().completed_evaluations()
    }
    pub async fn join(self) -> Net {
        self.join_with_stats().await.0
    }
    /// All passive handles and submission guards must be dropped before joining.
    pub async fn join_with_stats(mut self) -> (Net, NetworkBatchStats) {
        self.handle.take();
        self.executor
            .take()
            .unwrap()
            .await
            .expect("inference executor failed")
    }
}
impl<Net: AlphaZeroNet> Drop for ExecutorScope<Net> {
    fn drop(&mut self) {
        if let Some(executor) = &self.executor {
            executor.abort();
        }
    }
}
