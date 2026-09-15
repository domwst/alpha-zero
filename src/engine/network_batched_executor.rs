use std::{
    collections::BTreeMap,
    marker::PhantomData,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

use anyhow::{Context, Result, anyhow, ensure};
use serde::Serialize;
use tch::{Device, Kind, Tensor};
use tokio::sync::{Notify, mpsc, oneshot};

use crate::util::AtomicU64Ext;

use super::{AlphaZeroNet, masked_policy_probabilities};

#[cfg(test)]
use super::NetworkOutput;

pub fn validate_batch_grid(grid: &[usize]) -> Result<()> {
    ensure!(
        grid.iter().all(|&n| n > 0) && grid.windows(2).all(|w| w[0] < w[1]),
        "batch grid must be positive and strictly increasing"
    );
    Ok(())
}

struct InferenceRequest {
    input: Tensor,
    legal_policy_mask: Tensor,
    response: oneshot::Sender<(Tensor, Tensor)>,
    submitted_at: Instant,
}

#[derive(Default)]
struct NetworkRequestStats {
    position_encoding_us_total: AtomicU64,
    policy_mask_construction_us_total: AtomicU64,
    policy_decode_us_total: AtomicU64,
}

#[derive(Clone, Debug, Default, Serialize, serde::Deserialize)]
#[serde(default)]
pub struct NetworkBatchStats {
    /// Microsecond upper bounds (powers of two); request-weighted, not batch-weighted.
    pub request_latency_us_histogram: BTreeMap<u64, u64>,
    pub request_queue_wait_us_histogram: BTreeMap<u64, u64>,
    pub dispatch_reasons: BTreeMap<String, u64>,
    pub physical_batch_size_histogram: BTreeMap<usize, u64>,
    pub padded_requests: u64,
    pub invocations: u64,
    pub requests: u64,
    pub full_batches: u64,
    pub cancelled_requests: u64,
    pub batch_size_histogram: BTreeMap<usize, u64>,
    pub queue_wait_us_total: u64,
    pub queue_wait_us_max: u64,
    pub request_latency_us_total: u64,
    pub request_latency_us_max: u64,
    pub input_construction_us_total: u64,
    pub forward_submission_us_total: u64,
    pub policy_mask_batch_construction_us_total: u64,
    pub policy_mask_transfer_submission_us_total: u64,
    pub policy_postprocess_submission_us_total: u64,
    pub output_sync_us_total: u64,
    pub service_us_total: u64,
    pub position_encoding_us_total: u64,
    pub policy_mask_construction_us_total: u64,
    pub policy_decode_us_total: u64,
}

impl NetworkBatchStats {
    pub fn average_batch_size(&self) -> f64 {
        if self.invocations == 0 {
            0.0
        } else {
            self.requests as f64 / self.invocations as f64
        }
    }

    pub fn average_queue_wait_us(&self) -> f64 {
        if self.invocations == 0 {
            0.0
        } else {
            self.queue_wait_us_total as f64 / self.invocations as f64
        }
    }

    pub fn average_request_latency_us(&self) -> f64 {
        if self.requests == 0 {
            0.0
        } else {
            self.request_latency_us_total as f64 / self.requests as f64
        }
    }

    pub fn average_service_us(&self) -> f64 {
        if self.invocations == 0 {
            0.0
        } else {
            self.service_us_total as f64 / self.invocations as f64
        }
    }

    pub fn average_policy_mask_construction_us(&self) -> f64 {
        self.average_request_phase_us(self.policy_mask_construction_us_total)
    }

    pub fn average_position_encoding_us(&self) -> f64 {
        self.average_request_phase_us(self.position_encoding_us_total)
    }

    pub fn average_policy_decode_us(&self) -> f64 {
        self.average_request_phase_us(self.policy_decode_us_total)
    }

    pub fn average_policy_mask_batch_construction_us(&self) -> f64 {
        self.average_batch_phase_us(self.policy_mask_batch_construction_us_total)
    }

    pub fn average_policy_mask_transfer_submission_us(&self) -> f64 {
        self.average_batch_phase_us(self.policy_mask_transfer_submission_us_total)
    }

    pub fn average_policy_postprocess_submission_us(&self) -> f64 {
        self.average_batch_phase_us(self.policy_postprocess_submission_us_total)
    }

    fn average_request_phase_us(&self, total: u64) -> f64 {
        if self.requests == 0 {
            0.0
        } else {
            total as f64 / self.requests as f64
        }
    }

    fn average_batch_phase_us(&self, total: u64) -> f64 {
        if self.invocations == 0 {
            0.0
        } else {
            total as f64 / self.invocations as f64
        }
    }
}

fn elapsed_us(start: Instant) -> u64 {
    start.elapsed().as_micros().min(u64::MAX as u128) as u64
}

fn duration_us(duration: Duration) -> u64 {
    duration.as_micros().min(u64::MAX as u128) as u64
}

/// Lifecycle notifications only wake the batcher; the atomic count is authoritative.
#[derive(Default)]
struct ProducerActivity {
    active: AtomicUsize,
    outstanding: AtomicUsize,
    in_flight: AtomicUsize,
    live: Mutex<LiveNetworkStats>,
    changed: Notify,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct LiveNetworkStats {
    pub active_producers: usize,
    pub outstanding_requests: usize,
    pub in_flight_requests: usize,
    pub invocations: u64,
    pub completed_requests: u64,
    pub padded_requests: u64,
    pub average_batch_size: f64,
    pub average_latency_us: f64,
    pub max_latency_us: u64,
}

struct Outstanding<'a>(&'a ProducerActivity);
impl Drop for Outstanding<'_> {
    fn drop(&mut self) {
        self.0.outstanding.fetch_sub(1, Ordering::AcqRel);
        self.0.changed.notify_one();
    }
}

/// One sequential producer. Passive handle clones never reserve batch capacity.
pub struct SubmissionHandle<'a, Net: AlphaZeroNet> {
    handle: &'a mut NetworkBatchedExecutorHandle<Net>,
}

impl<Net: AlphaZeroNet> Drop for SubmissionHandle<'_, Net> {
    fn drop(&mut self) {
        self.handle.activity.active.fetch_sub(1, Ordering::AcqRel);
        self.handle.activity.changed.notify_one();
    }
}

impl<Net: AlphaZeroNet> SubmissionHandle<'_, Net> {
    pub async fn execute(&mut self, input: Tensor, mask: Tensor) -> Result<(Tensor, Tensor)> {
        self.handle.submit(input, mask).await
    }

    pub(super) fn record_position_encoding(&self, duration: Duration) {
        self.handle.record_position_encoding(duration);
    }

    pub(super) fn record_policy_mask_construction(&self, duration: Duration) {
        self.handle.record_policy_mask_construction(duration);
    }

    pub(super) fn record_policy_decode(&self, duration: Duration) {
        self.handle.record_policy_decode(duration);
    }
}

pub struct NetworkBatchedExecutor<Net: AlphaZeroNet> {
    activity: Arc<ProducerActivity>,
    batch_grid: Vec<usize>,
    active_producer_dispatch: bool,
    receiver: mpsc::Receiver<InferenceRequest>,
    sender: mpsc::Sender<InferenceRequest>,
    completed_requests: Arc<AtomicU64>,
    aggregate_request_stats: Arc<NetworkRequestStats>,
    nn: Net,
}

pub struct NetworkBatchedExecutorHandle<Net: AlphaZeroNet> {
    activity: Arc<ProducerActivity>,
    task_sender: mpsc::Sender<InferenceRequest>,
    completed_requests: Arc<AtomicU64>,
    local_request_stats: NetworkRequestStats,
    aggregate_request_stats: Arc<NetworkRequestStats>,
    _net: PhantomData<fn() -> Net>,
}

impl<Net: AlphaZeroNet> Clone for NetworkBatchedExecutorHandle<Net> {
    fn clone(&self) -> Self {
        Self {
            activity: self.activity.clone(),
            task_sender: self.task_sender.clone(),
            completed_requests: self.completed_requests.clone(),
            local_request_stats: NetworkRequestStats::default(),
            aggregate_request_stats: self.aggregate_request_stats.clone(),
            _net: PhantomData,
        }
    }
}

impl<Net: AlphaZeroNet> Drop for NetworkBatchedExecutorHandle<Net> {
    fn drop(&mut self) {
        let position_encoding_us = self
            .local_request_stats
            .position_encoding_us_total
            .load(Ordering::Relaxed);
        let mask_construction_us = self
            .local_request_stats
            .policy_mask_construction_us_total
            .load(Ordering::Relaxed);
        let decode_us = self
            .local_request_stats
            .policy_decode_us_total
            .load(Ordering::Relaxed);

        // Handles can finish concurrently, so aggregation still requires an RMW.
        if position_encoding_us != 0 {
            self.aggregate_request_stats
                .position_encoding_us_total
                .fetch_add(position_encoding_us, Ordering::Relaxed);
        }
        if mask_construction_us != 0 {
            self.aggregate_request_stats
                .policy_mask_construction_us_total
                .fetch_add(mask_construction_us, Ordering::Relaxed);
        }
        if decode_us != 0 {
            self.aggregate_request_stats
                .policy_decode_us_total
                .fetch_add(decode_us, Ordering::Relaxed);
        }
    }
}

impl<Net: AlphaZeroNet> NetworkBatchedExecutorHandle<Net> {
    pub fn live_stats(&self) -> LiveNetworkStats {
        let mut stats = self.activity.live.lock().unwrap().clone();
        stats.active_producers = self.activity.active.load(Ordering::Acquire);
        stats.outstanding_requests = self.activity.outstanding.load(Ordering::Acquire);
        stats.in_flight_requests = self.activity.in_flight.load(Ordering::Acquire);
        stats
    }

    pub fn completed_evaluations(&self) -> u64 {
        self.completed_requests.load(Ordering::Relaxed)
    }

    pub fn submission(&mut self) -> SubmissionHandle<'_, Net> {
        self.activity.active.fetch_add(1, Ordering::AcqRel);
        self.activity.changed.notify_one();
        SubmissionHandle { handle: self }
    }

    pub fn active_producers(&self) -> usize {
        self.activity.active.load(Ordering::Acquire)
    }

    async fn submit(&self, input: Tensor, legal_policy_mask: Tensor) -> Result<(Tensor, Tensor)> {
        ensure!(
            legal_policy_mask.kind() == Kind::Bool,
            "legal policy mask must be Boolean"
        );
        ensure!(
            legal_policy_mask.device() == Device::Cpu,
            "legal policy mask must originate on the CPU"
        );
        self.activity.outstanding.fetch_add(1, Ordering::AcqRel);
        let _outstanding = Outstanding(&self.activity);
        let (response, result) = oneshot::channel();
        self.task_sender
            .send(InferenceRequest {
                input,
                legal_policy_mask,
                response,
                submitted_at: Instant::now(),
            })
            .await
            .map_err(|_| anyhow!("network evaluator stopped before accepting the request"))?;
        result.await.context("network evaluator stopped")
    }

    pub(super) fn record_position_encoding(&self, duration: Duration) {
        self.local_request_stats
            .position_encoding_us_total
            .add_single_writer(duration_us(duration));
    }

    pub(super) fn record_policy_mask_construction(&self, duration: Duration) {
        self.local_request_stats
            .policy_mask_construction_us_total
            .add_single_writer(duration_us(duration));
    }

    pub(super) fn record_policy_decode(&self, duration: Duration) {
        self.local_request_stats
            .policy_decode_us_total
            .add_single_writer(duration_us(duration));
    }
}

pub enum BatcherCommand {
    SetBatchSize(usize),
}

impl<Net: AlphaZeroNet> NetworkBatchedExecutor<Net> {
    pub fn new(nn: Net, queue_capacity: usize) -> Self {
        assert!(queue_capacity > 0);
        let (sender, receiver) = mpsc::channel(queue_capacity);
        Self {
            activity: Arc::default(),
            batch_grid: Vec::new(),
            active_producer_dispatch: true,
            receiver,
            sender,
            completed_requests: Arc::new(AtomicU64::new(0)),
            aggregate_request_stats: Arc::new(NetworkRequestStats::default()),
            nn,
        }
    }

    /// Empty means unpadded inference; otherwise every physical shape is on this grid.
    pub fn with_batch_grid(mut self, grid: Vec<usize>) -> Result<Self> {
        validate_batch_grid(&grid)?;
        self.batch_grid = grid;
        Ok(self)
    }

    /// Benchmark control: disable activity-aware dispatch while retaining identical inference work.
    pub fn with_active_producer_dispatch(mut self, enabled: bool) -> Self {
        self.active_producer_dispatch = enabled;
        self
    }

    pub fn mint_handle(&self) -> NetworkBatchedExecutorHandle<Net> {
        NetworkBatchedExecutorHandle {
            activity: self.activity.clone(),
            task_sender: self.sender.clone(),
            completed_requests: self.completed_requests.clone(),
            local_request_stats: NetworkRequestStats::default(),
            aggregate_request_stats: self.aggregate_request_stats.clone(),
            _net: PhantomData,
        }
    }

    pub async fn serve(
        self,
        mut max_batch: usize,
        batch_acc_time: Duration,
        mut command_receiver: mpsc::Receiver<BatcherCommand>,
        (kind, device): (Kind, Device),
    ) -> (Net, NetworkBatchStats) {
        let NetworkBatchedExecutor {
            activity,
            batch_grid,
            active_producer_dispatch,
            mut receiver,
            nn,
            sender,
            completed_requests,
            aggregate_request_stats,
        } = self;
        drop(sender);
        assert!(max_batch > 0);
        if let Some(&largest) = batch_grid.last() {
            max_batch = max_batch.min(largest);
        }

        let mut pending = Vec::with_capacity(max_batch);
        let mut input_closed = false;
        let mut commands_open = true;
        let mut stats = NetworkBatchStats::default();

        loop {
            while pending.is_empty() && !input_closed {
                tokio::select! {
                    request = receiver.recv() => match request {
                        Some(request) => pending.push(request),
                        None => input_closed = true,
                    },
                    command = command_receiver.recv(), if commands_open => match command {
                        Some(BatcherCommand::SetBatchSize(size)) if size > 0 => {
                            tracing::debug!(batch_size = size, "network batch size changed");
                            max_batch = size.min(batch_grid.last().copied().unwrap_or(size));
                        }
                        Some(BatcherCommand::SetBatchSize(_)) => {
                            tracing::warn!("ignoring zero network batch size");
                        }
                        None => commands_open = false,
                    },
                }
            }

            if pending.is_empty() && input_closed {
                break;
            }

            let deadline = tokio::time::sleep(batch_acc_time);
            tokio::pin!(deadline);
            let mut timed_out = false;
            while pending.len()
                < if active_producer_dispatch {
                    max_batch.min(activity.active.load(Ordering::Acquire).max(1))
                } else {
                    max_batch
                }
                && !input_closed
            {
                tokio::select! {
                    () = &mut deadline => { timed_out = true; break; },
                    () = activity.changed.notified(), if active_producer_dispatch => {},
                    request = receiver.recv() => match request {
                        Some(request) => pending.push(request),
                        None => input_closed = true,
                    },
                    command = command_receiver.recv(), if commands_open => match command {
                        Some(BatcherCommand::SetBatchSize(size)) if size > 0 => {
                            tracing::debug!(batch_size = size, "network batch size changed");
                            max_batch = size.min(batch_grid.last().copied().unwrap_or(size));
                        }
                        Some(BatcherCommand::SetBatchSize(_)) => {
                            tracing::warn!("ignoring zero network batch size");
                        }
                        None => commands_open = false,
                    },
                }
            }

            let before_retain = pending.len();
            pending.retain(|request| !request.response.is_closed());
            stats.cancelled_requests += (before_retain - pending.len()) as u64;
            if pending.is_empty() {
                continue;
            }
            let batch_len = pending.len().min(max_batch);
            let requests = pending.drain(..batch_len).collect::<Vec<_>>();
            let reason = if batch_len == max_batch {
                "maximum"
            } else if timed_out {
                "deadline"
            } else if input_closed {
                "closed"
            } else {
                "active_producers"
            };
            *stats.dispatch_reasons.entry(reason.into()).or_default() += 1;
            let queue_wait_us = requests
                .first()
                .map_or(0, |request| elapsed_us(request.submitted_at));
            let mut inputs = Vec::with_capacity(batch_len);
            let mut legal_policy_masks = Vec::with_capacity(batch_len);
            let mut responses = Vec::with_capacity(batch_len);
            let mut submitted_at = Vec::with_capacity(batch_len);
            for request in requests {
                *stats
                    .request_queue_wait_us_histogram
                    .entry(elapsed_us(request.submitted_at).max(1).next_power_of_two())
                    .or_default() += 1;
                inputs.push(request.input);
                legal_policy_masks.push(request.legal_policy_mask);
                responses.push(request.response);
                submitted_at.push(request.submitted_at);
            }

            let physical_batch = batch_grid
                .iter()
                .copied()
                .find(|&size| size >= batch_len)
                .unwrap_or(batch_len);
            for _ in batch_len..physical_batch {
                // Duplicate a valid row: no all-masked softmax, and eval-mode has no
                // cross-example normalization. Dummy outputs are never returned.
                inputs.push(inputs[0].shallow_clone());
                legal_policy_masks.push(legal_policy_masks[0].shallow_clone());
            }
            *stats
                .physical_batch_size_histogram
                .entry(physical_batch)
                .or_default() += 1;
            stats.padded_requests += (physical_batch - batch_len) as u64;
            activity.in_flight.store(batch_len, Ordering::Release);
            let service_started = Instant::now();
            let phase_started = Instant::now();
            let input = Tensor::stack(&inputs, 0).totype(kind).to(device);
            let input_construction_us = elapsed_us(phase_started);
            let phase_started = Instant::now();
            let output = tch::no_grad(|| nn.forward_t(&input, false));
            let forward_submission_us = elapsed_us(phase_started);

            // CUDA forwards are submitted asynchronously. Build the CPU mask batch
            // afterward so this host work can overlap with the queued inference.
            let phase_started = Instant::now();
            let legal_policy_masks_cpu = Tensor::stack(&legal_policy_masks, 0);
            let policy_mask_batch_construction_us = elapsed_us(phase_started);
            assert_eq!(
                output.policy_logits.size(),
                legal_policy_masks_cpu.size(),
                "batched legal policy masks must match network policy logits"
            );
            let phase_started = Instant::now();
            let legal_policy_masks = if matches!(device, Device::Cuda(_) | Device::Mps) {
                legal_policy_masks_cpu.to_device_(device, Kind::Bool, true, false)
            } else {
                legal_policy_masks_cpu.to(device)
            };
            let policy_mask_transfer_submission_us = elapsed_us(phase_started);
            let phase_started = Instant::now();
            let policies = masked_policy_probabilities(&output.policy_logits, &legal_policy_masks);
            let policy_postprocess_submission_us = elapsed_us(phase_started);
            let phase_started = Instant::now();
            let policies = policies.to(Device::Cpu);
            let values = output.values.to(Device::Cpu);
            let output_sync_us = elapsed_us(phase_started);

            // A non-blocking CUDA copy may retain the host storage until synchronization.
            drop(legal_policy_masks_cpu);
            let service_us = elapsed_us(service_started);

            completed_requests.add_single_writer(batch_len as u64);

            for (index, (response, submitted_at)) in
                responses.into_iter().zip(submitted_at).enumerate()
            {
                let value = values.get(index as i64);
                let policy = policies.get(index as i64);
                let _ = response.send((value, policy));
                let latency_us = elapsed_us(submitted_at);
                *stats
                    .request_latency_us_histogram
                    .entry(latency_us.max(1).next_power_of_two())
                    .or_default() += 1;
                stats.request_latency_us_total =
                    stats.request_latency_us_total.saturating_add(latency_us);
                stats.request_latency_us_max = stats.request_latency_us_max.max(latency_us);
            }

            stats.invocations += 1;
            stats.requests += batch_len as u64;
            stats.full_batches += u64::from(batch_len == max_batch);
            *stats.batch_size_histogram.entry(batch_len).or_default() += 1;
            stats.queue_wait_us_total = stats.queue_wait_us_total.saturating_add(queue_wait_us);
            stats.queue_wait_us_max = stats.queue_wait_us_max.max(queue_wait_us);
            stats.input_construction_us_total = stats
                .input_construction_us_total
                .saturating_add(input_construction_us);
            stats.forward_submission_us_total = stats
                .forward_submission_us_total
                .saturating_add(forward_submission_us);
            stats.policy_mask_batch_construction_us_total = stats
                .policy_mask_batch_construction_us_total
                .saturating_add(policy_mask_batch_construction_us);
            stats.policy_mask_transfer_submission_us_total = stats
                .policy_mask_transfer_submission_us_total
                .saturating_add(policy_mask_transfer_submission_us);
            stats.policy_postprocess_submission_us_total = stats
                .policy_postprocess_submission_us_total
                .saturating_add(policy_postprocess_submission_us);
            stats.output_sync_us_total = stats.output_sync_us_total.saturating_add(output_sync_us);
            stats.service_us_total = stats.service_us_total.saturating_add(service_us);
            activity.in_flight.store(0, Ordering::Release);
            *activity.live.lock().unwrap() = LiveNetworkStats {
                invocations: stats.invocations,
                completed_requests: stats.requests,
                padded_requests: stats.padded_requests,
                average_batch_size: stats.average_batch_size(),
                average_latency_us: stats.average_request_latency_us(),
                max_latency_us: stats.request_latency_us_max,
                ..Default::default()
            };
        }

        // Handle drop flushes local stats before its task sender closes. Reaching
        // this point therefore guarantees that every shard has been aggregated.
        stats.position_encoding_us_total = aggregate_request_stats
            .position_encoding_us_total
            .load(Ordering::Relaxed);
        stats.policy_mask_construction_us_total = aggregate_request_stats
            .policy_mask_construction_us_total
            .load(Ordering::Relaxed);
        stats.policy_decode_us_total = aggregate_request_stats
            .policy_decode_us_total
            .load(Ordering::Relaxed);

        (nn, stats)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;

    struct EchoNet;

    fn all_legal_mask(actions: i64) -> Tensor {
        Tensor::ones([actions], (Kind::Bool, Device::Cpu))
    }

    impl AlphaZeroNet for EchoNet {
        fn forward_t(&self, input: &Tensor, _is_training: bool) -> NetworkOutput {
            let batch = input.size()[0];
            NetworkOutput {
                values: input.view([batch]),
                policy_logits: Tensor::zeros([batch, 1], (Kind::Float, Device::Cpu)),
            }
        }
    }

    struct RecordingNet(Arc<Mutex<Vec<i64>>>);

    impl AlphaZeroNet for RecordingNet {
        fn forward_t(&self, input: &Tensor, _is_training: bool) -> NetworkOutput {
            let batch = input.size()[0];
            self.0.lock().unwrap().push(batch);
            NetworkOutput {
                values: input.view([batch]),
                policy_logits: Tensor::zeros([batch, 1], (Kind::Float, Device::Cpu)),
            }
        }
    }

    struct FixedPolicyNet;

    impl AlphaZeroNet for FixedPolicyNet {
        fn forward_t(&self, input: &Tensor, _is_training: bool) -> NetworkOutput {
            let batch = input.size()[0];
            NetworkOutput {
                values: Tensor::zeros([batch], (Kind::Float, input.device())),
                policy_logits: Tensor::from_slice(&[1.0f32, 20.0, 3.0])
                    .to(input.device())
                    .repeat([batch, 1]),
            }
        }
    }

    #[tokio::test]
    async fn cancelled_request_does_not_change_the_next_response() {
        let executor = NetworkBatchedExecutor::new(EchoNet, 4);
        let mut handle = executor.mint_handle();
        let (command_sender, command_receiver) = mpsc::channel(1);
        let server = tokio::spawn(executor.serve(
            2,
            Duration::from_millis(50),
            command_receiver,
            (Kind::Float, Device::Cpu),
        ));

        let mut other = handle.clone();
        let other_activity = other.submission();
        {
            let mut submission = handle.submission();
            let first = submission.execute(Tensor::from_slice(&[1.0f32]), all_legal_mask(1));
            tokio::pin!(first);
            tokio::select! {
                _ = tokio::time::sleep(Duration::from_millis(1)) => {}
                result = &mut first => panic!("request unexpectedly completed: {result:?}"),
            }
        }

        drop(other_activity);
        drop(other);
        let (value, _) = handle
            .submission()
            .execute(Tensor::from_slice(&[2.0f32]), all_legal_mask(1))
            .await
            .unwrap();
        assert_eq!(f32::try_from(value).unwrap(), 2.0);
        assert_eq!(handle.completed_evaluations(), 1);

        drop((handle, command_sender));
        server.await.unwrap();
    }

    #[tokio::test]
    async fn stopped_executor_returns_an_error() {
        let executor = NetworkBatchedExecutor::new(EchoNet, 1);
        let mut handle = executor.mint_handle();
        drop(executor);

        let result = handle
            .submission()
            .execute(Tensor::from_slice(&[1.0f32]), all_legal_mask(1))
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn executor_masks_policy_logits_before_softmax() {
        let executor = NetworkBatchedExecutor::new(FixedPolicyNet, 1);
        let mut handle = executor.mint_handle();
        let (command_sender, command_receiver) = mpsc::channel(1);
        let server = tokio::spawn(executor.serve(
            1,
            Duration::ZERO,
            command_receiver,
            (Kind::Float, Device::Cpu),
        ));
        handle.record_position_encoding(Duration::from_micros(2));
        handle.record_policy_mask_construction(Duration::from_micros(3));

        let (_, policy) = handle
            .submission()
            .execute(
                Tensor::from_slice(&[0.0f32]),
                Tensor::from_slice(&[true, false, true]),
            )
            .await
            .unwrap();
        let policy = Vec::<f32>::try_from(policy).unwrap();

        assert_eq!(policy[1], 0.0);
        assert!((policy[0] + policy[2] - 1.0).abs() < 1e-6);
        assert!(policy[2] > policy[0]);
        handle.record_policy_decode(Duration::from_micros(5));

        drop((handle, command_sender));
        let (_, stats) = server.await.unwrap();
        assert_eq!(stats.requests, 1);
        assert_eq!(stats.position_encoding_us_total, 2);
        assert_eq!(stats.policy_mask_construction_us_total, 3);
        assert_eq!(stats.policy_decode_us_total, 5);
    }

    #[tokio::test]
    async fn request_timing_shards_are_aggregated_when_handles_drop() {
        let executor = NetworkBatchedExecutor::new(EchoNet, 1);
        let first = executor.mint_handle();
        let second = first.clone();
        let (command_sender, command_receiver) = mpsc::channel(1);
        let server = tokio::spawn(executor.serve(
            1,
            Duration::ZERO,
            command_receiver,
            (Kind::Float, Device::Cpu),
        ));

        first.record_position_encoding(Duration::from_micros(2));
        first.record_position_encoding(Duration::from_micros(3));
        second.record_position_encoding(Duration::from_micros(5));
        first.record_policy_mask_construction(Duration::from_micros(3));
        first.record_policy_mask_construction(Duration::from_micros(5));
        first.record_policy_decode(Duration::from_micros(7));
        second.record_policy_mask_construction(Duration::from_micros(11));
        second.record_policy_decode(Duration::from_micros(13));

        drop((first, second, command_sender));
        let (_, stats) = server.await.unwrap();
        assert_eq!(stats.position_encoding_us_total, 10);
        assert_eq!(stats.policy_mask_construction_us_total, 19);
        assert_eq!(stats.policy_decode_us_total, 20);
    }

    #[tokio::test]
    async fn shrinking_batch_size_limits_an_accumulated_batch() {
        let observed_batches = Arc::new(Mutex::new(Vec::new()));
        let executor = NetworkBatchedExecutor::new(RecordingNet(observed_batches.clone()), 8);
        let handle = executor.mint_handle();
        let mut idle = handle.clone();
        let idle_activity = idle.submission();
        let (command_sender, command_receiver) = mpsc::channel(1);
        let server = tokio::spawn(executor.serve(
            4,
            Duration::from_millis(50),
            command_receiver,
            (Kind::Float, Device::Cpu),
        ));

        let requests = (0..3)
            .map(|value| {
                let mut handle = handle.clone();
                tokio::spawn(async move {
                    handle
                        .submission()
                        .execute(Tensor::from_slice(&[value as f32]), all_legal_mask(1))
                        .await
                        .unwrap()
                })
            })
            .collect::<Vec<_>>();
        tokio::time::sleep(Duration::from_millis(5)).await;
        command_sender
            .send(BatcherCommand::SetBatchSize(2))
            .await
            .unwrap();

        for request in requests {
            let _ = request.await.unwrap();
        }
        drop(idle_activity);
        drop((idle, handle, command_sender));
        server.await.unwrap();

        let batches = observed_batches.lock().unwrap();
        assert_eq!(batches.iter().sum::<i64>(), 3);
        assert!(batches.iter().all(|size| *size <= 2));
    }
    #[tokio::test]
    async fn dropping_an_idle_producer_wakes_a_partial_batch() {
        let executor = NetworkBatchedExecutor::new(EchoNet, 4);
        let mut first = executor.mint_handle();
        let mut idle = first.clone();
        assert_eq!(first.active_producers(), 0);
        let idle_activity = idle.submission();
        let (tx, rx) = mpsc::channel(1);
        let server = tokio::spawn(executor.serve(
            4,
            Duration::from_secs(60),
            rx,
            (Kind::Float, Device::Cpu),
        ));
        let request = tokio::spawn(async move {
            first
                .submission()
                .execute(Tensor::from_slice(&[7f32]), all_legal_mask(1))
                .await
                .unwrap()
        });
        tokio::task::yield_now().await;
        drop(idle_activity);
        let (value, _) = tokio::time::timeout(Duration::from_secs(1), request)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(f32::try_from(value).unwrap(), 7.0);
        assert_eq!(idle.active_producers(), 0);
        drop((idle, tx));
        server.await.unwrap();
    }

    #[tokio::test]
    async fn three_real_requests_execute_in_one_bucket_of_four() {
        let executor = NetworkBatchedExecutor::new(EchoNet, 8)
            .with_batch_grid(vec![1, 2, 4, 8])
            .unwrap();
        let mut handles = (0..3).map(|_| executor.mint_handle()).collect::<Vec<_>>();
        let mut guards = handles
            .iter_mut()
            .map(|h| h.submission())
            .collect::<Vec<_>>();
        let (tx, rx) = mpsc::channel(1);
        let server = tokio::spawn(executor.serve(
            8,
            Duration::from_secs(60),
            rx,
            (Kind::Float, Device::Cpu),
        ));
        let results = tokio::time::timeout(
            Duration::from_secs(2),
            futures::future::join_all(
                guards
                    .iter_mut()
                    .enumerate()
                    .map(|(i, g)| g.execute(Tensor::from_slice(&[i as f32]), all_legal_mask(1))),
            ),
        )
        .await
        .unwrap();
        for (i, result) in results.into_iter().enumerate() {
            assert_eq!(f32::try_from(result.unwrap().0).unwrap(), i as f32);
        }
        drop(guards);
        drop((handles, tx));
        let (_, stats) = server.await.unwrap();
        assert_eq!(stats.requests, 3);
        assert_eq!(stats.padded_requests, 1);
        assert_eq!(stats.physical_batch_size_histogram.get(&4), Some(&1));
    }
}
