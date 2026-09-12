//! Replay batching with an unchanged position/symmetry order across backends.
use std::{sync::mpsc, thread};

use anyhow::{Context, Result, ensure};
use rand::{SeedableRng, rngs::SmallRng, seq::SliceRandom};
use serde::{Deserialize, Serialize};
use tch::{Device, Kind, Tensor};

use crate::{
    engine::{PositionCodec, TrainingCodec},
    gomoku::GomokuCodec,
    training_snapshot::{ReplayBuffer, ReplayPosition},
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum ReplayCacheMode {
    #[default]
    None,
    Cpu,
    Device,
}

pub struct TrainingBatch {
    pub states: Tensor,
    pub policies: Tensor,
    pub values: Tensor,
}

impl TrainingBatch {
    pub fn len(&self) -> usize {
        self.values.size()[0] as usize
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn to_device(&self, device: Device) -> Result<Self> {
        // Prefetch workers only prepare CPU tensors. All GPU work is submitted
        // on the consumer's stream. Torch's pinned allocator tracks async copies.
        let non_blocking = self.states.device() == Device::Cpu
            && matches!(device, Device::Cuda(_))
            && self.states.is_pinned_default();
        let copy = |t: &Tensor| t.f_to_device_(device, Kind::Float, non_blocking, false);
        Ok(Self {
            states: copy(&self.states)?,
            policies: copy(&self.policies)?,
            values: copy(&self.values)?,
        })
    }

    fn pin(self, device: Device) -> Result<Self> {
        if !matches!(device, Device::Cuda(_)) {
            return Ok(self);
        }
        Ok(Self {
            states: self.states.f_pin_memory_default()?,
            policies: self.policies.f_pin_memory_default()?,
            values: self.values.f_pin_memory_default()?,
        })
    }

    fn stack(states: &[Tensor], policies: &[Tensor], values: &[f32]) -> Self {
        Self {
            states: Tensor::stack(states, 0).to_kind(Kind::Float),
            policies: Tensor::stack(policies, 0).to_kind(Kind::Float),
            values: Tensor::from_slice(values),
        }
    }

    fn gather(&self, indices: &[i64]) -> Result<Self> {
        let indices = Tensor::f_from_slice(indices)?.f_to(self.states.device())?;
        Ok(Self {
            states: self.states.f_index_select(0, &indices)?,
            policies: self.policies.f_index_select(0, &indices)?,
            values: self.values.f_index_select(0, &indices)?,
        })
    }

    fn shallow_clone(&self) -> Self {
        Self {
            states: self.states.shallow_clone(),
            policies: self.policies.shallow_clone(),
            values: self.values.shallow_clone(),
        }
    }
}

pub struct TrainingBatches<'a> {
    positions: Vec<&'a ReplayPosition>,
    cache: Option<TrainingBatch>,
    mode: ReplayCacheMode,
    device: Device,
    samples: usize,
}

impl<'a> TrainingBatches<'a> {
    pub fn new(replay: &'a ReplayBuffer, mode: ReplayCacheMode, device: Device) -> Result<Self> {
        let positions = replay.iter().flatten().collect::<Vec<_>>();
        let samples = positions
            .len()
            .checked_mul(GomokuCodec::augmentation_count())
            .context("replay size overflow")?;
        ensure!(samples > 0, "replay has no training positions");
        let mut source = Self {
            positions,
            cache: None,
            mode,
            device,
            samples,
        };
        if mode != ReplayCacheMode::None {
            let started = std::time::Instant::now();
            let cache_device = if mode == ReplayCacheMode::Cpu {
                Device::Cpu
            } else {
                device
            };
            tracing::info!(
                ?mode,
                ?cache_device,
                samples,
                bytes = source.cache_bytes()?,
                "encoding replay cache"
            );
            source.cache = Some(source.encode_cache(cache_device).context("building replay cache; use --replay-cache cpu or none if device memory is insufficient")?);
            tracing::info!(
                seconds = started.elapsed().as_secs_f64(),
                "replay cache ready"
            );
        }
        Ok(source)
    }

    pub fn len(&self) -> usize {
        self.samples
    }
    pub fn is_empty(&self) -> bool {
        self.samples == 0
    }

    pub fn cache_bytes(&self) -> Result<usize> {
        self.samples
            .checked_mul((2 * 19 * 19 + 19 * 19 + 1) * 4)
            .context("cache byte size overflow")
    }

    fn encode_cache(&self, device: Device) -> Result<TrainingBatch> {
        let count = i64::try_from(self.samples)?;
        let cache = TrainingBatch {
            states: Tensor::f_empty([count, 2, 19, 19], (Kind::Float, device))?,
            policies: Tensor::f_empty([count, 19, 19], (Kind::Float, device))?,
            values: Tensor::f_empty([count], (Kind::Float, device))?,
        };
        // Bounded staging avoids a second full cache allocation. Encode each
        // position once, then store all eight symmetries in the legacy order.
        let mut offset = 0;
        for chunk in self.positions.chunks(64) {
            let mut states = Vec::new();
            let mut policies = Vec::new();
            let mut values = Vec::new();
            for sample in chunk {
                let state = GomokuCodec::encode_position(&sample.state);
                let policy = GomokuCodec::policy_to_tensor(&sample.policy);
                for symmetry in 0..GomokuCodec::augmentation_count() {
                    let (s, p) = GomokuCodec::augment(&state, &policy, symmetry);
                    states.push(s);
                    policies.push(p);
                    values.push(sample.value);
                }
            }
            let batch = TrainingBatch::stack(&states, &policies, &values);
            let n = batch.len() as i64;
            cache.states.narrow(0, offset, n).f_copy_(&batch.states)?;
            cache
                .policies
                .narrow(0, offset, n)
                .f_copy_(&batch.policies)?;
            cache.values.narrow(0, offset, n).f_copy_(&batch.values)?;
            offset += n;
        }
        Ok(cache)
    }

    fn gather(&self, indices: &[i64]) -> Result<TrainingBatch> {
        if let Some(cache) = &self.cache {
            return cache.gather(indices);
        }
        let augmentations = GomokuCodec::augmentation_count();
        let mut states = Vec::with_capacity(indices.len());
        let mut policies = Vec::with_capacity(indices.len());
        let mut values = Vec::with_capacity(indices.len());
        for &index in indices {
            let index = index as usize;
            let sample = self.positions[index / augmentations];
            let state = GomokuCodec::encode_position(&sample.state);
            let policy = GomokuCodec::policy_to_tensor(&sample.policy);
            let (state, policy) = GomokuCodec::augment(&state, &policy, index % augmentations);
            states.push(state);
            policies.push(policy);
            values.push(sample.value);
        }
        Ok(TrainingBatch::stack(&states, &policies, &values))
    }

    /// None gives validation order; a seed gives the legacy deterministic shuffle.
    pub fn batches(
        &self,
        batch_size: usize,
        seed: Option<u64>,
        prefetch: usize,
    ) -> Result<BatchIter<'_>> {
        ensure!(batch_size > 0, "batch size must be positive");
        ensure!(prefetch <= 16, "prefetch-batches must be at most 16");
        ensure!(
            prefetch == 0 || self.mode == ReplayCacheMode::Cpu,
            "prefetch-batches requires --replay-cache cpu"
        );
        let mut indices = (0..i64::try_from(self.samples)?).collect::<Vec<_>>();
        if let Some(seed) = seed {
            indices.shuffle(&mut SmallRng::seed_from_u64(seed));
        }
        if prefetch == 0 {
            let n = indices.len();
            let iter = (0..n)
                .step_by(batch_size)
                .map(move |start| self.gather(&indices[start..(start + batch_size).min(n)]));
            return Ok(BatchIter::Direct(Box::new(iter)));
        }
        let cache = self.cache.as_ref().unwrap().shallow_clone();
        let device = self.device;
        let (sender, receiver) = mpsc::sync_channel(prefetch);
        let worker = thread::Builder::new()
            .name("replay-prefetch".into())
            .spawn(move || {
                for chunk in indices.chunks(batch_size) {
                    let batch = cache.gather(chunk).and_then(|b| b.pin(device));
                    let failed = batch.is_err();
                    if sender.send(batch).is_err() || failed {
                        break;
                    }
                }
            })?;
        Ok(BatchIter::Prefetched(Prefetched {
            receiver: Some(receiver),
            worker: Some(worker),
        }))
    }
}

pub enum BatchIter<'a> {
    Direct(Box<dyn Iterator<Item = Result<TrainingBatch>> + 'a>),
    Prefetched(Prefetched),
}

impl Iterator for BatchIter<'_> {
    type Item = Result<TrainingBatch>;
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Direct(iter) => iter.next(),
            Self::Prefetched(prefetch) => match prefetch.receiver.as_ref()?.recv() {
                Ok(batch) => Some(batch),
                Err(_) => {
                    prefetch.receiver.take();
                    prefetch
                        .worker
                        .take()?
                        .join()
                        .err()
                        .map(|_| Err(anyhow::anyhow!("replay prefetch worker panicked")))
                }
            },
        }
    }
}

pub struct Prefetched {
    receiver: Option<mpsc::Receiver<Result<TrainingBatch>>>,
    worker: Option<thread::JoinHandle<()>>,
}

impl Drop for Prefetched {
    fn drop(&mut self) {
        // Disconnect before joining: a producer blocked on a full queue must
        // wake up when a caller stops early (benchmark limit or training error).
        self.receiver.take();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        engine::TrainingSample,
        gomoku::{BoardState, CellState, GomokuMove, GomokuPolicy},
    };

    fn fixture() -> ReplayBuffer {
        (0..3)
            .map(|i| {
                (0..3)
                    .map(|j| TrainingSample {
                        state: BoardState::new()
                            .set((2 + i, 3 + j), CellState::X)
                            .set((12, 7), CellState::O),
                        policy: GomokuPolicy::one_hot(GomokuMove::from_xy(i, j + 5)),
                        value: (i * 3 + j) as f32 / 10.0,
                    })
                    .collect()
            })
            .collect()
    }

    fn check_order(device: Device) {
        let replay = fixture();
        for seed in [None, Some(0), Some(20260906)] {
            // Independent oracle: the previous loop's (position, augmentation) shuffle.
            let mut expected = replay
                .iter()
                .flatten()
                .flat_map(|sample| (0..8).map(move |s| (sample, s)))
                .collect::<Vec<_>>();
            if let Some(seed) = seed {
                expected.shuffle(&mut SmallRng::seed_from_u64(seed));
            }
            for (mode, prefetch) in [
                (ReplayCacheMode::None, 0),
                (ReplayCacheMode::Cpu, 0),
                (ReplayCacheMode::Cpu, 2),
                (ReplayCacheMode::Device, 0),
            ] {
                let source = TrainingBatches::new(&replay, mode, device).unwrap();
                let mut count = 0;
                // A non-divisor batch size exercises the partial final batch.
                for (batch, chunk) in source
                    .batches(7, seed, prefetch)
                    .unwrap()
                    .zip(expected.chunks(7))
                {
                    let batch = batch
                        .unwrap()
                        .to_device(device)
                        .unwrap()
                        .to_device(Device::Cpu)
                        .unwrap();
                    assert_eq!(batch.len(), chunk.len());
                    for (i, &(sample, symmetry)) in chunk.iter().enumerate() {
                        let (state, policy) = GomokuCodec::augment(
                            &GomokuCodec::encode_position(&sample.state),
                            &GomokuCodec::policy_to_tensor(&sample.policy),
                            symmetry,
                        );
                        assert!(
                            batch
                                .states
                                .get(i as i64)
                                .equal(&state.to_kind(Kind::Float))
                        );
                        assert!(batch.policies.get(i as i64).equal(&policy));
                        assert_eq!(batch.values.double_value(&[i as i64]), sample.value as f64);
                    }
                    count += batch.len();
                }
                assert_eq!(count, expected.len());
            }
        }
    }

    #[test]
    fn all_batch_modes_preserve_legacy_order_and_targets() {
        check_order(Device::Cpu);
    }

    #[test]
    #[ignore = "requires a CUDA device"]
    fn cuda_cache_and_pinned_prefetch_match_legacy_batches() {
        assert!(tch::Cuda::is_available());
        check_order(Device::Cuda(0));
    }

    #[test]
    fn dropping_prefetch_with_a_full_queue_does_not_hang() {
        let replay = fixture();
        let source = TrainingBatches::new(&replay, ReplayCacheMode::Cpu, Device::Cpu).unwrap();
        for _ in 0..4 {
            let mut batches = source.batches(1, Some(1), 1).unwrap();
            batches.next().unwrap().unwrap();
            drop(batches);
        }
    }

    #[test]
    fn invalid_batch_configuration_is_rejected() {
        assert!(
            TrainingBatches::new(&ReplayBuffer::new(), ReplayCacheMode::None, Device::Cpu).is_err()
        );
        let replay = fixture();
        let source = TrainingBatches::new(&replay, ReplayCacheMode::None, Device::Cpu).unwrap();
        assert!(source.batches(0, None, 0).is_err());
        assert!(source.batches(1, None, 1).is_err());
        assert!(source.batches(1, None, 17).is_err());
    }
}
