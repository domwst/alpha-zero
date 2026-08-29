use std::{marker::PhantomData, time::Duration};

use anyhow::{anyhow, Context, Result};
use tch::{Device, Kind, Tensor};
use tokio::sync::{mpsc, oneshot};

use crate::util::Timer;

use super::AlphaZeroNet;

struct InferenceRequest {
    input: Tensor,
    response: oneshot::Sender<(Tensor, Tensor)>,
}

pub struct NetworkBatchedExecutor<Net: AlphaZeroNet> {
    receiver: mpsc::Receiver<InferenceRequest>,
    sender: mpsc::Sender<InferenceRequest>,
    nn: Net,
}

pub struct NetworkBatchedExecutorHandle<Net: AlphaZeroNet> {
    task_sender: mpsc::Sender<InferenceRequest>,
    _net: PhantomData<fn() -> Net>,
}

impl<Net: AlphaZeroNet> Clone for NetworkBatchedExecutorHandle<Net> {
    fn clone(&self) -> Self {
        Self {
            task_sender: self.task_sender.clone(),
            _net: PhantomData,
        }
    }
}

impl<Net: AlphaZeroNet> NetworkBatchedExecutorHandle<Net> {
    pub async fn execute(&self, input: Tensor) -> Result<(Tensor, Tensor)> {
        let (response, result) = oneshot::channel();
        self.task_sender
            .send(InferenceRequest { input, response })
            .await
            .map_err(|_| anyhow!("network evaluator stopped before accepting the request"))?;
        result.await.context("network evaluator stopped")
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
            receiver,
            sender,
            nn,
        }
    }

    pub fn mint_handle(&self) -> NetworkBatchedExecutorHandle<Net> {
        NetworkBatchedExecutorHandle {
            task_sender: self.sender.clone(),
            _net: PhantomData,
        }
    }

    pub async fn serve(
        self,
        mut max_batch: usize,
        batch_acc_time: Duration,
        mut command_receiver: mpsc::Receiver<BatcherCommand>,
        (kind, device): (Kind, Device),
    ) -> Net {
        let NetworkBatchedExecutor {
            mut receiver,
            nn,
            sender,
        } = self;
        drop(sender);
        assert!(max_batch > 0);

        let mut pending = Vec::with_capacity(max_batch);
        let mut input_closed = false;
        let mut commands_open = true;
        let mut invocations = 0;
        let mut total_tensors = 0;

        loop {
            while pending.is_empty() && !input_closed {
                tokio::select! {
                    request = receiver.recv() => match request {
                        Some(request) => pending.push(request),
                        None => input_closed = true,
                    },
                    command = command_receiver.recv(), if commands_open => match command {
                        Some(BatcherCommand::SetBatchSize(size)) if size > 0 => {
                            println!("Changing batch size to {size}");
                            max_batch = size;
                        }
                        Some(BatcherCommand::SetBatchSize(_)) => {
                            eprintln!("Ignoring zero network batch size");
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
            while pending.len() < max_batch && !input_closed {
                tokio::select! {
                    () = &mut deadline => break,
                    request = receiver.recv() => match request {
                        Some(request) => pending.push(request),
                        None => input_closed = true,
                    },
                    command = command_receiver.recv(), if commands_open => match command {
                        Some(BatcherCommand::SetBatchSize(size)) if size > 0 => {
                            println!("Changing batch size to {size}");
                            max_batch = size;
                        }
                        Some(BatcherCommand::SetBatchSize(_)) => {
                            eprintln!("Ignoring zero network batch size");
                        }
                        None => commands_open = false,
                    },
                }
            }

            pending.retain(|request| !request.response.is_closed());
            if pending.is_empty() {
                continue;
            }
            let batch_len = pending.len().min(max_batch);
            if batch_len != max_batch {
                println!("Batch of size {batch_len} (max_batch = {max_batch})");
            }

            let (inputs, responses): (Vec<_>, Vec<_>) = pending
                .drain(..batch_len)
                .map(|request| (request.input, request.response))
                .unzip();

            let timer = Timer::new();
            let input = Tensor::stack(&inputs, 0).totype(kind).to(device);
            timer.print_if_greater(Duration::from_secs(1), "Input construction took {t}");
            let (values, policies) = tch::no_grad(|| nn.forward_t(&input, false));
            timer.print_if_greater(Duration::from_secs(1), "Input evaluation took {t}");
            let values = values.to(Device::Cpu);
            let policies = policies.to(Device::Cpu);
            timer.print_if_greater(Duration::from_secs(1), "CPU conversion took {t}");

            for (index, response) in responses.into_iter().enumerate() {
                let value = values.get(index as i64);
                let policy = policies.get(index as i64);
                let _ = response.send((value, policy));
            }

            invocations += 1;
            total_tensors += inputs.len();
            if invocations % 1000 == 0 {
                println!("Invocations: {invocations}, total tensors: {total_tensors}");
            }
        }

        nn
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;

    struct EchoNet;

    impl AlphaZeroNet for EchoNet {
        fn forward_t(&self, input: &Tensor, _is_training: bool) -> (Tensor, Tensor) {
            let batch = input.size()[0];
            (
                input.view([batch]),
                Tensor::zeros([batch, 1], (Kind::Float, Device::Cpu)),
            )
        }
    }

    struct RecordingNet(Arc<Mutex<Vec<i64>>>);

    impl AlphaZeroNet for RecordingNet {
        fn forward_t(&self, input: &Tensor, _is_training: bool) -> (Tensor, Tensor) {
            let batch = input.size()[0];
            self.0.lock().unwrap().push(batch);
            (
                input.view([batch]),
                Tensor::zeros([batch, 1], (Kind::Float, Device::Cpu)),
            )
        }
    }

    #[tokio::test]
    async fn cancelled_request_does_not_change_the_next_response() {
        let executor = NetworkBatchedExecutor::new(EchoNet, 4);
        let handle = executor.mint_handle();
        let (command_sender, command_receiver) = mpsc::channel(1);
        let server = tokio::spawn(executor.serve(
            2,
            Duration::from_millis(50),
            command_receiver,
            (Kind::Float, Device::Cpu),
        ));

        {
            let first = handle.execute(Tensor::from_slice(&[1.0f32]));
            tokio::pin!(first);
            tokio::select! {
                _ = tokio::time::sleep(Duration::from_millis(1)) => {}
                result = &mut first => panic!("request unexpectedly completed: {result:?}"),
            }
        }

        let (value, _) = handle.execute(Tensor::from_slice(&[2.0f32])).await.unwrap();
        assert_eq!(f32::try_from(value).unwrap(), 2.0);

        drop((handle, command_sender));
        server.await.unwrap();
    }

    #[tokio::test]
    async fn stopped_executor_returns_an_error() {
        let executor = NetworkBatchedExecutor::new(EchoNet, 1);
        let handle = executor.mint_handle();
        drop(executor);

        let result = handle.execute(Tensor::from_slice(&[1.0f32])).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn shrinking_batch_size_limits_an_accumulated_batch() {
        let observed_batches = Arc::new(Mutex::new(Vec::new()));
        let executor = NetworkBatchedExecutor::new(RecordingNet(observed_batches.clone()), 8);
        let handle = executor.mint_handle();
        let (command_sender, command_receiver) = mpsc::channel(1);
        let server = tokio::spawn(executor.serve(
            4,
            Duration::from_millis(50),
            command_receiver,
            (Kind::Float, Device::Cpu),
        ));

        let requests = (0..3)
            .map(|value| {
                let handle = handle.clone();
                tokio::spawn(async move {
                    handle
                        .execute(Tensor::from_slice(&[value as f32]))
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
        drop((handle, command_sender));
        server.await.unwrap();

        let batches = observed_batches.lock().unwrap();
        assert_eq!(batches.iter().sum::<i64>(), 3);
        assert!(batches.iter().all(|size| *size <= 2));
    }
}
