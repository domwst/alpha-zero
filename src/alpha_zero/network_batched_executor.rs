use std::{marker::PhantomData, time::Duration};

use futures::{stream::FuturesUnordered, StreamExt};
use tch::{Device, Kind, Tensor};

use tokio::sync::mpsc::{
    channel, unbounded_channel, Receiver, Sender, UnboundedReceiver, UnboundedSender,
};

use crate::alpha_zero::Timer;

use super::AlphaZeroNet;

pub struct NetworkBatchedExecutor<Net: AlphaZeroNet> {
    receiver: UnboundedReceiver<(Tensor, Sender<(Tensor, Tensor)>)>,
    sender: UnboundedSender<(Tensor, Sender<(Tensor, Tensor)>)>,
    nn: Net,
}

pub struct NetworkBatchedExecutorHandle<Net: AlphaZeroNet> {
    task_sender: UnboundedSender<(Tensor, Sender<(Tensor, Tensor)>)>,
    result_sender: Sender<(Tensor, Tensor)>,
    result_receiver: Receiver<(Tensor, Tensor)>,
    _p: PhantomData<Net>,
}

impl<Net: AlphaZeroNet> Clone for NetworkBatchedExecutorHandle<Net> {
    fn clone(&self) -> Self {
        let (tx, rx) = tokio::sync::mpsc::channel(1);
        Self {
            task_sender: self.task_sender.clone(),
            result_sender: tx,
            result_receiver: rx,
            _p: PhantomData,
        }
    }
}

impl<Net: AlphaZeroNet> NetworkBatchedExecutorHandle<Net> {
    pub async fn execute(&mut self, task: Tensor) -> (Tensor, Tensor) {
        self.task_sender
            .send((task, self.result_sender.clone()))
            .unwrap();
        self.result_receiver.recv().await.unwrap()
    }
}

pub enum BatcherCommand {
    SetBatchSize(usize),
}

impl<Net: AlphaZeroNet> NetworkBatchedExecutor<Net> {
    pub fn new(nn: Net) -> Self {
        let (tx, rx) = unbounded_channel();
        Self {
            receiver: rx,
            sender: tx,
            nn,
        }
    }

    pub fn mint_handle(&self) -> NetworkBatchedExecutorHandle<Net> {
        let (tx, rx) = channel(1);
        NetworkBatchedExecutorHandle {
            task_sender: self.sender.clone(),
            result_sender: tx,
            result_receiver: rx,
            _p: PhantomData,
        }
    }

    pub async fn serve(
        self,
        mut max_batch: usize,
        batch_acc_time: Duration,
        mut command_receiver: Receiver<BatcherCommand>,
        (kind, device): (Kind, Device),
    ) -> Net {
        const MAX_PAR_RESPS: usize = 1;

        let NetworkBatchedExecutor {
            mut receiver,
            nn,
            sender,
        } = self;
        drop(sender);
        assert!(max_batch > 0);

        let mut inputs = vec![];
        let mut responses = vec![];
        let mut buf = vec![];

        let mut response_tasks = FuturesUnordered::new();

        let mut acc_time = batch_acc_time;

        let mut invocations = 0;
        let mut total_tensors = 0;

        'main: loop {
            let deadline = tokio::time::sleep(acc_time);
            tokio::pin!(deadline);
            loop {
                let cur_len = buf.len();
                tokio::select! {
                    () = &mut deadline => {
                        // println!("Executor finished accumulating batch due to deadline");
                        break;
                    },
                    p = receiver.recv_many(&mut buf, max_batch - cur_len) => {
                        // println!("Executer received {p} tensors");
                        if p == 0 {
                            // println!("Stopping work");
                            assert!(buf.is_empty());
                            break 'main;
                        }
                    },
                    cmd = command_receiver.recv() => {
                        let cmd = match cmd {
                            Some(v) => v,
                            None => continue,
                        };
                        match cmd {
                            BatcherCommand::SetBatchSize(s) => {
                                println!("Changing batch size to {s}");
                                max_batch = s;
                            },
                        }
                    }
                }
                if buf.len() >= max_batch {
                    // println!("Executor finished accumulating batch due to size");
                    break;
                }
            }

            if buf.len() != max_batch {
                println!(
                    "Batch of size {} (max_batch = {max_batch} expected",
                    buf.len()
                );
            }
            if buf.is_empty() {
                acc_time *= 2;
                continue;
            } else {
                // println!("Batch of size {}", buf.len());
                acc_time = batch_acc_time;
            }

            while let Some((inp, send)) = buf.pop() {
                inputs.push(inp);
                responses.push(send);
            }

            let timer = Timer::new();
            let input = Tensor::stack(&inputs, 0).totype(kind).to(device);
            timer.print_if_greater(Duration::from_secs(1), "Input construction took {t}");
            let (values, policies) = nn.forward_t(&input, false);
            timer.print_if_greater(Duration::from_secs(1), "Input evaluation took {t}");
            let values = values.to(Device::Cpu);
            let policies = policies.to(Device::Cpu);
            timer.print_if_greater(Duration::from_secs(1), "CPU conversion took {t}");
            response_tasks.push(tokio::spawn(async move {
                for (i, resp) in responses.iter().enumerate() {
                    let value = values.get(i as i64);
                    let policy = policies.get(i as i64);
                    resp.send((value, policy)).await.unwrap();
                }
                timer.print_if_greater(Duration::from_secs(1), "Reply took {t}");
            }));

            invocations += 1;
            total_tensors += inputs.len();

            if invocations % 1000 == 0 {
                println!("Invocations: {invocations}, total tensors: {total_tensors}");
            }

            inputs.clear();
            responses = Vec::with_capacity(max_batch);
            while response_tasks.len() > MAX_PAR_RESPS {
                response_tasks.next().await;
            }
        }

        while let Some(_) = response_tasks.next().await {}

        nn
    }
}
