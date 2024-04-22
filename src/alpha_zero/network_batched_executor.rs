use std::{marker::PhantomData, time::Duration};

use futures::{stream::FuturesUnordered, StreamExt};
use tch::{Device, Kind, Tensor};

use tokio::sync::mpsc::{
    channel, unbounded_channel, Receiver, Sender, UnboundedReceiver, UnboundedSender,
};

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

impl<Net: AlphaZeroNet> NetworkBatchedExecutorHandle<Net> {
    pub async fn execute(&mut self, task: Tensor) -> (Tensor, Tensor) {
        self.task_sender
            .send((task, self.result_sender.clone()))
            .unwrap();
        self.result_receiver.recv().await.unwrap()
    }
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
        max_batch: usize,
        batch_acc_time: Duration,
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
                }
                if buf.len() >= max_batch {
                    // println!("Executor finished accumulating batch due to size");
                    assert_eq!(buf.len(), max_batch);
                    break;
                }
            }

            if buf.is_empty() {
                println!("Empty buf");
                acc_time *= 2;
                continue;
            } else {
                if buf.len() != max_batch {
                    println!("Batch of size {}", buf.len());
                }
                // println!("Batch of size {}", buf.len());
                acc_time = batch_acc_time;
            }

            while let Some((inp, send)) = buf.pop() {
                inputs.push(inp);
                responses.push(send);
            }

            // let start = Instant::now();
            let input = Tensor::stack(&inputs, 0).totype(kind).to(device);
            // println!("Constructed input in {:?}", Instant::now() - start);
            let (values, policies) = nn.forward_t(&input, false);
            // println!("Evaluated input in {:?}", Instant::now() - start);
            let values = values.to(Device::Cpu);
            let policies = policies.to(Device::Cpu);
            // println!("Moved everything to GPU in {:?}", Instant::now() - start);
            response_tasks.push(tokio::spawn(async move {
                for (i, resp) in responses.iter().enumerate() {
                    let value = values.get(i as i64);
                    let policy = policies.get(i as i64);
                    resp.send((value, policy)).await.unwrap();
                }
                // println!("Sent responses back in {:?}", Instant::now() - start);
            }));
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
