use std::{collections::HashMap, future::Future, hash::Hash, time::Duration};

use pytorch::{
    alpha_zero::{AlphaZeroNet, NetworkBatchedExecutor},
    tictactoe::TicTacToeNet,
};
use tch::{nn, Device, Kind, Tensor};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let vs = nn::VarStore::new(Device::Mps);

    let net = TicTacToeNet::new(&vs.root());
    // let net_cp = net.clone();

    let executor = NetworkBatchedExecutor::new(net);

    let mut handles = vec![];
    for _ in 0..32 {
        let h = tokio::spawn({
            // let net = net_cp.clone();
            let mut handle = executor.mint_handle();
            let device = vs.device();

            async move {
                // let vs = vs2;
                for _ in 0..16 {
                    let input = Tensor::rand([2, 19, 19], (Kind::Float, device));
                    let (exec_val, exec_pol) = handle.execute(input).await;
                    // println!("Worker {i} received response on iter {j}");
                    assert_eq!(exec_val.size(), [0i64; 0]);
                    assert_eq!(exec_pol.size(), [19, 19]);
                    // if !exec_pol.exp().sum(None).allclose(
                    //     &Tensor::ones([], (Kind::Float, device)),
                    //     1e-2,
                    //     1e-2,
                    //     false,
                    // ) {
                    //     println!("Received: {}", exec_pol.exp().to(Device::Cpu));
                    // }
                    // if !exec_val.allclose(&my_val.view([]), 1e-4, 1e-4, false) {
                    //     println!("exec_val = {exec_val}, my_val={my_val}");
                    //     break;
                    // }
                    // if !exec_pol.allclose(&my_pol.view([19, 19]), 1e-4, 1e-4, false) {
                    //     println!("exec_pol = {exec_pol}, my_pol={my_pol}");
                    //     break;
                    // }
                }
                // drop(handle);
            }
        });
        handles.push(h);
    }

    executor.serve(1024, Duration::from_millis(10)).await;

    for h in handles {
        h.await?;
    }

    Ok(())
}
