use std::time::Duration;

use alz::{
    alpha_zero::{do_battle, ExecutorScope, NetworkPositionEvaluator, Seat},
    tictactoe::{BoardState, TicTacToeCodec, TicTacToeResNet},
};
use anyhow::{ensure, Result};
use tch::Kind;

use crate::cli::BattleArgs;

use super::common::{load_network, resolve_device};

pub async fn run(args: BattleArgs) -> Result<()> {
    ensure!(
        args.simulations > 0,
        "simulations must be greater than zero"
    );
    ensure!(
        args.c_puct.is_finite() && args.c_puct >= 0.0,
        "c-puct must be finite and non-negative"
    );
    ensure!(
        args.temperature.is_finite() && args.temperature >= 0.0,
        "temperature must be finite and non-negative"
    );

    let device = resolve_device(&args.device)?;
    let (first_var_store, first_network, first_epoch) =
        load_network(&args.first_checkpoint_dir, device)?;
    let (second_var_store, second_network, second_epoch) =
        load_network(&args.second_checkpoint_dir, device)?;

    let first_executor = ExecutorScope::<(), _>::new(
        first_network,
        1,
        1,
        Duration::from_millis(1),
        (Kind::Float, first_var_store.device()),
    );
    let second_executor = ExecutorScope::<(), _>::new(
        second_network,
        1,
        1,
        Duration::from_millis(1),
        (Kind::Float, second_var_store.device()),
    );
    let first_evaluator = NetworkPositionEvaluator::<TicTacToeResNet, TicTacToeCodec>::new(
        first_executor.evaluator_handle(),
    );
    let second_evaluator = NetworkPositionEvaluator::<TicTacToeResNet, TicTacToeCodec>::new(
        second_executor.evaluator_handle(),
    );

    let result = do_battle(
        BoardState::new(),
        args.simulations,
        args.c_puct,
        move |_| args.temperature,
        first_evaluator,
        second_evaluator,
    )
    .await;
    let (_first_network, _second_network) =
        tokio::join!(first_executor.join(), second_executor.join());
    let record = result?;

    println!("First snapshot epoch: {first_epoch}");
    println!("Second snapshot epoch: {second_epoch}");
    for (ply, turn) in record.plies.iter().enumerate() {
        let (row, column) = turn.action.to_xy();
        println!(
            "Ply {}: {:?} played {} {}",
            ply + 1,
            turn.actor,
            row + 1,
            column + 1
        );
    }
    match record.value_for(Seat::First) {
        value if value > 0.0 => println!("First snapshot won"),
        value if value < 0.0 => println!("Second snapshot won"),
        _ => println!("Draw"),
    }
    Ok(())
}
