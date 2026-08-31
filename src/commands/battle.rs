use std::time::Duration;

use alz::{
    engine::{ExecutorScope, NetworkPositionEvaluator, Seat, do_battle},
    gomoku::{BoardState, GomokuCodec, GomokuResNet},
};
use anyhow::{Result, ensure};
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
    let first_evaluator = NetworkPositionEvaluator::<GomokuResNet, GomokuCodec>::new(
        first_executor.evaluator_handle(),
    );
    let second_evaluator = NetworkPositionEvaluator::<GomokuResNet, GomokuCodec>::new(
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

    tracing::info!(first_epoch, second_epoch, "battle snapshots loaded");
    for (ply, turn) in record.plies.iter().enumerate() {
        let (row, column) = turn.action.to_xy();
        tracing::info!(
            ply = ply + 1,
            actor = ?turn.actor,
            row = row + 1,
            column = column + 1,
            "battle move"
        );
    }
    match record.value_for(Seat::First) {
        value if value > 0.0 => tracing::info!(winner = "first", "battle complete"),
        value if value < 0.0 => tracing::info!(winner = "second", "battle complete"),
        _ => tracing::info!(winner = "draw", "battle complete"),
    }
    Ok(())
}
