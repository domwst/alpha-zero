use std::{
    future::Future,
    io::{self, Write},
    str::FromStr,
    time::Duration,
};

use alz::{
    engine::{
        Agent, ExecutorScope, MctsAgent, MoveDecision, NetworkPositionEvaluator, PolicyAgent,
        RootNoise, Seat, Shared, Turn, Versus, run_match,
    },
    gomoku::{BoardState, CellState, GomokuCodec, GomokuMove, GomokuResNet},
};
use anyhow::{Result, ensure};
use rand::{SeedableRng, rngs::SmallRng};
use tch::Kind;

use crate::cli::{HumanArgs, HumanSeat, PolicyArgs};

use super::common::{load_network, resolve_device};

struct HumanAgent;

impl Agent<BoardState> for HumanAgent {
    fn select_move<'a>(
        &'a mut self,
        turn: Turn<'a, BoardState>,
    ) -> impl Future<Output = Result<MoveDecision>> + Send + 'a {
        let state = turn.state.clone();
        let legal_moves = turn.legal_moves.to_vec();
        async move {
            tokio::task::spawn_blocking(move || {
                print_state(&state);
                print!("Enter row and column: ");
                io::stdout().flush()?;

                let mut input = String::new();
                io::stdin().read_line(&mut input)?;
                let [row, column] = input
                    .split_whitespace()
                    .map(FromStr::from_str)
                    .collect::<Result<Vec<usize>, _>>()?
                    .try_into()
                    .map_err(|_| anyhow::anyhow!("expected a row and column"))?;
                ensure!(row > 0 && column > 0, "row and column are one-based");

                let selected = GomokuMove::from_xy(row - 1, column - 1);
                let move_index = legal_moves
                    .iter()
                    .position(|&candidate| candidate == selected)
                    .ok_or_else(|| anyhow::anyhow!("selected move is not legal"))?;
                Ok(MoveDecision::new(move_index))
            })
            .await?
        }
    }
}

pub async fn run_human(args: HumanArgs) -> Result<()> {
    ensure!(
        args.simulations > 0,
        "simulations must be greater than zero"
    );
    ensure!(
        args.c_puct.is_finite() && args.c_puct >= 0.0,
        "c-puct must be finite and non-negative"
    );
    validate_temperature(args.temperature)?;

    let device = resolve_device(&args.model.device)?;
    let (var_store, network, _) = load_network(&args.model.checkpoint_dir, device)?;
    let executor = ExecutorScope::<(), _>::new(
        network,
        1,
        1,
        Duration::from_millis(1),
        (Kind::Float, var_store.device()),
    );
    let evaluator = NetworkPositionEvaluator::<GomokuResNet, GomokuCodec>::new(
        executor.evaluator_handle(),
    );
    let start = BoardState::new();
    let network_agent = MctsAgent::new(
        start.clone(),
        evaluator,
        RootNoise::None,
        args.simulations,
        args.c_puct,
        SmallRng::from_rng(&mut rand::rng()),
        move |_| args.temperature,
    );

    let result = match args.human_seat {
        HumanSeat::First => {
            let mut controller = Versus::new(HumanAgent, network_agent);
            run_match(start, &mut controller).await
        }
        HumanSeat::Second => {
            let mut controller = Versus::new(network_agent, HumanAgent);
            run_match(start, &mut controller).await
        }
    };
    let _network = executor.join().await;
    let record = result?;

    print_state(&record.terminal_state);
    let human_seat = match args.human_seat {
        HumanSeat::First => Seat::First,
        HumanSeat::Second => Seat::Second,
    };
    match record.value_for(human_seat) {
        value if value > 0.0 => println!("You won"),
        value if value < 0.0 => println!("You lost"),
        _ => println!("Draw"),
    }
    Ok(())
}

pub async fn run_policy(args: PolicyArgs) -> Result<()> {
    validate_temperature(args.temperature)?;

    let device = resolve_device(&args.model.device)?;
    let (var_store, network, _) = load_network(&args.model.checkpoint_dir, device)?;
    let executor = ExecutorScope::<(), _>::new(
        network,
        1,
        1,
        Duration::from_millis(1),
        (Kind::Float, var_store.device()),
    );
    let evaluator = NetworkPositionEvaluator::<GomokuResNet, GomokuCodec>::new(
        executor.evaluator_handle(),
    );
    let agent = PolicyAgent::<BoardState, _, _, _>::new(
        evaluator,
        SmallRng::from_rng(&mut rand::rng()),
        move |_| args.temperature,
    );
    let mut controller = Shared::new(agent);
    let result = run_match(BoardState::new(), &mut controller).await;
    drop(controller);
    let _network = executor.join().await;
    let record = result?;

    for (ply, turn) in record.plies.iter().enumerate() {
        let (row, column) = turn.action.to_xy();
        if let Some(value) = turn.decision.diagnostics.value_estimate {
            tracing::info!(
                ply = ply + 1,
                actor = ?turn.actor,
                row = row + 1,
                column = column + 1,
                value_estimate = value,
                "policy move"
            );
        }
    }
    print_state(&record.terminal_state);
    print_outcome(record.value_for(Seat::First));
    Ok(())
}

fn validate_temperature(temperature: f32) -> Result<()> {
    ensure!(
        temperature.is_finite() && temperature >= 0.0,
        "temperature must be finite and non-negative"
    );
    Ok(())
}

fn print_state(state: &BoardState) {
    for row in 0..BoardState::N {
        for column in 0..BoardState::N {
            let symbol = match state[(row, column)] {
                CellState::X => 'X',
                CellState::O => 'O',
                CellState::Empty => ' ',
            };
            print!("{symbol}");
        }
        println!();
    }
}

fn print_outcome(first_value: f32) {
    match first_value {
        value if value > 0.0 => println!("First seat won"),
        value if value < 0.0 => println!("Second seat won"),
        _ => println!("Draw"),
    }
}
