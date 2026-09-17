use crate::cli::AnalysisArgs;
use alz::{
    engine::{
        AlphaZeroNet, Game, GameAdapter, MonteCarloTree, PositionCodec, PositionEnvelope,
        PositionEvaluation, PositionEvaluator, RootNoise, masked_policy_probabilities,
    },
    gomoku::{BoardState, GomokuAdapter, GomokuCodec, GomokuModel},
};
use anyhow::{Result, ensure};
use rand::{SeedableRng, rngs::SmallRng};
use serde::Deserialize;
use serde_json::json;
use std::{
    io::{BufRead, Write},
    sync::Mutex,
    time::{Duration, Instant},
};
use tch::{Device, Kind};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    position: PositionEnvelope,
    #[serde(default)]
    simulations: usize,
    #[serde(default)]
    inspect: bool,
    #[serde(default)]
    layers: Vec<String>,
    #[serde(default)]
    stream: bool,
}

struct Evaluator<'a> {
    network: &'a Mutex<GomokuModel>,
    device: Device,
}
impl PositionEvaluator<BoardState> for Evaluator<'_> {
    async fn evaluate(
        &mut self,
        state: &BoardState,
        moves: &[alz::gomoku::GomokuMove],
    ) -> Result<PositionEvaluation> {
        let input = GomokuCodec::encode_position(state)
            .unsqueeze(0)
            .totype(Kind::Float)
            .to(self.device);
        let mask = GomokuCodec::encode_policy_mask(state, moves)?
            .unsqueeze(0)
            .to(self.device);
        let output = tch::no_grad(|| self.network.lock().unwrap().forward_t(&input, false));
        let policy = masked_policy_probabilities(&output.policy_logits, &mask)
            .get(0)
            .to(Device::Cpu);
        Ok(PositionEvaluation {
            value: f32::try_from(output.values.get(0))?,
            legal_policy: GomokuCodec::decode_policy(&policy, moves)?,
        })
    }
}

struct Search<'a> {
    state: BoardState,
    tree: MonteCarloTree<BoardState, Evaluator<'a>>,
    rng: SmallRng,
    completed: usize,
}

async fn analyze<'a>(
    request: Request,
    network: &'a Mutex<GomokuModel>,
    device: Device,
    maximum: usize,
    cached: &mut Option<Search<'a>>,
    mut publish: impl FnMut(serde_json::Value) -> Result<()>,
) -> Result<serde_json::Value> {
    ensure!(
        request.position.game_type == GomokuAdapter::ID,
        "Unsupported game adapter"
    );
    ensure!(
        request.simulations <= maximum,
        "Simulation budget exceeds analysis limit"
    );
    ensure!(
        request.layers.len() <= 64 && request.layers.iter().all(|n| n.len() <= 128),
        "Too many layer selections"
    );
    let state = GomokuAdapter::decode(request.position.state)?;
    let terminal = state.get_state().get_terminal();
    if let Some(previous) = cached.as_mut()
        && previous.state != state
    {
        let successor = previous.tree.root_snapshot().and_then(|root| {
            root.moves
                .iter()
                .enumerate()
                .find(|(_, m)| previous.state.make_move(&m.action) == state)
                .map(|(i, m)| (i, m.action))
        });
        if let Some((i, action)) = successor {
            previous.tree.advance(i, &action, state.clone())?;
            previous.state = state.clone();
            previous.completed = previous.tree.get_total_descends().unwrap_or(0);
        } else {
            *cached = None;
        }
    }

    let search = cached.get_or_insert_with(|| Search {
        state: state.clone(),
        tree: MonteCarloTree::new(
            state.clone(),
            Evaluator { network, device },
            RootNoise::None,
        ),
        rng: SmallRng::seed_from_u64(0),
        completed: 0,
    });
    let started = Instant::now();
    let carried = search.completed;
    // The requested budget counts NEW simulations; visits retained from the
    // previous position's tree are carried on top of it. A tree that has never
    // been expanded still needs one evaluation for its root.
    let expanded = search.tree.root_snapshot().is_some();
    let budget = if terminal.is_some() {
        if expanded { 0 } else { 1 }
    } else {
        request.simulations.max(if expanded { 0 } else { 1 })
    };
    let target = carried + budget;
    let mut updated = Instant::now();
    let response = |search: &Search<'_>, complete: bool, activations: serde_json::Value| {
        let root = search.tree.root_snapshot().unwrap();
        let elapsed = started.elapsed().as_secs_f64();
        json!({"game_type":GomokuAdapter::ID,"game":GomokuAdapter::describe(),"terminal":terminal,
            "network_value":root.network_value,"search_value":root.search_value(),
            "total_visits":root.total_visits,"search":search.tree.search_stats(),"activations":activations,
            "complete":complete,"target_simulations":target,"searched_simulations":search.completed,
            "carried_visits":carried,"elapsed_ms":elapsed * 1000.0,
            "simulations_per_second": (search.completed - carried) as f64 / elapsed.max(1e-9),
            "moves":root.moves.iter().map(|m| {let (row,column)=m.action.to_xy();json!({"row":row,"column":column,"prior":m.prior,"visits":m.visits,"mean_value":m.mean_value()})}).collect::<Vec<_>>() })
    };
    while search.completed < target {
        // Small chunks bound update latency without rebuilding the tree.
        let count = (target - search.completed).min(8);
        search
            .tree
            .do_simulations(count, 1.0, &mut search.rng)
            .await?;
        search.completed += count;
        if request.stream
            && (updated.elapsed() >= Duration::from_millis(150) || search.completed == count)
        {
            publish(response(search, false, json!([])))?;
            updated = Instant::now();
        }
    }
    let tensors = if request.inspect {
        network.lock().unwrap().inspect(
            &GomokuCodec::encode_position(&state)
                .unsqueeze(0)
                .totype(Kind::Float)
                .to(device),
            &request.layers,
        )?
    } else {
        vec![]
    };
    Ok(response(search, true, serde_json::to_value(tensors)?))
}

/// Drain oversized records without allocating the full input or losing framing.
fn read_request_line(input: &mut impl BufRead) -> Result<Option<Result<Vec<u8>>>> {
    const LIMIT: usize = 65536;
    let mut line = Vec::new();
    let mut oversized = false;
    loop {
        let buffer = input.fill_buf()?;
        if buffer.is_empty() {
            if line.is_empty() && !oversized {
                return Ok(None);
            }
            break;
        }
        let newline = buffer.iter().position(|&b| b == b'\n');
        let length = newline.unwrap_or(buffer.len());
        if !oversized {
            if line.len() + length > LIMIT {
                oversized = true;
                line.clear();
            } else {
                line.extend_from_slice(&buffer[..length]);
            }
        }
        input.consume(length + usize::from(newline.is_some()));
        if newline.is_some() {
            break;
        }
    }
    Ok(Some(if oversized {
        Err(anyhow::anyhow!("Analysis request exceeds size limit"))
    } else {
        Ok(line)
    }))
}

pub async fn run(args: AnalysisArgs) -> Result<()> {
    let device = super::common::resolve_device(&args.model.device)?;
    let (_vs, network, snapshot) =
        super::common::load_network(&args.model.checkpoint_dir, args.model.architecture, device)?;
    let identity = serde_json::to_value(snapshot.descriptor())?;
    let network = Mutex::new(network);
    let mut output = std::io::stdout().lock();
    let mut cached = None;
    let mut input = std::io::stdin().lock();
    while let Some(line) = read_request_line(&mut input)? {
        let parsed = line.and_then(|line| Ok(serde_json::from_slice::<Request>(&line)?));
        let invalid_request = parsed.is_err();
        let result = match parsed {
            Ok(request) => {
                analyze(
                    request,
                    &network,
                    device,
                    args.max_simulations,
                    &mut cached,
                    |value| {
                        serde_json::to_writer(
                            &mut output,
                            &json!({"checkpoint":identity,"result":value}),
                        )?;
                        output.write_all(b"\n")?;
                        output.flush()?;
                        Ok(())
                    },
                )
                .await
            }
            Err(e) => Err(e),
        };
        let response = match result {
            Ok(value) => json!({"checkpoint":identity,"result":value}),
            Err(error) => {
                json!({"error":error.to_string(), "error_kind": if invalid_request { "invalid_request" } else { "analysis_failed" }})
            }
        };
        serde_json::to_writer(&mut output, &response)?;
        output.write_all(b"\n")?;
        output.flush()?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn oversized_requests_are_drained_and_leave_the_next_record_readable() {
        let mut bytes = vec![b'x'; 2 * 65536];
        bytes.extend_from_slice(b"\n{}\n");
        let mut input = std::io::BufReader::with_capacity(17, bytes.as_slice());
        assert!(read_request_line(&mut input).unwrap().unwrap().is_err());
        assert_eq!(
            read_request_line(&mut input).unwrap().unwrap().unwrap(),
            b"{}"
        );
        assert!(read_request_line(&mut input).unwrap().is_none());
        let bytes = vec![b'x'; 65536];
        let mut input = std::io::Cursor::new(bytes);
        assert_eq!(
            read_request_line(&mut input)
                .unwrap()
                .unwrap()
                .unwrap()
                .len(),
            65536
        );
        assert!(read_request_line(&mut input).unwrap().is_none());
    }

    #[tokio::test]
    async fn analysis_encodes_float_inputs_and_rejects_invalid_positions() {
        let vs = tch::nn::VarStore::new(Device::Cpu);
        let net = Mutex::new(GomokuModel::new(
            vs.root(),
            &alz::gomoku::ModelSpec::KataGeluBoardMaskValue64x2V1,
        ));
        let request = || Request {
            position: PositionEnvelope {
                game_type: GomokuAdapter::ID.into(),
                state: json!({"cells":vec![0;361]}),
            },
            simulations: 2,
            inspect: true,
            layers: vec!["trunk.block_0.conv1".into()],
            stream: false,
        };
        let mut cache = None;
        let result = analyze(request(), &net, Device::Cpu, 4, &mut cache, |_| Ok(()))
            .await
            .unwrap();
        let repeated = analyze(request(), &net, Device::Cpu, 4, &mut cache, |_| Ok(()))
            .await
            .unwrap();
        // The repeated request budgets 2 new simulations on top of the 2 retained.
        assert_eq!(repeated["carried_visits"], 2);
        assert_eq!(repeated["searched_simulations"], 4);
        let mut larger = request();
        larger.simulations = 4;
        larger.stream = true;
        let mut updates = vec![];
        let extended = analyze(larger, &net, Device::Cpu, 4, &mut cache, |v| {
            updates.push(v);
            Ok(())
        })
        .await
        .unwrap();
        assert_eq!(extended["searched_simulations"], 8);
        assert_eq!(extended["carried_visits"], 4);
        assert_eq!(result["moves"].as_array().unwrap().len(), 361);
        assert!(
            result["activations"]
                .as_array()
                .unwrap()
                .iter()
                .any(|a| a["name"] == "trunk.block_0.conv1"
                    && !a["values"].as_array().unwrap().is_empty())
        );
        let mut invalid = request();
        invalid.position.state["cells"][0] = json!(3);
        assert!(
            analyze(invalid, &net, Device::Cpu, 4, &mut cache, |_| Ok(()))
                .await
                .is_err()
        );
    }
}
