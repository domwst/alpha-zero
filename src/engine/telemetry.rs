//! Low-frequency durable protocol. The hot path only updates atomic counters.
use anyhow::Result;
use serde_json::{Value, json};
use std::{
    fs::{self, OpenOptions},
    io::{BufWriter, Write},
    path::PathBuf,
    sync::{
        Mutex, OnceLock,
        atomic::{AtomicU64, Ordering},
    },
    time::{SystemTime, UNIX_EPOCH},
};

static MOVES: AtomicU64 = AtomicU64::new(0);
static WRITER: Mutex<()> = Mutex::new(());
static DIRECTORY: OnceLock<Option<PathBuf>> = OnceLock::new();

pub fn directory() -> Option<&'static PathBuf> {
    DIRECTORY
        .get_or_init(|| std::env::var_os("ALZ_JOB_DIR").map(PathBuf::from))
        .as_ref()
}
pub fn record_move() {
    MOVES.fetch_add(1, Ordering::Relaxed);
}
pub fn completed_moves() -> u64 {
    MOVES.load(Ordering::Relaxed)
}

pub fn event(kind: &str, payload: Value) -> Result<()> {
    let Some(dir) = directory() else {
        return Ok(());
    };
    let _guard = WRITER.lock().unwrap();
    let mut file = BufWriter::new(
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(dir.join("events.jsonl"))?,
    );
    serde_json::to_writer(
        &mut file,
        &json!({"schema_version":1,"time":SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs_f64(),"kind":kind,"payload":payload}),
    )?;
    file.write_all(b"\n")?;
    file.flush()?;
    file.get_ref().sync_data()?;
    Ok(())
}

/// Only acknowledge a pause once the caller has committed a recovery boundary.
pub fn stop_at_boundary(epoch_complete: bool) -> Result<bool> {
    let Some(dir) = directory() else {
        return Ok(false);
    };
    let path = dir.join("control.json");
    if !path.exists() {
        return Ok(false);
    }
    let request: Value = serde_json::from_reader(std::fs::File::open(path)?)?;
    if request["mode"] == "epoch" && !epoch_complete {
        return Ok(false);
    }
    let ack = json!({"reason":"requested stop at a committed boundary", "command_id":request["command_id"], "epoch_complete":epoch_complete});
    let temporary = dir.join("stopped.json.tmp");
    let mut file = std::fs::File::create(&temporary)?;
    serde_json::to_writer(&mut file, &ack)?;
    file.sync_all()?;
    fs::rename(temporary, dir.join("stopped.json"))?;
    event("stopped", ack)?;
    Ok(true)
}

#[derive(Clone, Default, serde::Serialize)]
pub struct GameProgress {
    pub game_id: usize,
    pub move_sequence: usize,
    pub network_value: Option<f32>,
    pub search_value: Option<f32>,
    pub expanded_nodes: u64,
    pub allocated_nodes: u64,
}

tokio::task_local! {
    static GAME: tokio::sync::watch::Sender<GameProgress>;
}

pub fn game_channel(
    game_id: usize,
) -> (
    tokio::sync::watch::Sender<GameProgress>,
    tokio::sync::watch::Receiver<GameProgress>,
) {
    tokio::sync::watch::channel(GameProgress {
        game_id,
        ..Default::default()
    })
}

pub async fn in_game<F: std::future::Future>(
    sender: tokio::sync::watch::Sender<GameProgress>,
    future: F,
) -> F::Output {
    GAME.scope(sender, future).await
}

pub fn move_made(sequence: usize, decision: &super::MoveDecision) {
    record_move();
    let _ = GAME.try_with(|sender| {
        sender.send_modify(|progress| {
            progress.move_sequence = sequence;
            progress.network_value = decision.diagnostics.value_estimate;
            progress.search_value = decision.diagnostics.search_value;
            if let Some(search) = &decision.diagnostics.search {
                progress.expanded_nodes = search.expanded_by_depth.iter().sum();
                progress.allocated_nodes = search.allocated_by_depth.iter().sum();
            }
        });
    });
}
