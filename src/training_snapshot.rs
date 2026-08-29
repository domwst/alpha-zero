use std::{
    cmp::Reverse,
    collections::VecDeque,
    fs::{self, File},
    io::{BufReader, BufWriter, Write},
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use alz::{
    alpha_zero::TrainingSample,
    tictactoe::{BoardState, TicTacToePolicy, ACTION_SCHEMA},
};
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use tch::nn::{Optimizer, VarStore};

const FORMAT_VERSION: u32 = 1;
const MODEL_FILE: &str = "model.safetensors";
const OPTIMIZER_FILE: &str = "optimizer.ot";
const REPLAY_FILE: &str = "replay.bin.zst";
const METADATA_FILE: &str = "metadata.json";

pub type ReplayPosition = TrainingSample<BoardState, TicTacToePolicy>;
pub type ReplayGame = Vec<ReplayPosition>;
pub type ReplayBuffer = VecDeque<ReplayGame>;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct SnapshotMetadata {
    format_version: u32,
    action_schema: String,
    epoch: usize,
    replay_games: usize,
    replay_positions: usize,
}

#[derive(Clone, Debug)]
pub struct TrainingSnapshot {
    path: PathBuf,
    metadata: SnapshotMetadata,
}

impl TrainingSnapshot {
    pub fn epoch(&self) -> usize {
        self.metadata.epoch
    }

    pub fn load_model(&self, var_store: &mut VarStore) -> Result<()> {
        let path = self.path.join(MODEL_FILE);
        var_store
            .load(&path)
            .with_context(|| format!("loading {}", path.display()))
    }
}

fn snapshot_dir(root: &Path, epoch: usize) -> PathBuf {
    root.join(format!("{epoch:08}"))
}

fn read_metadata(path: &Path) -> Result<SnapshotMetadata> {
    serde_json::from_reader(BufReader::new(
        File::open(path).with_context(|| format!("opening {}", path.display()))?,
    ))
    .with_context(|| format!("reading {}", path.display()))
}

fn is_complete_snapshot(path: &Path) -> bool {
    [MODEL_FILE, OPTIMIZER_FILE, REPLAY_FILE, METADATA_FILE]
        .into_iter()
        .all(|name| path.join(name).is_file())
}

pub fn find_latest_snapshot(root: &Path) -> Result<Option<TrainingSnapshot>> {
    if !root.exists() {
        return Ok(None);
    }

    let mut candidates = Vec::new();
    for entry in fs::read_dir(root).with_context(|| format!("reading {}", root.display()))? {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let Some(epoch) = entry
            .file_name()
            .to_str()
            .and_then(|name| name.parse::<usize>().ok())
        else {
            continue;
        };

        let path = entry.path();
        if !is_complete_snapshot(&path) {
            continue;
        }
        candidates.push((epoch, path));
    }

    candidates.sort_unstable_by_key(|(epoch, _)| Reverse(*epoch));
    if let Some((epoch, path)) = candidates.into_iter().next() {
        return read_snapshot(path, epoch).map(Some);
    }
    Ok(None)
}

pub fn resolve_snapshot(path: &Path) -> Result<Option<TrainingSnapshot>> {
    if !is_complete_snapshot(path) {
        return find_latest_snapshot(path);
    }

    let epoch = path
        .file_name()
        .and_then(|name| name.to_str())
        .and_then(|name| name.parse::<usize>().ok())
        .with_context(|| format!("snapshot directory {} has no numeric epoch", path.display()))?;
    read_snapshot(path.to_owned(), epoch).map(Some)
}

fn read_snapshot(path: PathBuf, epoch: usize) -> Result<TrainingSnapshot> {
    let metadata = read_metadata(&path.join(METADATA_FILE))?;
    if metadata.format_version != FORMAT_VERSION {
        bail!(
            "unsupported snapshot version {} in {}",
            metadata.format_version,
            path.display()
        );
    }
    if metadata.action_schema != ACTION_SCHEMA {
        bail!(
            "unsupported action schema {} in {}",
            metadata.action_schema,
            path.display()
        );
    }
    if metadata.epoch != epoch {
        bail!(
            "snapshot directory epoch {epoch} does not match metadata epoch {}",
            metadata.epoch
        );
    }
    Ok(TrainingSnapshot { path, metadata })
}

fn save_replay(path: &Path, replay: &ReplayBuffer) -> Result<()> {
    let writer =
        BufWriter::new(File::create(path).with_context(|| format!("creating {}", path.display()))?);
    let mut encoder = zstd::stream::write::Encoder::new(writer, 3)
        .with_context(|| format!("creating zstd stream for {}", path.display()))?;
    bincode::serialize_into(&mut encoder, replay)
        .with_context(|| format!("serializing {}", path.display()))?;
    let mut writer = encoder
        .finish()
        .with_context(|| format!("finishing {}", path.display()))?;
    writer.flush()?;
    writer.get_ref().sync_all()?;
    Ok(())
}

fn load_replay(path: &Path) -> Result<ReplayBuffer> {
    let reader =
        BufReader::new(File::open(path).with_context(|| format!("opening {}", path.display()))?);
    let decoder = zstd::stream::read::Decoder::new(reader)
        .with_context(|| format!("opening zstd stream for {}", path.display()))?;
    bincode::deserialize_from(decoder).with_context(|| format!("deserializing {}", path.display()))
}

fn validate_replay(replay: &ReplayBuffer) -> Result<()> {
    for (game_index, game) in replay.iter().enumerate() {
        for (position_index, sample) in game.iter().enumerate() {
            sample.policy.validate_for(&sample.state).with_context(|| {
                format!("validating replay game {game_index}, position {position_index}")
            })?;
            if !sample.value.is_finite() || !(-1.0..=1.0).contains(&sample.value) {
                bail!("replay game {game_index}, position {position_index} has an invalid value");
            }
        }
    }
    Ok(())
}

fn write_metadata(path: &Path, metadata: &SnapshotMetadata) -> Result<()> {
    let mut writer =
        BufWriter::new(File::create(path).with_context(|| format!("creating {}", path.display()))?);
    serde_json::to_writer_pretty(&mut writer, metadata)?;
    writeln!(writer)?;
    writer.flush()?;
    writer.get_ref().sync_all()?;
    Ok(())
}

fn sync_file(path: &Path) -> Result<()> {
    File::open(path)
        .with_context(|| format!("opening {} for sync", path.display()))?
        .sync_all()
        .with_context(|| format!("syncing {}", path.display()))
}

#[cfg(unix)]
fn sync_directory(path: &Path) -> Result<()> {
    File::open(path)
        .with_context(|| format!("opening {} for sync", path.display()))?
        .sync_all()
        .with_context(|| format!("syncing {}", path.display()))
}

#[cfg(not(unix))]
fn sync_directory(_path: &Path) -> Result<()> {
    Ok(())
}

pub fn save_training_snapshot(
    root: &Path,
    epoch: usize,
    var_store: &VarStore,
    optimizer: &Optimizer,
    replay: &ReplayBuffer,
) -> Result<()> {
    fs::create_dir_all(root).with_context(|| format!("creating {}", root.display()))?;

    let final_path = snapshot_dir(root, epoch);
    if final_path.exists() {
        bail!("snapshot {} already exists", final_path.display());
    }

    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let pending_path = root.join(format!(".{epoch:08}.tmp-{}-{unique}", std::process::id()));
    fs::create_dir(&pending_path)
        .with_context(|| format!("creating {}", pending_path.display()))?;

    let result = (|| -> Result<()> {
        let model_path = pending_path.join(MODEL_FILE);
        var_store
            .save(&model_path)
            .with_context(|| format!("saving {}", model_path.display()))?;
        sync_file(&model_path)?;

        let optimizer_path = pending_path.join(OPTIMIZER_FILE);
        optimizer
            .save(&optimizer_path)
            .with_context(|| format!("saving {}", optimizer_path.display()))?;
        sync_file(&optimizer_path)?;

        save_replay(&pending_path.join(REPLAY_FILE), replay)?;
        let metadata = SnapshotMetadata {
            format_version: FORMAT_VERSION,
            action_schema: ACTION_SCHEMA.to_owned(),
            epoch,
            replay_games: replay.len(),
            replay_positions: replay.iter().map(Vec::len).sum(),
        };
        write_metadata(&pending_path.join(METADATA_FILE), &metadata)?;
        sync_directory(&pending_path)?;
        Ok(())
    })();

    if let Err(error) = result {
        let _ = fs::remove_dir_all(&pending_path);
        return Err(error);
    }

    if let Err(error) = fs::rename(&pending_path, &final_path) {
        let _ = fs::remove_dir_all(&pending_path);
        return Err(error).with_context(|| {
            format!(
                "publishing snapshot {} as {}",
                pending_path.display(),
                final_path.display()
            )
        });
    }
    sync_directory(root)?;
    Ok(())
}

pub fn load_training_snapshot(
    snapshot: &TrainingSnapshot,
    var_store: &mut VarStore,
    optimizer: &mut Optimizer,
) -> Result<ReplayBuffer> {
    let replay = load_replay(&snapshot.path.join(REPLAY_FILE))?;
    validate_replay(&replay)?;
    let replay_positions = replay.iter().map(Vec::len).sum::<usize>();
    if replay.len() != snapshot.metadata.replay_games
        || replay_positions != snapshot.metadata.replay_positions
    {
        bail!("replay metadata does not match {}", snapshot.path.display());
    }

    snapshot.load_model(var_store)?;
    let optimizer_path = snapshot.path.join(OPTIMIZER_FILE);
    optimizer
        .load(&optimizer_path)
        .with_context(|| format!("loading {}", optimizer_path.display()))?;
    Ok(replay)
}

#[cfg(test)]
mod tests {
    use std::{
        sync::atomic::{AtomicU64, Ordering},
        time::{SystemTime, UNIX_EPOCH},
    };

    use tch::{
        nn::{self, Module, OptimizerConfig},
        Device, Kind, Tensor,
    };

    use alz::tictactoe::TicTacToeMove;

    use super::*;

    static NEXT_TEMP_ID: AtomicU64 = AtomicU64::new(0);

    fn temp_dir() -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let id = NEXT_TEMP_ID.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "alz-training-snapshot-{}-{unique}-{id}",
            std::process::id()
        ))
    }

    fn write_fake_snapshot(root: &Path, epoch: usize, complete: bool) {
        let path = snapshot_dir(root, epoch);
        fs::create_dir_all(&path).unwrap();
        for name in [MODEL_FILE, REPLAY_FILE, METADATA_FILE] {
            fs::write(path.join(name), b"").unwrap();
        }
        if complete {
            fs::write(path.join(OPTIMIZER_FILE), b"").unwrap();
        }
        write_metadata(
            &path.join(METADATA_FILE),
            &SnapshotMetadata {
                format_version: FORMAT_VERSION,
                action_schema: ACTION_SCHEMA.to_owned(),
                epoch,
                replay_games: 0,
                replay_positions: 0,
            },
        )
        .unwrap();
    }

    #[test]
    fn discovery_uses_latest_complete_snapshot() {
        let root = temp_dir();
        fs::create_dir(&root).unwrap();
        write_fake_snapshot(&root, 1, true);
        write_fake_snapshot(&root, 20, false);
        write_fake_snapshot(&root, 7, true);

        let snapshot = find_latest_snapshot(&root).unwrap().unwrap();
        assert_eq!(snapshot.epoch(), 7);

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn resolution_accepts_an_exact_snapshot() {
        let root = temp_dir();
        fs::create_dir(&root).unwrap();
        write_fake_snapshot(&root, 1, true);
        write_fake_snapshot(&root, 7, true);

        let latest = resolve_snapshot(&root).unwrap().unwrap();
        assert_eq!(latest.epoch(), 7);
        let snapshot = resolve_snapshot(&snapshot_dir(&root, 1)).unwrap().unwrap();
        assert_eq!(snapshot.epoch(), 1);

        fs::remove_dir_all(root).unwrap();
    }

    fn training_step(model: &nn::Linear, optimizer: &mut nn::Optimizer, device: Device) {
        let input = Tensor::from_slice(&[1.0f32, -2.0])
            .view([1, 2])
            .to_device(device);
        let loss = model.forward(&input).square().mean(Kind::Float);
        optimizer.backward_step(&loss);
    }

    fn assert_snapshot_restores_training(device: Device) {
        tch::manual_seed(1);
        let root = temp_dir();
        let replay = VecDeque::from([vec![TrainingSample {
            state: BoardState::new(),
            policy: TicTacToePolicy::one_hot(TicTacToeMove(0, 0)),
            value: -1.0,
        }]]);

        let vs = VarStore::new(device);
        let model = nn::linear(vs.root(), 2, 1, Default::default());
        let mut optimizer = nn::Adam::default().build(&vs, 1e-3).unwrap();
        training_step(&model, &mut optimizer, device);
        save_training_snapshot(&root, 3, &vs, &optimizer, &replay).unwrap();

        training_step(&model, &mut optimizer, device);
        let expected = vs
            .trainable_variables()
            .into_iter()
            .map(|tensor| tensor.copy())
            .collect::<Vec<_>>();

        let mut restored_vs = VarStore::new(device);
        let restored_model = nn::linear(restored_vs.root(), 2, 1, Default::default());
        let mut restored_optimizer = nn::Adam::default().build(&restored_vs, 1e-3).unwrap();
        let snapshot = find_latest_snapshot(&root).unwrap().unwrap();
        let restored_replay =
            load_training_snapshot(&snapshot, &mut restored_vs, &mut restored_optimizer).unwrap();
        assert!(restored_replay == replay);

        training_step(&restored_model, &mut restored_optimizer, device);
        for (expected, actual) in expected
            .iter()
            .zip(restored_vs.trainable_variables().iter())
        {
            assert!(expected.allclose(actual, 1e-6, 1e-7, false));
        }

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn snapshot_restores_model_optimizer_and_replay() {
        assert_snapshot_restores_training(Device::Cpu);
    }

    #[test]
    fn snapshot_restores_mps_optimizer() {
        if tch::utils::has_mps() {
            assert_snapshot_restores_training(Device::Mps);
        }
    }
}
