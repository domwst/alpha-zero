use std::{
    cmp::Reverse,
    collections::VecDeque,
    fs::{self, File},
    io::{BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use alz::{
    engine::TrainingSample,
    gomoku::{
        ACTION_SCHEMA, BoardState, GAME_SCHEMA, GomokuPolicy, ModelSpec, POSITION_SCHEMA,
        REPLAY_SCHEMA, VALUE_SCHEMA,
    },
};
use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tch::{
    Tensor,
    nn::{Optimizer, VarStore},
};

const FORMAT_VERSION: u32 = 2;
const MODEL_FILE: &str = "model.safetensors";
const OPTIMIZER_FILE: &str = "optimizer.ot";
const REPLAY_FILE: &str = "replay.bin.zst";
const METADATA_FILE: &str = "metadata.json";

pub type ReplayPosition = TrainingSample<BoardState, GomokuPolicy>;
pub type ReplayGame = Vec<ReplayPosition>;
pub type ReplayBuffer = VecDeque<ReplayGame>;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct SnapshotMetadata {
    format_version: u32,
    model: ModelSpec,
    model_sha256: String,
    tensor_schema_sha256: String,
    game_schema: String,
    position_schema: String,
    action_schema: String,
    value_schema: String,
    replay_schema: String,
    optimizer_schema: String,
    epoch: usize,
    replay_games: usize,
    replay_positions: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct SnapshotDescriptor {
    pub path: PathBuf,
    pub format_version: u32,
    pub epoch: usize,
    pub model: ModelSpec,
    pub model_sha256: String,
    pub tensor_schema_sha256: String,
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

    pub fn model_spec(&self) -> &ModelSpec {
        &self.metadata.model
    }

    pub fn descriptor(&self) -> SnapshotDescriptor {
        SnapshotDescriptor {
            path: self.path.clone(),
            format_version: self.metadata.format_version,
            epoch: self.metadata.epoch,
            model: self.metadata.model.clone(),
            model_sha256: self.metadata.model_sha256.clone(),
            tensor_schema_sha256: self.metadata.tensor_schema_sha256.clone(),
        }
    }

    pub fn load_model(&self, var_store: &mut VarStore) -> Result<()> {
        let path = self.path.join(MODEL_FILE);
        let stored_model_sha256 = file_sha256(&path)?;
        ensure!(
            stored_model_sha256 == self.metadata.model_sha256,
            "model digest {} does not match metadata digest {} in {}",
            stored_model_sha256,
            self.metadata.model_sha256,
            path.display()
        );
        let constructed_schema = var_store_tensor_schema_sha256(var_store);
        ensure!(
            constructed_schema == self.metadata.tensor_schema_sha256,
            "constructed {} model tensor schema {} does not match checkpoint schema {}",
            self.metadata.model.architecture_id(),
            constructed_schema,
            self.metadata.tensor_schema_sha256
        );
        let stored_schema = model_file_tensor_schema_sha256(&path)?;
        ensure!(
            stored_schema == self.metadata.tensor_schema_sha256,
            "model tensor schema {} does not match metadata schema {} in {}",
            stored_schema,
            self.metadata.tensor_schema_sha256,
            path.display()
        );
        var_store
            .load(&path)
            .with_context(|| format!("loading {}", path.display()))
    }
}

fn tensor_schema_sha256<'a>(tensors: impl IntoIterator<Item = (&'a str, &'a Tensor)>) -> String {
    let mut tensors = tensors.into_iter().collect::<Vec<_>>();
    tensors.sort_unstable_by_key(|(name, _)| *name);

    let mut hash = Sha256::new();
    for (name, tensor) in tensors {
        hash.update((name.len() as u64).to_le_bytes());
        hash.update(name.as_bytes());
        let dimensions = tensor.size();
        hash.update((dimensions.len() as u64).to_le_bytes());
        for dimension in dimensions {
            hash.update(dimension.to_le_bytes());
        }
        let kind = format!("{:?}", tensor.kind());
        hash.update((kind.len() as u64).to_le_bytes());
        hash.update(kind.as_bytes());
    }
    format!("{:x}", hash.finalize())
}

fn var_store_tensor_schema_sha256(var_store: &VarStore) -> String {
    let variables = var_store.variables();
    tensor_schema_sha256(
        variables
            .iter()
            .map(|(name, tensor)| (name.as_str(), tensor)),
    )
}

fn model_file_tensor_schema_sha256(path: &Path) -> Result<String> {
    let tensors = Tensor::read_safetensors(path)
        .with_context(|| format!("reading tensor schema from {}", path.display()))?;
    Ok(tensor_schema_sha256(
        tensors.iter().map(|(name, tensor)| (name.as_str(), tensor)),
    ))
}

fn file_sha256(path: &Path) -> Result<String> {
    let mut reader = BufReader::new(
        File::open(path).with_context(|| format!("opening {} for hashing", path.display()))?,
    );
    let mut hash = Sha256::new();
    let mut buffer = [0; 64 * 1024];
    loop {
        let read = reader
            .read(&mut buffer)
            .with_context(|| format!("hashing {}", path.display()))?;
        if read == 0 {
            break;
        }
        hash.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

fn snapshot_dir(root: &Path, epoch: usize) -> PathBuf {
    root.join(format!("{epoch:08}"))
}

fn read_metadata(path: &Path) -> Result<SnapshotMetadata> {
    let value: serde_json::Value = serde_json::from_reader(BufReader::new(
        File::open(path).with_context(|| format!("opening {}", path.display()))?,
    ))
    .with_context(|| format!("reading {}", path.display()))?;
    let version = value
        .get("format_version")
        .and_then(serde_json::Value::as_u64)
        .with_context(|| format!("missing format_version in {}", path.display()))?;
    ensure!(
        version == FORMAT_VERSION as u64,
        "unsupported snapshot metadata version {version} in {}; expected version {FORMAT_VERSION}",
        path.display(),
    );
    serde_json::from_value(value).with_context(|| format!("reading {}", path.display()))
}

fn is_complete_snapshot(path: &Path) -> bool {
    [MODEL_FILE, OPTIMIZER_FILE, REPLAY_FILE, METADATA_FILE]
        .into_iter()
        .all(|name| path.join(name).is_file())
}

fn is_model_checkpoint(path: &Path) -> bool {
    [MODEL_FILE, METADATA_FILE]
        .into_iter()
        .all(|name| path.join(name).is_file())
}

pub fn find_latest_snapshot(root: &Path) -> Result<Option<TrainingSnapshot>> {
    find_latest(root, is_complete_snapshot)
}

pub fn find_latest_model_checkpoint(root: &Path) -> Result<Option<TrainingSnapshot>> {
    find_latest(root, is_model_checkpoint)
}

fn find_latest(root: &Path, is_usable: impl Fn(&Path) -> bool) -> Result<Option<TrainingSnapshot>> {
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
        if !is_usable(&path) {
            continue;
        }
        candidates.push((epoch, path));
    }

    candidates.sort_unstable_by_key(|(epoch, _)| Reverse(*epoch));
    let mut candidates = candidates.into_iter();
    let Some((latest_epoch, latest_path)) = candidates.next() else {
        return Ok(None);
    };
    let latest = read_snapshot(latest_path, latest_epoch)?;

    for (epoch, path) in candidates {
        let snapshot = read_snapshot(path, epoch)?;
        ensure!(
            snapshot.metadata.model == latest.metadata.model
                && snapshot.metadata.tensor_schema_sha256 == latest.metadata.tensor_schema_sha256,
            "checkpoint directory {} mixes model schemas: epoch {} is {} ({}) while epoch {} is {} ({})",
            root.display(),
            snapshot.epoch(),
            snapshot.model_spec().architecture_id(),
            snapshot.metadata.tensor_schema_sha256,
            latest.epoch(),
            latest.model_spec().architecture_id(),
            latest.metadata.tensor_schema_sha256,
        );
    }
    Ok(Some(latest))
}

pub fn resolve_model_checkpoint(path: &Path) -> Result<Option<TrainingSnapshot>> {
    if !is_model_checkpoint(path) {
        return find_latest_model_checkpoint(path);
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
    if metadata.game_schema != GAME_SCHEMA {
        bail!(
            "unsupported game schema {} in {}",
            metadata.game_schema,
            path.display()
        );
    }
    if metadata.position_schema != POSITION_SCHEMA {
        bail!(
            "unsupported position schema {} in {}",
            metadata.position_schema,
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
    if metadata.value_schema != VALUE_SCHEMA {
        bail!(
            "unsupported value schema {} in {}",
            metadata.value_schema,
            path.display()
        );
    }
    if metadata.replay_schema != REPLAY_SCHEMA {
        bail!(
            "unsupported replay schema {} in {}",
            metadata.replay_schema,
            path.display()
        );
    }
    if metadata.optimizer_schema != "adam-v1" {
        bail!(
            "unsupported optimizer schema {} in {}",
            metadata.optimizer_schema,
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
    model_spec: &ModelSpec,
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
            model: model_spec.clone(),
            model_sha256: file_sha256(&model_path)?,
            tensor_schema_sha256: var_store_tensor_schema_sha256(var_store),
            game_schema: GAME_SCHEMA.to_owned(),
            position_schema: POSITION_SCHEMA.to_owned(),
            action_schema: ACTION_SCHEMA.to_owned(),
            value_schema: VALUE_SCHEMA.to_owned(),
            replay_schema: REPLAY_SCHEMA.to_owned(),
            optimizer_schema: "adam-v1".to_owned(),
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
        Device, Tensor,
        nn::{self, OptimizerConfig},
    };

    use alz::{
        engine::AlphaZeroNet,
        gomoku::{GomokuModel, GomokuMove},
    };

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
        write_fake_snapshot_with_schema(root, epoch, complete, "fake-schema");
    }

    fn write_fake_snapshot_with_schema(
        root: &Path,
        epoch: usize,
        complete: bool,
        tensor_schema: &str,
    ) {
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
                model: ModelSpec::LegacyResNetV1,
                model_sha256: "fake-model-digest".to_owned(),
                tensor_schema_sha256: tensor_schema.to_owned(),
                game_schema: GAME_SCHEMA.to_owned(),
                position_schema: POSITION_SCHEMA.to_owned(),
                action_schema: ACTION_SCHEMA.to_owned(),
                value_schema: VALUE_SCHEMA.to_owned(),
                replay_schema: REPLAY_SCHEMA.to_owned(),
                optimizer_schema: "adam-v1".to_owned(),
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

        let latest = resolve_model_checkpoint(&root).unwrap().unwrap();
        assert_eq!(latest.epoch(), 7);
        let snapshot = resolve_model_checkpoint(&snapshot_dir(&root, 1))
            .unwrap()
            .unwrap();
        assert_eq!(snapshot.epoch(), 1);

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn run_directory_rejects_mixed_tensor_schemas() {
        let root = temp_dir();
        fs::create_dir(&root).unwrap();
        write_fake_snapshot_with_schema(&root, 1, true, "first-schema");
        write_fake_snapshot_with_schema(&root, 2, true, "second-schema");

        let error = find_latest_snapshot(&root).unwrap_err();
        assert!(error.to_string().contains("mixes model schemas"));

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn legacy_metadata_version_is_rejected() {
        let root = temp_dir();
        fs::create_dir(&root).unwrap();
        write_fake_snapshot(&root, 1, true);

        let metadata_path = snapshot_dir(&root, 1).join(METADATA_FILE);
        let mut metadata: serde_json::Value =
            serde_json::from_slice(&fs::read(&metadata_path).unwrap()).unwrap();
        metadata["format_version"] = serde_json::Value::from(1);
        fs::write(
            &metadata_path,
            serde_json::to_vec_pretty(&metadata).unwrap(),
        )
        .unwrap();

        let error = resolve_model_checkpoint(&root).unwrap_err();
        assert!(error.to_string().contains("expected version 2"));

        fs::remove_dir_all(root).unwrap();
    }

    fn training_step(model: &GomokuModel, optimizer: &mut nn::Optimizer, device: Device) {
        let input = Tensor::zeros([1, 2, 19, 19], (tch::Kind::Float, device));
        let output = model.forward_t(&input, true);
        let loss = output.values.square().mean(tch::Kind::Float)
            + output.policy_logits.square().mean(tch::Kind::Float);
        optimizer.backward_step(&loss);
    }

    fn assert_snapshot_restores_training(device: Device, model_spec: ModelSpec) {
        tch::manual_seed(1);
        let root = temp_dir();
        let replay = VecDeque::from([vec![TrainingSample {
            state: BoardState::new(),
            policy: GomokuPolicy::one_hot(GomokuMove::from_xy(0, 0)),
            value: -1.0,
        }]]);

        let vs = VarStore::new(device);
        let model = GomokuModel::new(vs.root(), &model_spec);
        let mut optimizer = nn::Adam::default().build(&vs, 1e-3).unwrap();
        training_step(&model, &mut optimizer, device);
        save_training_snapshot(&root, 3, &model_spec, &vs, &optimizer, &replay).unwrap();

        training_step(&model, &mut optimizer, device);
        let expected = vs
            .trainable_variables()
            .into_iter()
            .map(|tensor| tensor.copy())
            .collect::<Vec<_>>();

        let mut restored_vs = VarStore::new(device);
        let restored_model = GomokuModel::new(restored_vs.root(), &model_spec);
        let mut restored_optimizer = nn::Adam::default().build(&restored_vs, 1e-3).unwrap();
        let snapshot = find_latest_snapshot(&root).unwrap().unwrap();
        assert_eq!(snapshot.model_spec(), &model_spec);
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
        assert_snapshot_restores_training(Device::Cpu, ModelSpec::LegacyResNetV1);
    }

    #[test]
    fn kata_snapshot_restores_model_optimizer_and_replay() {
        assert_snapshot_restores_training(Device::Cpu, ModelSpec::KataV1);
    }

    #[test]
    fn corrupted_model_is_rejected() {
        let root = temp_dir();
        let replay = ReplayBuffer::new();
        let vs = VarStore::new(Device::Cpu);
        let _model = GomokuModel::new(vs.root(), &ModelSpec::LegacyResNetV1);
        let optimizer = nn::Adam::default().build(&vs, 1e-3).unwrap();
        save_training_snapshot(
            &root,
            4,
            &ModelSpec::LegacyResNetV1,
            &vs,
            &optimizer,
            &replay,
        )
        .unwrap();

        let snapshot = resolve_model_checkpoint(&root).unwrap().unwrap();
        assert_eq!(snapshot.model_spec(), &ModelSpec::LegacyResNetV1);
        let mut restored_vs = VarStore::new(Device::Cpu);
        let _restored_model = GomokuModel::new(restored_vs.root(), &ModelSpec::LegacyResNetV1);
        snapshot.load_model(&mut restored_vs).unwrap();

        let model_path = snapshot_dir(&root, 4).join(MODEL_FILE);
        let mut bytes = fs::read(&model_path).unwrap();
        *bytes.last_mut().unwrap() ^= 1;
        fs::write(&model_path, bytes).unwrap();
        let error = snapshot.load_model(&mut restored_vs).unwrap_err();
        assert!(error.to_string().contains("model digest"));

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn snapshot_restores_mps_optimizer() {
        if tch::utils::has_mps() {
            assert_snapshot_restores_training(Device::Mps, ModelSpec::LegacyResNetV1);
        }
    }
}
