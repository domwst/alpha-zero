//! Rich records are separate from the compact training buffer. Publish atomically.
use super::{Game, MatchRecord};
use anyhow::{Result, ensure};
use serde::{Serialize, de::DeserializeOwned};
use std::{
    fs::{self, File},
    io::{BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
};

pub const SUFFIX: &str = ".msgpack.zst";
const MAX_REPLAY_BYTES: u64 = 128 * 1024 * 1024;

/// Prefer the compressed record when both formats exist during migration.
pub fn find(directory: &Path, game_id: &str) -> Option<PathBuf> {
    [SUFFIX, ".json"]
        .into_iter()
        .map(|suffix| directory.join(format!("{game_id}{suffix}")))
        .find(|path| path.is_file())
}

#[derive(Serialize, serde::Deserialize)]
#[serde(bound(
    serialize = "G: Serialize, G::Move: Serialize",
    deserialize = "G: serde::Deserialize<'de>, G::Move: serde::Deserialize<'de>"
))]
pub struct ArchivedGame<G: Game> {
    pub schema_version: u32,
    pub game_type: String,
    pub game_id: String,
    pub seed: u64,
    pub model_identity: Option<String>,
    #[serde(default)]
    pub models_by_seat: std::collections::BTreeMap<String, String>,
    #[serde(default)]
    pub duration_seconds: Option<f64>,
    pub policy_semantics: String,
    pub provenance: String,
    pub record: MatchRecord<G>,
}

pub fn save<G: Game + Serialize>(directory: &Path, game: &ArchivedGame<G>) -> Result<()>
where
    G::Move: Serialize,
{
    fs::create_dir_all(directory)?;
    ensure!(
        !game.game_id.is_empty()
            && game.game_id.len() <= 64
            && game
                .game_id
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
        "invalid game id"
    );
    let target = directory.join(format!("{}{SUFFIX}", game.game_id));
    let mut payload = Vec::new();
    game.serialize(&mut rmp_serde::Serializer::new(&mut payload).with_struct_map())?;
    ensure!(
        payload.len() as u64 <= MAX_REPLAY_BYTES,
        "replay exceeds size limit"
    );
    let temporary = target.with_extension("zst.tmp");
    let mut encoder = zstd::stream::Encoder::new(BufWriter::new(File::create(&temporary)?), 6)?;
    encoder.include_checksum(true)?;
    encoder.set_pledged_src_size(Some(payload.len() as u64))?;
    encoder.write_all(&payload)?;
    let mut writer = encoder.finish()?;
    writer.flush()?;
    writer.get_ref().sync_all()?;
    fs::rename(temporary, target)?;
    File::open(directory)?.sync_all()?;
    Ok(())
}

pub fn publish_json<T: Serialize>(target: &Path, value: &T) -> Result<()> {
    let directory = target.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(directory)?;
    let tmp = target.with_extension("json.tmp");
    let mut writer = BufWriter::with_capacity(64 * 1024, File::create(&tmp)?);
    serde_json::to_writer(&mut writer, value)?;
    writer.flush()?;
    writer.get_ref().sync_all()?;
    fs::rename(tmp, target)?;
    File::open(directory)?.sync_all()?;
    Ok(())
}

pub fn load<G: Game + DeserializeOwned>(path: &Path, seed: u64) -> Result<ArchivedGame<G>>
where
    G::Move: DeserializeOwned,
{
    let game: ArchivedGame<G> = if path.to_string_lossy().ends_with(SUFFIX) {
        let decoder = zstd::stream::read::Decoder::new(File::open(path)?)?;
        let mut payload = Vec::new();
        decoder
            .take(MAX_REPLAY_BYTES + 1)
            .read_to_end(&mut payload)?;
        ensure!(
            payload.len() as u64 <= MAX_REPLAY_BYTES,
            "replay exceeds size limit"
        );
        let mut cursor = std::io::Cursor::new(&payload);
        let game = serde::Deserialize::deserialize(&mut rmp_serde::Deserializer::new(&mut cursor))?;
        ensure!(
            cursor.position() == payload.len() as u64,
            "trailing replay data"
        );
        game
    } else {
        serde_json::from_reader(BufReader::new(File::open(path)?))?
    };
    ensure!(
        game.schema_version == 1 && game.seed == seed,
        "game archive identity mismatch"
    );
    Ok(game)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{engine::Seat, gomoku::BoardState};

    struct Directory(PathBuf);

    impl Directory {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!(
                "alz-archive-test-{}-{}",
                std::process::id(),
                rand::random::<u64>()
            ));
            fs::create_dir(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for Directory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn fixture() -> ArchivedGame<BoardState> {
        ArchivedGame {
            schema_version: 1,
            game_type: "gomoku".into(),
            game_id: "00000001".into(),
            seed: u64::MAX,
            model_identity: None,
            models_by_seat: Default::default(),
            duration_seconds: Some(1.2345678901234567),
            policy_semantics: "visits".into(),
            provenance: "test".into(),
            record: MatchRecord {
                plies: vec![],
                terminal_state: BoardState::new(),
                terminal_actor: Seat::Second,
                terminal_value: 0.12345678,
            },
        }
    }

    #[test]
    fn json_and_compressed_games_roundtrip_and_prefer_compressed() {
        let directory = Directory::new();
        let game = fixture();
        let legacy = directory.0.join("00000001.json");
        publish_json(&legacy, &game).unwrap();
        assert_eq!(find(&directory.0, &game.game_id), Some(legacy.clone()));
        save(&directory.0, &game).unwrap();
        let compressed = directory.0.join(format!("00000001{SUFFIX}"));
        assert_eq!(find(&directory.0, &game.game_id), Some(compressed.clone()));
        for path in [legacy, compressed] {
            let restored = load::<BoardState>(&path, game.seed).unwrap();
            assert_eq!(
                restored.record.terminal_value.to_bits(),
                game.record.terminal_value.to_bits()
            );
            assert_eq!(restored.duration_seconds, game.duration_seconds);
            assert_eq!(restored.seed, u64::MAX);
            assert_eq!(restored.record.terminal_actor, Seat::Second);
            assert!(load::<BoardState>(&path, 0).is_err());
        }
    }

    #[test]
    fn rejects_corruption_and_trailing_messagepack_data() {
        let directory = Directory::new();
        let game = fixture();
        save(&directory.0, &game).unwrap();
        let path = find(&directory.0, &game.game_id).unwrap();
        let encoded = fs::read(&path).unwrap();
        fs::write(&path, &encoded[..encoded.len() - 1]).unwrap();
        assert!(load::<BoardState>(&path, game.seed).is_err());
        let mut payload = zstd::decode_all(encoded.as_slice()).unwrap();
        payload.push(0xc0);
        fs::write(&path, zstd::encode_all(payload.as_slice(), 1).unwrap()).unwrap();
        assert!(load::<BoardState>(&path, game.seed).is_err());
    }
}
