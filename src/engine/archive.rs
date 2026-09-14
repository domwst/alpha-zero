//! Rich records are separate from the compact training buffer. Publish atomically.
use super::{Game, MatchRecord};
use anyhow::{Result, ensure};
use serde::{Serialize, de::DeserializeOwned};
use std::{
    fs::{self, File},
    io::{BufReader, BufWriter, Write},
    path::Path,
};

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
    let target = directory.join(format!("{}.json", game.game_id));
    publish_json(&target, game)
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
    let game: ArchivedGame<G> = serde_json::from_reader(BufReader::new(File::open(path)?))?;
    ensure!(
        game.schema_version == 1 && game.seed == seed,
        "game archive identity mismatch"
    );
    Ok(game)
}
