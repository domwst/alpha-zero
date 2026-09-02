use std::{
    collections::VecDeque,
    fs::File,
    io::{BufReader, BufWriter},
    path::PathBuf,
};

use alz::{
    engine::TrainingSample,
    gomoku::{BoardState, GomokuPolicy},
};
use anyhow::{Context, Result, ensure};
use serde::Serialize;

type ReplayGame = Vec<TrainingSample<BoardState, GomokuPolicy>>;

#[derive(Serialize)]
struct ReplayLengths {
    schema_version: u32,
    replay_games: usize,
    replay_positions: usize,
    game_lengths: Vec<usize>,
}

fn main() -> Result<()> {
    let mut args = std::env::args_os().skip(1);
    let replay_path = PathBuf::from(args.next().context("missing replay path")?);
    let output_path = PathBuf::from(args.next().context("missing output path")?);
    ensure!(args.next().is_none(), "expected: REPLAY OUTPUT");

    let reader = BufReader::new(
        File::open(&replay_path).with_context(|| format!("opening {}", replay_path.display()))?,
    );
    let decoder = zstd::stream::read::Decoder::new(reader)
        .with_context(|| format!("opening zstd stream for {}", replay_path.display()))?;
    let replay: VecDeque<ReplayGame> = bincode::deserialize_from(decoder)
        .with_context(|| format!("deserializing {}", replay_path.display()))?;
    let lengths = ReplayLengths {
        schema_version: 1,
        replay_games: replay.len(),
        replay_positions: replay.iter().map(Vec::len).sum(),
        game_lengths: replay.iter().map(Vec::len).collect(),
    };

    let writer = BufWriter::new(
        File::create(&output_path)
            .with_context(|| format!("creating {}", output_path.display()))?,
    );
    serde_json::to_writer_pretty(writer, &lengths)
        .with_context(|| format!("writing {}", output_path.display()))?;
    Ok(())
}
