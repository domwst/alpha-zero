//! Small interoperability probe: read either archive format and write the new one.
use alz::{engine::archive, gomoku::BoardState};
use anyhow::{Result, bail};
use std::path::Path;

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 4 {
        bail!("usage: replay_codec INPUT OUTPUT_DIRECTORY SEED");
    }
    let game = archive::load::<BoardState>(Path::new(&args[1]), args[3].parse()?)?;
    archive::save(Path::new(&args[2]), &game)
}
