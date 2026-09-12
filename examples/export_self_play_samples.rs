//! Recover eight game samples and publish browser-sized tiled frame sheets.
use alz::{gomoku::generate_game_image, training_snapshot::load_replay_checkpoint};
use anyhow::{Context, Result, ensure};
use image::{ImageFormat, Rgb, RgbImage};
use rand::{SeedableRng, rngs::SmallRng, seq::SliceRandom};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{collections::HashSet, fs, path::Path};

const SIDE: u32 = 380;
const STRIDE: u32 = 390;

fn save_png(path: &Path, image: &RgbImage) -> Result<()> {
    let temporary = path.with_extension("png.tmp");
    image.save_with_format(&temporary, ImageFormat::Png)?;
    fs::rename(temporary, path)?;
    Ok(())
}

fn tile_strip(strip: &RgbImage) -> Result<RgbImage> {
    ensure!(
        strip.height() == SIDE && (strip.width() + 10) % STRIDE == 0,
        "Invalid frame strip"
    );
    let positions = (strip.width() + 10) / STRIDE;
    ensure!((1..=361).contains(&positions), "Invalid position count");
    let columns = positions.min(16);
    let mut tiled = RgbImage::from_pixel(
        columns * STRIDE - 10,
        positions.div_ceil(columns) * STRIDE - 10,
        Rgb([255; 3]),
    );
    for frame in 0..positions {
        let board = image::imageops::crop_imm(strip, frame * STRIDE, 0, SIDE, SIDE).to_image();
        image::imageops::replace(
            &mut tiled,
            &board,
            ((frame % columns) * STRIDE).into(),
            ((frame / columns) * STRIDE).into(),
        );
    }
    Ok(tiled)
}

fn publish_tiles(path: &Path, strip: &RgbImage) -> Result<()> {
    let tiled = tile_strip(strip)?;
    save_png(&path.with_extension("tiles.png"), &tiled)?;
    Ok(())
}

fn main() -> Result<()> {
    let stats_path = std::env::args_os()
        .nth(1)
        .context("Expected epoch stats path")?;
    let stats_path = Path::new(&stats_path);
    let content = fs::read(stats_path)?;
    let stats: Value = serde_json::from_slice(&content)?;
    let epoch = stats["epoch"].as_u64().context("Missing epoch")?;
    let count = stats["games"].as_u64().context("Missing game count")? as usize;
    let run = stats_path
        .parent()
        .context("Stats directory")?
        .parent()
        .context("Run directory")?;
    let (_, replay) = load_replay_checkpoint(&run.join("checkpoints").join(format!("{epoch:08}")))?;
    ensure!(
        count > 0 && replay.len() >= count,
        "Epoch games no longer fully retained"
    );
    let mut games = replay.iter().rev().take(count).collect::<Vec<_>>();
    ensure!(
        games.iter().map(|g| g.len() as u64).sum::<u64>()
            == stats["total_game_length"]
                .as_u64()
                .context("Missing game lengths")?,
        "Recent replay games do not match epoch lengths"
    );
    let score: f64 = games
        .iter()
        .map(|g| g.first().map(|s| s.value as f64).unwrap_or(f64::NAN))
        .sum();
    ensure!(
        (score
            - stats["total_score"]
                .as_f64()
                .context("Missing epoch score")?)
        .abs()
            < 1e-6,
        "Recent replay outcomes do not match epoch"
    );
    let directory = run.join("games");
    fs::create_dir_all(&directory)?;
    let mut seen = HashSet::new();
    let target = count.min(8);
    let mut missing = Vec::new();
    // Keep the already published sample identities, including the failing long game.
    for sample in 0..target {
        let path = directory.join(format!("{epoch:08}.{sample:02}.png"));
        if path.exists() {
            let strip = image::open(&path)?.to_rgb8();
            seen.insert(Sha256::digest(strip.as_raw()).to_vec());
            publish_tiles(&path, &strip)?;
        } else {
            missing.push(sample);
        }
    }
    let seed = stats["config"]["seed"].as_u64().unwrap_or(0) ^ epoch ^ 0x5341_4d50_4c45;
    games.shuffle(&mut SmallRng::seed_from_u64(seed));
    let mut candidates = games.into_iter();
    for sample in missing {
        let strip = loop {
            let Some(game) = candidates.next() else {
                anyhow::bail!("Too few distinct game images");
            };
            let image = generate_game_image(game);
            if seen.insert(Sha256::digest(image.as_raw()).to_vec()) {
                break image;
            }
        };
        let path = directory.join(format!("{epoch:08}.{sample:02}.png"));
        save_png(&path, &strip)?;
        publish_tiles(&path, &strip)?;
    }
    let marker = directory.join(format!("{epoch:08}.samples.json"));
    let temporary = marker.with_extension("json.tmp");
    fs::write(
        &temporary,
        serde_json::to_vec_pretty(&json!({"epoch":epoch,"samples":target,
        "stats_sha256":format!("{:x}",Sha256::digest(&content)),"layout":"tiled_16_columns_v1"}))?,
    )?;
    fs::rename(temporary, marker)?;
    println!("epoch {}: {} samples ready", epoch + 1, target);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn long_game_tiles_preserve_every_frame() {
        let count = 194;
        let mut strip = RgbImage::new(count * STRIDE - 10, SIDE);
        for frame in 0..count {
            strip.put_pixel(frame * STRIDE + 17, 23, Rgb([frame as u8, 19, 42]));
        }
        let tiled = tile_strip(&strip).unwrap();
        assert_eq!(tiled.dimensions(), (6230, 5060));
        for frame in 0..count {
            assert_eq!(
                tiled.get_pixel(frame % 16 * STRIDE + 17, frame / 16 * STRIDE + 23),
                &Rgb([frame as u8, 19, 42])
            );
        }
    }
}
