//! Recover outcomes of a completed self-play epoch from the newest replay games.
use alz::{gomoku::BoardState, training_snapshot::load_replay_checkpoint};
use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use std::{fs, path::PathBuf};

fn main() -> Result<()> {
    let stats_path = PathBuf::from(
        std::env::args_os()
            .nth(1)
            .context("expected epoch stats path")?,
    );
    let stats: Value = serde_json::from_slice(&fs::read(&stats_path)?)?;
    let games = stats["games"].as_u64().context("missing games")? as usize;
    let epoch = stats["epoch"].as_u64().context("missing epoch")?;
    let checkpoint = stats_path
        .parent()
        .context("stats parent")?
        .parent()
        .context("run parent")?
        .join("checkpoints")
        .join(format!("{epoch:08}"));
    let (_, replay) = load_replay_checkpoint(&checkpoint)?;
    ensure!(
        games > 0 && replay.len() >= games,
        "epoch games not fully retained in replay"
    );
    // Each epoch is shuffled, then appended to the FIFO buffer. Older games precede it.
    let mut wins = 0usize;
    let mut losses = 0usize;
    let mut draws = 0usize;
    let mut lengths = Vec::with_capacity(games);
    for game in replay.iter().rev().take(games) {
        let first = game.first().context("empty replay game")?;
        ensure!(
            first.state == BoardState::new(),
            "trajectory does not start with first player"
        );
        match first.value {
            1.0 => wins += 1,
            -1.0 => losses += 1,
            0.0 => draws += 1,
            value => anyhow::bail!("unexpected game outcome {value}"),
        }
        lengths.push(game.len());
    }
    ensure!(
        lengths.iter().sum::<usize>() as u64
            == stats["total_game_length"]
                .as_u64()
                .context("missing total length")?,
        "replay lengths do not match epoch; some new games may have been evicted"
    );
    ensure!(
        (wins as f64 - losses as f64 - stats["total_score"].as_f64().context("missing score")?)
            .abs()
            < 1e-6,
        "replay outcomes do not match epoch score"
    );
    println!(
        "{}",
        json!({"epoch":epoch,"games":games,"first_player_wins":wins,
        "second_player_wins":losses,"draws":draws,"first_player_win_rate":wins as f64/games as f64,
        "source":"verified_epoch_replays","total_game_length":lengths.iter().sum::<usize>()})
    );
    Ok(())
}
