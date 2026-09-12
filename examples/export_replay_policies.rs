//! Export stored distributions and reconstruct non-final actions for offline sampling audits.
use std::{
    collections::BTreeSet,
    fs::File,
    io::{BufWriter, Write},
    path::PathBuf,
};

use alz::{
    engine::Game,
    gomoku::{BoardState, CellState, GomokuMove},
    training_snapshot::load_replay_checkpoint,
};
use anyhow::{Context, Result, ensure};
use serde_json::json;
use sha2::{Digest, Sha256};

fn selected_action(state: &BoardState, next: &BoardState) -> Result<usize> {
    let changed = (0..361)
        .filter(|&a| {
            state[(a / 19, a % 19)] == CellState::Empty
                && next[(a / 19, a % 19)] != CellState::Empty
        })
        .collect::<Vec<_>>();
    ensure!(
        changed.len() == 1,
        "trajectory does not add exactly one stone"
    );
    let action = changed[0];
    ensure!(
        state.make_move(&GomokuMove::from_xy(action / 19, action % 19)) == *next,
        "successor does not match reconstructed action"
    );
    Ok(action)
}

fn main() -> Result<()> {
    let mut args = std::env::args_os().skip(1);
    let output = PathBuf::from(args.next().context("expected OUTPUT.jsonl CHECKPOINT...")?);
    let paths = args.map(PathBuf::from).collect::<Vec<_>>();
    ensure!(!paths.is_empty(), "at least one checkpoint is required");
    let mut writer = BufWriter::new(File::create(&output)?);
    let mut seen = BTreeSet::new();
    let (mut positions, mut duplicates, mut recoverable) = (0, 0, 0);
    let mut sources = Vec::new();
    for path in paths {
        let replay_sha256 = format!(
            "{:x}",
            Sha256::digest(std::fs::read(path.join("replay.bin.zst"))?)
        );
        let (descriptor, games) = load_replay_checkpoint(&path)?;
        sources.push(
            json!({"checkpoint": descriptor, "replay_sha256": replay_sha256, "games": games.len()}),
        );
        for game in games {
            let digest: [u8; 32] = Sha256::digest(bincode::serialize(&game)?).into();
            if !seen.insert(digest) {
                duplicates += 1;
                continue;
            }
            let game_id = digest
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect::<String>();
            for (ply, sample) in game.iter().enumerate() {
                let chosen = game
                    .get(ply + 1)
                    .map(|next| selected_action(&sample.state, &next.state))
                    .transpose()?;
                if let Some(action) = chosen {
                    ensure!(
                        sample.policy.as_flattened()[action] > 0.0,
                        "played action has zero stored probability"
                    );
                    recoverable += 1;
                }
                let policy = sample
                    .policy
                    .as_flattened()
                    .iter()
                    .enumerate()
                    .filter(|(_, p)| **p > 0.0)
                    .map(|(a, &p)| (a, p))
                    .collect::<Vec<_>>();
                let stones = (0..361)
                    .filter_map(|a| match sample.state[(a / 19, a % 19)] {
                        CellState::Empty => None,
                        CellState::X => Some((a, 1)),
                        CellState::O => Some((a, 2)),
                    })
                    .collect::<Vec<_>>();
                ensure!(
                    stones.len() == ply,
                    "expected complete games starting on an empty board"
                );
                serde_json::to_writer(
                    &mut writer,
                    &json!({"game": game_id, "source_epoch": descriptor.epoch,
                    "ply": ply, "value": sample.value, "chosen": chosen, "policy": policy, "stones": stones}),
                )?;
                writeln!(writer)?;
                positions += 1;
            }
        }
        eprintln!(
            "checkpoint {}: {} unique games, {} positions",
            descriptor.epoch,
            seen.len(),
            positions
        );
    }
    writer.flush()?;
    let metadata = json!({"schema_version": 1, "sources": sources, "unique_games": seen.len(),
        "duplicate_games_removed": duplicates, "positions": positions, "recoverable_actions": recoverable,
        "policy_semantics": "stored targets, analyzed without another temperature transform",
        "deduplication": "SHA256 of complete bincode game including states, policies and outcomes",
        "missing_actions": "final action of each game is not stored and cannot be uniquely reconstructed"});
    serde_json::to_writer_pretty(
        File::create(output.with_extension("metadata.json"))?,
        &metadata,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reconstruction_accounts_for_canonical_player_flip() {
        let first = BoardState::new().make_move(&GomokuMove::from_xy(9, 9));
        let second = first.make_move(&GomokuMove::from_xy(10, 8));
        assert_eq!(selected_action(&first, &second).unwrap(), 10 * 19 + 8);
        assert!(selected_action(&first, &first).is_err());
    }
}
