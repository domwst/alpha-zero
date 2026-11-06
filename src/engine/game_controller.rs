//! Admission policy for independent games. Inference batching remains in the executor.
use super::telemetry::{self, GameProgress};
use anyhow::{Context, Result};
use std::{
    collections::{BTreeMap, VecDeque},
    future::Future,
    time::{Duration, Instant},
};
use tokio::{
    sync::watch,
    task::{Id, JoinSet},
};

pub struct CompletedGame<T> {
    pub game_id: usize,
    pub duration: Duration,
    pub output: T,
}

pub struct GameController<T> {
    pending: VecDeque<usize>,
    active: JoinSet<Result<CompletedGame<T>>>,
    task_games: BTreeMap<Id, usize>,
    progress: BTreeMap<usize, watch::Receiver<GameProgress>>,
    limit: usize,
}

impl<T: Send + 'static> GameController<T> {
    pub fn new(games: usize, limit: usize) -> Self {
        assert!(limit > 0);
        Self {
            pending: (0..games).collect(),
            active: JoinSet::new(),
            task_games: BTreeMap::new(),
            progress: BTreeMap::new(),
            limit,
        }
    }
    /// Reducing the limit leaves current games running and stops new admissions.
    pub fn set_limit(&mut self, limit: usize) {
        assert!(limit > 0);
        self.limit = limit;
    }
    pub fn active_games(&self) -> usize {
        self.active.len()
    }
    pub fn pending_games(&self) -> usize {
        self.pending.len()
    }
    pub fn progress(&self) -> Vec<GameProgress> {
        self.progress
            .values()
            .map(|rx| rx.borrow().clone())
            .collect()
    }

    /// Durable heartbeats contain aggregates; per-game watch channels retain only
    /// the latest update and can later feed a transient live-game view.
    pub fn summary(&self) -> serde_json::Value {
        let progress = self.progress();
        serde_json::json!({
            "active_games": progress.len(),
            "pending_games": self.pending.len(),
            "moves_in_active_games": progress.iter().map(|p| p.move_sequence).sum::<usize>(),
            "expanded_nodes": progress.iter().map(|p| p.expanded_nodes).sum::<u64>(),
            "allocated_nodes": progress.iter().map(|p| p.allocated_nodes).sum::<u64>()
        })
    }

    pub fn admit<F, Fut>(&mut self, mut create: F)
    where
        F: FnMut(usize) -> Fut,
        Fut: Future<Output = Result<T>> + Send + 'static,
    {
        while self.active.len() < self.limit {
            let Some(game_id) = self.pending.pop_front() else {
                break;
            };
            let (sender, receiver) = telemetry::game_channel(game_id);
            self.progress.insert(game_id, receiver);
            let future = create(game_id);
            let task = self.active.spawn(async move {
                let started = Instant::now();
                let output = telemetry::in_game(sender, future)
                    .await
                    .with_context(|| format!("self-play game {game_id} failed"))?;
                Ok(CompletedGame {
                    game_id,
                    duration: started.elapsed(),
                    output,
                })
            });
            self.task_games.insert(task.id(), game_id);
        }
    }
    pub async fn next(&mut self) -> Option<Result<CompletedGame<T>>> {
        let result = self.active.join_next_with_id().await?;
        let task_id = match &result {
            Ok((id, _)) => *id,
            Err(error) => error.id(),
        };
        if let Some(game_id) = self.task_games.remove(&task_id) {
            self.progress.remove(&game_id);
        }
        Some(match result {
            Ok((_, game)) => game,
            Err(error) => Err(error.into()),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn failed_and_panicked_games_release_progress_and_admission() {
        for panic in [false, true] {
            let mut controller = GameController::<usize>::new(2, 1);
            controller.admit(move |_| async move {
                assert!(!panic, "test game panic");
                anyhow::bail!("test game failure")
            });
            assert!(controller.next().await.unwrap().is_err());
            assert_eq!(controller.active_games(), 0);
            assert!(controller.progress().is_empty());
            controller.admit(|id| async move { Ok(id) });
            assert_eq!(controller.next().await.unwrap().unwrap().output, 1);
            assert!(controller.progress().is_empty());
        }
    }

    #[tokio::test]
    async fn lowering_limit_drains_without_admitting_another_game() {
        let mut controller = GameController::new(5, 2);
        controller.admit(|id| async move { Ok(id) });
        assert_eq!(controller.active_games(), 2);
        controller.set_limit(1);
        controller.next().await.unwrap().unwrap();
        controller.admit(|id| async move { Ok(id) });
        assert_eq!(controller.pending_games(), 3);
        controller.next().await.unwrap().unwrap();
        controller.admit(|id| async move { Ok(id) });
        assert_eq!(controller.active_games(), 1);
        assert_eq!(controller.pending_games(), 2);
    }
}
