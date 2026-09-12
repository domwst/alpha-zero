use std::path::PathBuf;

use alz::{gomoku::ModelSpec, training_batches::ReplayCacheMode};
use clap::{Args, Parser, Subcommand, ValueEnum};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum AdamBackendChoice {
    #[default]
    Standard,
    Fused,
}

#[derive(Clone, Debug, Default, Args, Serialize)]
pub struct TrainingPerformanceArgs {
    /// Adam implementation. Fused supports CPU/CUDA and may change rounding.
    #[arg(long, value_enum, default_value = "standard")]
    pub adam_backend: AdamBackendChoice,

    /// Pre-encode all replay symmetries in CPU or training-device memory.
    #[arg(long, value_enum, default_value = "none")]
    pub replay_cache: ReplayCacheMode,

    /// CPU batches prepared ahead of training (0..16); requires --replay-cache cpu.
    #[arg(long, default_value_t = 0)]
    pub prefetch_batches: usize,
}

impl TrainingPerformanceArgs {
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.prefetch_batches <= 16,
            "prefetch-batches must be at most 16"
        );
        anyhow::ensure!(
            self.prefetch_batches == 0 || self.replay_cache == ReplayCacheMode::Cpu,
            "prefetch-batches requires --replay-cache cpu"
        );
        Ok(())
    }
}

#[derive(Debug, Parser)]
#[command(
    version,
    about = "AlphaZero training and evaluation",
    arg_required_else_help = true
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Train from the latest complete snapshot, or initialize a new run.
    Train(TrainArgs),
    /// Train a fresh model on a fixed saved replay buffer, with a held-out game split.
    TrainReplay(TrainReplayArgs),
    /// Play a game using the latest complete snapshot.
    Play(PlayArgs),
    /// Evaluate two snapshots against each other.
    Battle(BattleArgs),
    /// Measure inference, training, or self-play throughput without creating checkpoints.
    Benchmark(BenchmarkArgs),
}

#[derive(Clone, Copy, Debug, Serialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum DeviceChoice {
    Auto,
    Cpu,
    Mps,
    Cuda,
}

#[derive(Clone, Debug, Args, Serialize)]
pub struct DeviceArgs {
    /// Compute device. Auto prefers MPS, then CUDA, then CPU.
    #[arg(long, value_enum, default_value = "auto")]
    pub device: DeviceChoice,

    /// CUDA device index when --device cuda is selected.
    #[arg(long, default_value_t = 0)]
    pub cuda_index: usize,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum ArchitectureChoice {
    #[value(name = "legacy-resnet-v1")]
    #[serde(rename = "legacy_resnet_v1")]
    LegacyResNetV1,
    #[value(name = "kata-v1")]
    #[serde(rename = "kata_v1")]
    KataV1,
    #[value(name = "kata-gelu-v1")]
    #[serde(rename = "kata_gelu_v1")]
    KataGeluV1,
    #[value(name = "kata-value64-v1")]
    #[serde(rename = "kata_value64_v1")]
    KataValue64V1,
    #[value(name = "kata-value64x2-v1")]
    #[serde(rename = "kata_value64x2_v1")]
    KataValue64x2V1,
    #[value(name = "kata-gelu-value64-v1")]
    #[serde(rename = "kata_gelu_value64_v1")]
    KataGeluValue64V1,
    #[value(name = "kata-gelu-value64x2-v1")]
    #[serde(rename = "kata_gelu_value64x2_v1")]
    KataGeluValue64x2V1,
    #[value(name = "kata-gelu-boardmask-value64x2-v1")]
    #[serde(rename = "kata_gelu_boardmask_value64x2_v1")]
    #[default]
    KataGeluBoardMaskValue64x2V1,
    #[value(name = "kata-gelu-b16c32-value64x2-v1")]
    #[serde(rename = "kata_gelu_b16c32_value64x2_v1")]
    KataGeluB16C32Value64x2V1,
    #[value(name = "kata-gelu-b16c32g3-value64x2-v1")]
    #[serde(rename = "kata_gelu_b16c32g3_value64x2_v1")]
    KataGeluB16C32G3Value64x2V1,
    #[value(name = "kata-gelu-b10c48-value64x2-v1")]
    #[serde(rename = "kata_gelu_b10c48_value64x2_v1")]
    KataGeluB10C48Value64x2V1,
    #[value(name = "kata-pool-v1")]
    #[serde(rename = "kata_pool_v1")]
    KataPoolV1,
    #[value(name = "kata-gelu-pool-v1")]
    #[serde(rename = "kata_gelu_pool_v1")]
    KataGeluPoolV1,
    #[value(name = "kata-pool-value64-v1")]
    #[serde(rename = "kata_pool_value64_v1")]
    KataPoolValue64V1,
    #[value(name = "kata-pool-value64x2-v1")]
    #[serde(rename = "kata_pool_value64x2_v1")]
    KataPoolValue64x2V1,
    #[value(name = "kata-gelu-pool-value64-v1")]
    #[serde(rename = "kata_gelu_pool_value64_v1")]
    KataGeluPoolValue64V1,
    #[value(name = "kata-gelu-pool-value64x2-v1")]
    #[serde(rename = "kata_gelu_pool_value64x2_v1")]
    KataGeluPoolValue64x2V1,
}

impl std::fmt::Display for ArchitectureChoice {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(
            self.to_possible_value()
                .expect("architecture CLI name")
                .get_name(),
        )
    }
}

impl From<ArchitectureChoice> for ModelSpec {
    fn from(value: ArchitectureChoice) -> Self {
        match value {
            ArchitectureChoice::LegacyResNetV1 => Self::LegacyResNetV1,
            ArchitectureChoice::KataV1 => Self::KataV1,
            ArchitectureChoice::KataPoolV1 => Self::KataPoolV1,
            ArchitectureChoice::KataGeluPoolV1 => Self::KataGeluPoolV1,
            ArchitectureChoice::KataPoolValue64V1 => Self::KataPoolValue64V1,
            ArchitectureChoice::KataPoolValue64x2V1 => Self::KataPoolValue64x2V1,
            ArchitectureChoice::KataGeluPoolValue64V1 => Self::KataGeluPoolValue64V1,
            ArchitectureChoice::KataGeluPoolValue64x2V1 => Self::KataGeluPoolValue64x2V1,

            ArchitectureChoice::KataGeluV1 => Self::KataGeluV1,
            ArchitectureChoice::KataValue64V1 => Self::KataValue64V1,
            ArchitectureChoice::KataValue64x2V1 => Self::KataValue64x2V1,
            ArchitectureChoice::KataGeluValue64V1 => Self::KataGeluValue64V1,
            ArchitectureChoice::KataGeluValue64x2V1 => Self::KataGeluValue64x2V1,
            ArchitectureChoice::KataGeluBoardMaskValue64x2V1 => Self::KataGeluBoardMaskValue64x2V1,
            ArchitectureChoice::KataGeluB16C32Value64x2V1 => Self::KataGeluB16C32Value64x2V1,
            ArchitectureChoice::KataGeluB16C32G3Value64x2V1 => Self::KataGeluB16C32G3Value64x2V1,
            ArchitectureChoice::KataGeluB10C48Value64x2V1 => Self::KataGeluB10C48Value64x2V1,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum InferenceSymmetryChoice {
    /// Use the canonical board orientation for every network evaluation.
    None,
    /// Independently sample one of the eight rotations/reflections per evaluation.
    Random,
}

#[derive(Clone, Debug, Args, Serialize)]
pub struct ModelArgs {
    /// Directory used to select or store checkpoints.
    #[arg(long, default_value = "checkpoints")]
    pub checkpoint_dir: PathBuf,

    /// Architecture assertion for existing snapshots; fresh training defaults to the board-mask model.
    #[arg(long, value_enum)]
    pub architecture: Option<ArchitectureChoice>,

    #[command(flatten)]
    pub device: DeviceArgs,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum ReplayLrSchedule {
    #[default]
    Constant,
    Cosine,
}

#[derive(Debug, Args, Serialize)]
pub struct TrainReplayArgs {
    #[command(flatten)]
    #[serde(flatten)]
    pub performance: TrainingPerformanceArgs,
    /// Individual numeric snapshot directory. Repeat to pool buffers; duplicate games are removed.
    #[arg(long, required = true)]
    pub replay_checkpoint_dir: Vec<PathBuf>,

    /// Independent output directory. Matching invocations resume completed epochs.
    #[arg(long)]
    pub run_dir: PathBuf,

    #[arg(long, value_enum, default_value_t = ArchitectureChoice::default())]
    pub architecture: ArchitectureChoice,

    #[command(flatten)]
    pub device: DeviceArgs,

    /// Total passes over the training games, with all eight board symmetries per pass.
    #[arg(long, default_value_t = 20)]
    pub epochs: usize,

    #[arg(long, default_value_t = 256)]
    pub training_batch_size: usize,

    #[arg(long, default_value_t = 0.001)]
    pub learning_rate: f64,

    /// Per-pass schedule; cosine includes the initial and final rates over the epoch budget.
    #[arg(long, value_enum, default_value_t = ReplayLrSchedule::Constant)]
    pub lr_schedule: ReplayLrSchedule,

    /// Required for cosine scheduling; the schedule horizon is fixed when the run is created.
    #[arg(long)]
    pub final_learning_rate: Option<f64>,

    /// Initialize only BatchNorm gamma to one after normal construction, preserving other weights/RNG.
    #[arg(long)]
    pub bn_gamma_one: bool,

    /// Separate fixed data-split seed when replicating model initialization and batch-order seeds.
    #[arg(long)]
    pub split_seed: Option<u64>,

    /// Write and validate initial state without optimizer updates (keeps the full schedule horizon).
    #[arg(long)]
    pub initialize_only: bool,

    #[arg(long, default_value_t = 0.0001)]
    pub weight_decay: f64,

    /// Fraction of game trajectory groups reserved for validation; never used for updates.
    #[arg(long, default_value_t = 0.1)]
    pub validation_fraction: f64,

    #[arg(long, default_value_t = 20260906)]
    pub seed: u64,

    /// Validate the source and print the split sizes without creating a run or training.
    #[arg(long)]
    pub inspect_only: bool,
}

#[derive(Debug, Args, Serialize)]
pub struct TrainArgs {
    /// Reconstruct early epochs from this self-play run's checkpoints and stats, without generating games.
    #[arg(long, requires = "replay_history_epochs")]
    pub replay_history_dir: Option<PathBuf>,

    /// Number of initial epochs to replay in order; later epochs generate new self-play games.
    #[arg(long, requires = "replay_history_dir")]
    pub replay_history_epochs: Option<usize>,
    /// Nucleus cutoff applied after temperature, only to move sampling. Preserve boundary ties.
    #[arg(long, default_value_t = 1.0)]
    pub top_p: f64,

    /// Initialize BatchNorm scales to one for fresh runs; resumed weights are preserved.
    #[arg(long)]
    pub bn_gamma_one: bool,

    /// LR = initial learning-rate * (initial replay capacity / current capacity)^exponent.
    #[arg(long, requires = "learning_rate", conflicts_with = "replay_games")]
    pub replay_lr_exponent: Option<f64>,

    #[command(flatten)]
    #[serde(flatten)]
    pub performance: TrainingPerformanceArgs,
    #[command(flatten)]
    pub model: ModelArgs,

    /// Directory for rendered sample games.
    #[arg(long, default_value = "games")]
    pub games_dir: PathBuf,

    /// Directory for epoch statistics.
    #[arg(long, default_value = "stats")]
    pub stats_dir: PathBuf,

    /// Total number of epochs to reach, including restored epochs. Omit to train indefinitely.
    #[arg(long)]
    pub epochs: Option<usize>,

    #[arg(long, default_value_t = 600)]
    pub games_per_epoch: usize,

    #[arg(long, default_value_t = 2048)]
    pub simulations: usize,

    #[arg(long, default_value_t = 1.0)]
    pub c_puct: f32,

    /// Legacy fixed game-count capacity; overrides the default position schedule.
    #[arg(long, conflicts_with_all = ["replay_positions", "replay_position_growth", "replay_growth_start_epoch"])]
    pub replay_games: Option<usize>,

    /// Initial base-position capacity (before symmetry augmentation): 600 * 25 * 2.5.
    #[arg(long, default_value_t = 37_500)]
    pub replay_positions: usize,

    /// Positions added per epoch after the initial plateau: 600 * 25 * 0.15.
    #[arg(long, default_value_t = 2_250)]
    pub replay_position_growth: usize,

    /// Number of completed epochs before the position capacity begins growing.
    #[arg(long, default_value_t = 15)]
    pub replay_growth_start_epoch: usize,

    #[arg(long, default_value_t = 128)]
    pub inference_batch_size: usize,

    /// Spatial symmetry applied before self-play network inference.
    #[arg(long, value_enum, default_value = "random")]
    pub inference_symmetry: InferenceSymmetryChoice,

    /// Maximum number of games that may perform self-play concurrently.
    #[arg(long, default_value_t = 160)]
    pub games_parallelism: usize,

    /// Maximum wait after the first queued request before dispatching a partial inference batch.
    #[arg(long, default_value_t = 100_000)]
    pub batch_timeout_us: u64,

    #[arg(long, default_value_t = 256)]
    pub training_batch_size: usize,

    /// Override the restored learning rate. New runs default to 0.001.
    #[arg(long)]
    pub learning_rate: Option<f64>,

    /// Override the restored weight decay. New runs default to 0.0001.
    #[arg(long)]
    pub weight_decay: Option<f64>,

    #[arg(long, default_value_t = 20)]
    pub rendered_games: usize,

    /// Base seed. Per-game and per-epoch streams are deterministically derived from it.
    #[arg(long, default_value_t = 0)]
    pub seed: u64,

    /// Print one self-play progress line per this many completed games. Zero disables progress.
    #[arg(long, default_value_t = 10)]
    pub progress_every_games: usize,

    /// Print a self-play heartbeat at this interval. Zero disables time-based heartbeats.
    #[arg(long, default_value_t = 60)]
    pub heartbeat_seconds: u64,
}

#[derive(Debug, Args)]
pub struct BenchmarkArgs {
    #[command(subcommand)]
    pub mode: BenchmarkMode,
}

#[derive(Debug, Subcommand)]
pub enum BenchmarkMode {
    /// Measure forward inference including host/device transfers and output synchronization.
    Inference(InferenceBenchmarkArgs),
    /// Measure training steps on synthetic inputs or a saved replay dataset.
    Training(TrainingBenchmarkArgs),
    /// Measure the complete MCTS self-play scheduler without training or checkpoints.
    SelfPlay(SelfPlayBenchmarkArgs),
}

#[derive(Debug, Args, Serialize)]
pub struct InferenceBenchmarkArgs {
    #[command(flatten)]
    pub device: DeviceArgs,

    #[arg(long, value_enum, default_value_t = ArchitectureChoice::default())]
    pub architecture: ArchitectureChoice,

    #[arg(long, default_value_t = 128)]
    pub batch_size: usize,

    #[arg(long, default_value_t = 20)]
    pub warmup_iterations: usize,

    #[arg(long, default_value_t = 100)]
    pub iterations: usize,

    #[arg(long, default_value_t = 0)]
    pub seed: u64,

    /// Optional path for the JSON result. The result is always printed to stdout.
    #[arg(long)]
    pub output: Option<PathBuf>,
}

#[derive(Debug, Args, Serialize)]
pub struct TrainingBenchmarkArgs {
    #[command(flatten)]
    #[serde(flatten)]
    pub performance: TrainingPerformanceArgs,

    /// Use real replay batches, with the same order across cache/optimizer backends.
    #[arg(long)]
    pub replay_checkpoint_dir: Option<PathBuf>,

    /// Adam's coupled weight decay. Production training uses 0.0001.
    #[arg(long, default_value_t = 0.0)]
    pub weight_decay: f64,
    #[command(flatten)]
    pub device: DeviceArgs,

    #[arg(long, value_enum, default_value_t = ArchitectureChoice::default())]
    pub architecture: ArchitectureChoice,

    #[arg(long, default_value_t = 1024)]
    pub batch_size: usize,

    #[arg(long, default_value_t = 5)]
    pub warmup_iterations: usize,

    #[arg(long, default_value_t = 20)]
    pub iterations: usize,

    #[arg(long, default_value_t = 0)]
    pub seed: u64,

    #[arg(long)]
    pub output: Option<PathBuf>,
}

#[derive(Debug, Args, Serialize)]
pub struct SelfPlayBenchmarkArgs {
    #[command(flatten)]
    pub device: DeviceArgs,

    #[arg(long, value_enum, default_value_t = ArchitectureChoice::default())]
    pub architecture: ArchitectureChoice,

    #[arg(long, default_value_t = 32)]
    pub games: usize,

    /// Full synthetic inference batches used to initialize CUDA and convolution algorithms.
    #[arg(long, default_value_t = 10)]
    pub warmup_batches: usize,

    #[arg(long, default_value_t = 256)]
    pub simulations: usize,

    #[arg(long, default_value_t = 1.0)]
    pub c_puct: f32,

    #[arg(long, default_value_t = 128)]
    pub inference_batch_size: usize,

    /// Spatial symmetry applied before self-play network inference.
    #[arg(long, value_enum, default_value = "random")]
    pub inference_symmetry: InferenceSymmetryChoice,

    #[arg(long, default_value_t = 160)]
    pub games_parallelism: usize,

    #[arg(long, default_value_t = 100_000)]
    pub batch_timeout_us: u64,

    #[arg(long, default_value_t = 0)]
    pub seed: u64,

    #[arg(long)]
    pub output: Option<PathBuf>,
}

#[derive(Debug, Args)]
pub struct PlayArgs {
    #[command(subcommand)]
    pub mode: PlayMode,
}

#[derive(Debug, Subcommand)]
pub enum PlayMode {
    /// Play against MCTS in the terminal.
    Human(HumanArgs),
    /// Let the raw network policy play both seats.
    Policy(PolicyArgs),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum HumanSeat {
    First,
    Second,
}

#[derive(Debug, Args)]
pub struct HumanArgs {
    #[command(flatten)]
    pub model: ModelArgs,

    #[arg(long, value_enum, default_value = "first")]
    pub human_seat: HumanSeat,

    #[arg(long, default_value_t = 4096)]
    pub simulations: usize,

    #[arg(long, default_value_t = 1.0)]
    pub c_puct: f32,

    #[arg(long, default_value_t = 0.33)]
    pub temperature: f32,
}

#[derive(Debug, Args)]
pub struct PolicyArgs {
    #[command(flatten)]
    pub model: ModelArgs,

    #[arg(long, default_value_t = 1.0)]
    pub temperature: f32,
}

#[derive(Debug, Args, Serialize)]
pub struct BattleArgs {
    /// Snapshot or run directory for the first checkpoint identity.
    #[arg(long)]
    pub first_checkpoint_dir: PathBuf,

    /// Snapshot or run directory for the second checkpoint identity.
    #[arg(long)]
    pub second_checkpoint_dir: PathBuf,

    #[command(flatten)]
    pub device: DeviceArgs,

    /// Number of games. Checkpoint seat assignments alternate between games.
    #[arg(long, default_value_t = 20)]
    pub games: usize,

    /// Maximum number of games evaluated concurrently.
    #[arg(long, default_value_t = 16)]
    pub games_parallelism: usize,

    /// Maximum inference batch size for each checkpoint.
    #[arg(long, default_value_t = 16)]
    pub inference_batch_size: usize,

    /// Maximum wait after the first request before dispatching a partial batch.
    #[arg(long, default_value_t = 1_000)]
    pub batch_timeout_us: u64,

    #[arg(long, default_value_t = 2048)]
    pub simulations: usize,

    #[arg(long, default_value_t = 1.0)]
    pub c_puct: f32,

    /// Default move-selection temperature for both checkpoints.
    #[arg(long, default_value_t = 0.0)]
    pub temperature: f32,

    /// Override move-selection temperature for the first checkpoint.
    #[arg(long)]
    pub first_temperature: Option<f32>,

    /// Override move-selection temperature for the second checkpoint.
    #[arg(long)]
    pub second_temperature: Option<f32>,

    /// Base seed used to derive independent deterministic streams for every game and model.
    #[arg(long, default_value_t = 0)]
    pub seed: u64,

    /// Print progress at this interval while games are running. Zero disables heartbeats.
    #[arg(long, default_value_t = 60)]
    pub heartbeat_seconds: u64,

    /// Suppress per-move logs. Per-game summaries are always printed.
    #[arg(long)]
    pub no_move_logs: bool,

    /// Optional JSON path for the complete aggregate report and game records.
    #[arg(long)]
    pub output: Option<PathBuf>,
}

#[cfg(test)]
mod tests {
    use super::{AdamBackendChoice, ReplayCacheMode};
    use clap::Parser;

    use super::{
        ArchitectureChoice, BenchmarkMode, Cli, Command, HumanSeat, InferenceSymmetryChoice,
        PlayMode,
    };

    #[test]
    fn parses_train_defaults() {
        let cli = Cli::try_parse_from(["alz", "train"]).unwrap();
        let Command::Train(args) = cli.command else {
            panic!("expected train command");
        };
        assert_eq!(args.games_per_epoch, 600);
        assert_eq!(args.replay_games, None);
        assert_eq!(args.replay_positions, 37_500);
        assert_eq!(args.replay_position_growth, 2_250);
        assert_eq!(args.replay_growth_start_epoch, 15);
        assert_eq!(args.simulations, 2048);
        assert_eq!(args.epochs, None);
        assert_eq!(args.learning_rate, None);
        assert_eq!(args.weight_decay, None);
        assert_eq!(args.inference_symmetry, InferenceSymmetryChoice::Random);
        assert_eq!(args.games_parallelism, 160);
        assert_eq!(args.batch_timeout_us, 100_000);
        assert_eq!(args.training_batch_size, 256);
        assert_eq!(args.heartbeat_seconds, 60);
        assert_eq!(args.performance.adam_backend, AdamBackendChoice::Standard);
        assert_eq!(args.performance.replay_cache, ReplayCacheMode::None);
        assert_eq!(args.performance.prefetch_batches, 0);
        // Leave this unset so resuming any existing architecture still works.
        assert_eq!(args.model.architecture, None);
    }

    #[test]
    fn replay_capacity_options_preserve_legacy_invocations_and_reject_ambiguity() {
        let cli = Cli::try_parse_from(["alz", "train", "--replay-games", "1800"]).unwrap();
        let Command::Train(args) = cli.command else {
            panic!("expected train command");
        };
        assert_eq!(args.replay_games, Some(1800));
        for option in [
            "--replay-positions",
            "--replay-position-growth",
            "--replay-growth-start-epoch",
        ] {
            assert!(
                Cli::try_parse_from(["alz", "train", "--replay-games", "1800", option, "10"])
                    .is_err()
            );
        }
    }

    #[test]
    fn fresh_network_commands_default_to_the_selected_architecture() {
        let selected = ArchitectureChoice::KataGeluBoardMaskValue64x2V1;
        assert_eq!(ArchitectureChoice::default(), selected);
        let cli = Cli::try_parse_from([
            "alz",
            "train-replay",
            "--replay-checkpoint-dir",
            "replays/00000069",
            "--run-dir",
            "new-run",
        ])
        .unwrap();
        let Command::TrainReplay(args) = cli.command else {
            panic!("expected replay training")
        };
        assert_eq!(args.architecture, selected);
        for mode in ["inference", "training", "self-play"] {
            let cli = Cli::try_parse_from(["alz", "benchmark", mode]).unwrap();
            let Command::Benchmark(args) = cli.command else {
                panic!("expected benchmark")
            };
            let architecture = match args.mode {
                super::BenchmarkMode::Inference(args) => args.architecture,
                super::BenchmarkMode::Training(args) => args.architecture,
                super::BenchmarkMode::SelfPlay(args) => args.architecture,
            };
            assert_eq!(architecture, selected);
        }
    }

    #[test]
    fn training_performance_options_require_cpu_cache_for_prefetch() {
        let cli = Cli::try_parse_from([
            "alz",
            "train",
            "--adam-backend",
            "fused",
            "--replay-cache",
            "cpu",
            "--prefetch-batches",
            "2",
        ])
        .unwrap();
        let Command::Train(mut args) = cli.command else {
            panic!("expected train")
        };
        assert_eq!(args.performance.adam_backend, AdamBackendChoice::Fused);
        args.performance.validate().unwrap();
        args.performance.replay_cache = ReplayCacheMode::Device;
        assert!(args.performance.validate().is_err());
        args.performance.replay_cache = ReplayCacheMode::Cpu;
        args.performance.prefetch_batches = 17;
        assert!(args.performance.validate().is_err());
    }

    #[test]
    fn parses_kata_architecture_for_new_training_run() {
        let cli = Cli::try_parse_from([
            "alz",
            "train",
            "--architecture",
            "kata-v1",
            "--inference-symmetry",
            "random",
        ])
        .unwrap();
        let Command::Train(args) = cli.command else {
            panic!("expected train command");
        };
        assert_eq!(args.model.architecture, Some(ArchitectureChoice::KataV1));
        assert_eq!(args.inference_symmetry, InferenceSymmetryChoice::Random);
    }

    #[test]
    fn parses_nested_human_command() {
        let cli = Cli::try_parse_from([
            "alz",
            "play",
            "human",
            "--human-seat",
            "second",
            "--simulations",
            "32",
        ])
        .unwrap();
        let Command::Play(play) = cli.command else {
            panic!("expected play command");
        };
        let PlayMode::Human(args) = play.mode else {
            panic!("expected human mode");
        };
        assert_eq!(args.human_seat, HumanSeat::Second);
        assert_eq!(args.simulations, 32);
    }

    #[test]
    fn battle_requires_both_snapshots() {
        assert!(Cli::try_parse_from(["alz", "battle"]).is_err());
        assert!(
            Cli::try_parse_from([
                "alz",
                "battle",
                "--first-checkpoint-dir",
                "first",
                "--second-checkpoint-dir",
                "second",
            ])
            .is_ok()
        );
    }

    #[test]
    fn parses_inference_benchmark() {
        let cli = Cli::try_parse_from([
            "alz",
            "benchmark",
            "inference",
            "--device",
            "cuda",
            "--batch-size",
            "256",
        ])
        .unwrap();
        let Command::Benchmark(args) = cli.command else {
            panic!("expected benchmark command");
        };
        let BenchmarkMode::Inference(args) = args.mode else {
            panic!("expected inference benchmark");
        };
        assert_eq!(args.batch_size, 256);
    }
}
