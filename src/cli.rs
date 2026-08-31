use std::path::PathBuf;

use alz::gomoku::ModelSpec;
use clap::{Args, Parser, Subcommand, ValueEnum};
use serde::Serialize;

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
    /// Play a game using the latest complete snapshot.
    Play(PlayArgs),
    /// Evaluate two snapshots against each other.
    Battle(BattleArgs),
    /// Measure inference, training, or self-play throughput without creating checkpoints.
    Benchmark(BenchmarkArgs),
    /// Inspect or migrate checkpoint metadata.
    Checkpoint(CheckpointArgs),
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

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum ArchitectureChoice {
    #[value(name = "legacy-resnet-v1")]
    #[serde(rename = "legacy_resnet_v1")]
    LegacyResNetV1,
}

impl From<ArchitectureChoice> for ModelSpec {
    fn from(value: ArchitectureChoice) -> Self {
        match value {
            ArchitectureChoice::LegacyResNetV1 => Self::LegacyResNetV1,
        }
    }
}

#[derive(Clone, Debug, Args, Serialize)]
pub struct ModelArgs {
    /// Directory used to select or store checkpoints.
    #[arg(long, default_value = "checkpoints")]
    pub checkpoint_dir: PathBuf,

    /// Expected architecture. Existing snapshots infer it when this is omitted.
    #[arg(long, value_enum)]
    pub architecture: Option<ArchitectureChoice>,

    #[command(flatten)]
    pub device: DeviceArgs,
}

#[derive(Debug, Args, Serialize)]
pub struct TrainArgs {
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

    #[arg(long, default_value_t = 1800)]
    pub replay_games: usize,

    #[arg(long, default_value_t = 128)]
    pub inference_batch_size: usize,

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
    /// Measure optimizer training steps on a fixed synthetic batch.
    Training(TrainingBenchmarkArgs),
    /// Measure the complete MCTS self-play scheduler without training or checkpoints.
    SelfPlay(SelfPlayBenchmarkArgs),
}

#[derive(Debug, Args, Serialize)]
pub struct InferenceBenchmarkArgs {
    #[command(flatten)]
    pub device: DeviceArgs,

    #[arg(long, value_enum, default_value = "legacy-resnet-v1")]
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
    pub device: DeviceArgs,

    #[arg(long, value_enum, default_value = "legacy-resnet-v1")]
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

    #[arg(long, value_enum, default_value = "legacy-resnet-v1")]
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

#[derive(Debug, Args)]
pub struct BattleArgs {
    /// Snapshot or run directory for the first seat.
    #[arg(long)]
    pub first_checkpoint_dir: PathBuf,

    /// Snapshot or run directory for the second seat.
    #[arg(long)]
    pub second_checkpoint_dir: PathBuf,

    #[command(flatten)]
    pub device: DeviceArgs,

    #[arg(long, default_value_t = 2048)]
    pub simulations: usize,

    #[arg(long, default_value_t = 1.0)]
    pub c_puct: f32,

    /// Move-selection temperature. Zero selects the most visited move.
    #[arg(long, default_value_t = 0.0)]
    pub temperature: f32,
}

#[derive(Debug, Args)]
pub struct CheckpointArgs {
    #[command(subcommand)]
    pub mode: CheckpointMode,
}

#[derive(Debug, Subcommand)]
pub enum CheckpointMode {
    /// Add architecture and tensor-schema metadata to version-1 checkpoints.
    MigrateV1(MigrateV1Args),
}

#[derive(Debug, Args)]
pub struct MigrateV1Args {
    /// Run directory containing numeric checkpoint directories.
    #[arg(long, default_value = "checkpoints")]
    pub checkpoint_dir: PathBuf,

    /// Replace metadata files after validating every checkpoint. Without this flag, only inspect.
    #[arg(long)]
    pub apply: bool,
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use clap::Parser;

    use super::{BenchmarkMode, CheckpointMode, Cli, Command, HumanSeat, PlayMode};

    #[test]
    fn parses_train_defaults() {
        let cli = Cli::try_parse_from(["alz", "train"]).unwrap();
        let Command::Train(args) = cli.command else {
            panic!("expected train command");
        };
        assert_eq!(args.games_per_epoch, 600);
        assert_eq!(args.simulations, 2048);
        assert_eq!(args.epochs, None);
        assert_eq!(args.learning_rate, None);
        assert_eq!(args.weight_decay, None);
        assert_eq!(args.games_parallelism, 160);
        assert_eq!(args.batch_timeout_us, 100_000);
        assert_eq!(args.training_batch_size, 256);
        assert_eq!(args.heartbeat_seconds, 60);
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

    #[test]
    fn checkpoint_migration_is_dry_run_by_default() {
        let cli = Cli::try_parse_from([
            "alz",
            "checkpoint",
            "migrate-v1",
            "--checkpoint-dir",
            "copied-checkpoints",
        ])
        .unwrap();
        let Command::Checkpoint(args) = cli.command else {
            panic!("expected checkpoint command");
        };
        let CheckpointMode::MigrateV1(args) = args.mode;
        assert_eq!(args.checkpoint_dir, PathBuf::from("copied-checkpoints"));
        assert!(!args.apply);
    }
}
