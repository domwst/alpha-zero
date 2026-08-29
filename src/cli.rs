use std::path::PathBuf;

use clap::{Args, Parser, Subcommand, ValueEnum};

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
}

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum DeviceChoice {
    Auto,
    Cpu,
    Mps,
    Cuda,
}

#[derive(Clone, Debug, Args)]
pub struct DeviceArgs {
    /// Compute device. Auto prefers MPS, then CUDA, then CPU.
    #[arg(long, value_enum, default_value = "auto")]
    pub device: DeviceChoice,

    /// CUDA device index when --device cuda is selected.
    #[arg(long, default_value_t = 0)]
    pub cuda_index: usize,
}

#[derive(Clone, Debug, Args)]
pub struct ModelArgs {
    /// Directory used to select or store checkpoints.
    #[arg(long, default_value = "checkpoints")]
    pub checkpoint_dir: PathBuf,

    #[command(flatten)]
    pub device: DeviceArgs,
}

#[derive(Debug, Args)]
pub struct TrainArgs {
    #[command(flatten)]
    pub model: ModelArgs,

    /// Directory for rendered sample games.
    #[arg(long, default_value = "games")]
    pub games_dir: PathBuf,

    /// Directory for epoch statistics.
    #[arg(long, default_value = "stats")]
    pub stats_dir: PathBuf,

    /// Stop after this many epochs. Omit to train indefinitely.
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

    #[arg(long, default_value_t = 32)]
    pub parallelism_padding: usize,

    #[arg(long, default_value_t = 100)]
    pub batch_accumulation_ms: u64,

    #[arg(long, default_value_t = 1024)]
    pub training_batch_size: usize,

    /// Override the restored learning rate. New runs default to 0.001.
    #[arg(long)]
    pub learning_rate: Option<f64>,

    /// Override the restored weight decay. New runs default to 0.0001.
    #[arg(long)]
    pub weight_decay: Option<f64>,

    #[arg(long, default_value_t = 20)]
    pub rendered_games: usize,

    #[arg(long, default_value_t = 16)]
    pub parallelism_step: usize,

    #[arg(long, default_value_t = 8)]
    pub parallelism_steps: usize,

    #[arg(long, default_value_t = 60)]
    pub parallelism_step_seconds: u64,
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

#[cfg(test)]
mod tests {
    use clap::Parser;

    use super::{Cli, Command, HumanSeat, PlayMode};

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
        assert!(Cli::try_parse_from([
            "alz",
            "battle",
            "--first-checkpoint-dir",
            "first",
            "--second-checkpoint-dir",
            "second",
        ])
        .is_ok());
    }
}
