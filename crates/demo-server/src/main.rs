mod protocol;
mod session;

use std::{
    future::Future,
    net::SocketAddr,
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    thread,
    time::Duration,
};

use alz::{
    engine::{NetworkBatchedExecutor, NetworkBatchedExecutorHandle, NetworkPositionEvaluator},
    gomoku::{GomokuCodec, GomokuModel},
    training_snapshot::resolve_model_checkpoint,
};
use anyhow::{Context, Result, ensure};
use axum::{
    Json, Router,
    extract::{State, WebSocketUpgrade},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::get,
};
use clap::{Parser, ValueEnum};
use protocol::CheckpointInfo;
use serde_json::json;
use session::{SessionConfig, run_websocket};
use tch::{Cuda, Device, Kind, nn};
use tokio::sync::{Semaphore, mpsc, watch};
use tower_http::{services::ServeDir, trace::TraceLayer};
use tracing_subscriber::EnvFilter;

const DEFAULT_MAX_SEARCH_SIMULATIONS: u32 = 10_000;
const DEFAULT_SEARCH_SIMULATIONS: u32 = 2_000;
const SHUTDOWN_GRACE_PERIOD: Duration = Duration::from_secs(5);

#[derive(Clone, Copy, Debug, ValueEnum)]
enum DeviceChoice {
    Auto,
    Cpu,
    Cuda,
    Mps,
}

#[derive(Debug, Parser)]
#[command(about = "Interactive AlphaZero Gomoku web demo")]
struct Args {
    /// Exact snapshot directory, or a run/checkpoint directory containing snapshots.
    #[arg(long)]
    checkpoint_dir: PathBuf,

    /// Pre-built frontend directory containing index.html.
    #[arg(long, default_value = "web/dist")]
    assets: PathBuf,

    #[arg(long, default_value = "127.0.0.1:8080")]
    listen: SocketAddr,

    #[arg(long, value_enum, default_value = "auto")]
    device: DeviceChoice,

    #[arg(long, default_value_t = 0)]
    cuda_index: usize,

    /// Hard per-position upper bound accepted from a browser.
    #[arg(long, default_value_t = DEFAULT_MAX_SEARCH_SIMULATIONS)]
    max_search_simulations: u32,

    #[arg(long, default_value_t = DEFAULT_SEARCH_SIMULATIONS)]
    default_search_simulations: u32,

    #[arg(long, default_value_t = 1.0)]
    c_puct: f32,

    #[arg(long, default_value_t = 32)]
    search_chunk_size: u32,

    #[arg(long, default_value_t = 100)]
    snapshot_interval_ms: u64,

    /// Maximum simultaneous browser sessions. Excess upgrades are rejected.
    #[arg(long, default_value_t = 4)]
    max_sessions: usize,

    #[arg(long, default_value_t = 16)]
    inference_batch_size: usize,

    #[arg(long, default_value_t = 1_000)]
    batch_timeout_us: u64,

    /// Randomly transform positions before inference and invert policies afterward.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    inference_symmetries: bool,

    #[arg(long, default_value_t = 0)]
    seed: u64,
}

#[derive(Clone)]
struct AppState {
    evaluator: NetworkBatchedExecutorHandle<GomokuModel>,
    session_config: SessionConfig,
    sessions: Arc<Semaphore>,
    next_seed: Arc<AtomicU64>,
    inference_symmetries: bool,
    shutdown: watch::Receiver<bool>,
}

#[derive(Clone, Copy, Debug)]
enum ShutdownReason {
    Interrupt,
    #[cfg(unix)]
    Terminate,
}

impl ShutdownReason {
    fn name(self) -> &'static str {
        match self {
            Self::Interrupt => "SIGINT",
            #[cfg(unix)]
            Self::Terminate => "SIGTERM",
        }
    }

    fn forced_exit_code(self) -> i32 {
        match self {
            Self::Interrupt => 130,
            #[cfg(unix)]
            Self::Terminate => 143,
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .with_target(false)
        .compact()
        .init();

    let args = Args::parse();
    validate_args(&args)?;
    ensure!(
        args.assets.join("index.html").is_file(),
        "frontend assets are missing from {}; run `npm --prefix web run build` first",
        args.assets.display()
    );

    let device = resolve_device(args.device, args.cuda_index)?;
    let snapshot = resolve_model_checkpoint(&args.checkpoint_dir)?.with_context(|| {
        format!(
            "no model checkpoint found in {}",
            args.checkpoint_dir.display()
        )
    })?;
    let descriptor = snapshot.descriptor();
    let mut var_store = nn::VarStore::new(device);
    let model = GomokuModel::new(var_store.root(), snapshot.model_spec());
    snapshot.load_model(&mut var_store)?;
    let checkpoint = CheckpointInfo {
        architecture: snapshot.model_spec().architecture_id().to_owned(),
        epoch: snapshot.epoch(),
        model_digest: descriptor.model_sha256.chars().take(12).collect(),
    };
    tracing::info!(
        device = ?device,
        architecture = %checkpoint.architecture,
        epoch = checkpoint.epoch,
        digest = %checkpoint.model_digest,
        "loaded demo checkpoint"
    );

    let executor =
        NetworkBatchedExecutor::new(model, args.max_sessions.max(args.inference_batch_size));
    let evaluator = executor.mint_handle();
    let (batch_commands, batch_command_receiver) = mpsc::channel(1);
    drop(batch_commands);
    let batch_size = args.inference_batch_size;
    let batch_timeout = Duration::from_micros(args.batch_timeout_us);
    let executor_task = tokio::spawn(async move {
        executor
            .serve(
                batch_size,
                batch_timeout,
                batch_command_receiver,
                (Kind::Float, device),
            )
            .await
    });

    let session_config = SessionConfig {
        checkpoint,
        compute_device: device_label(device),
        max_search_simulations: args.max_search_simulations,
        default_search_simulations: args.default_search_simulations,
        c_puct: args.c_puct,
        search_chunk_size: args.search_chunk_size,
        snapshot_interval: Duration::from_millis(args.snapshot_interval_ms),
    };
    session_config.validate()?;
    let (shutdown_sender, shutdown_receiver) = watch::channel(false);
    let shutdown_finished = Arc::new(AtomicBool::new(false));
    let state = AppState {
        evaluator,
        session_config,
        sessions: Arc::new(Semaphore::new(args.max_sessions)),
        next_seed: Arc::new(AtomicU64::new(args.seed)),
        inference_symmetries: args.inference_symmetries,
        shutdown: shutdown_receiver,
    };

    let app = Router::new()
        .route("/api/health", get(health))
        .route("/api/ws", get(websocket))
        .fallback_service(ServeDir::new(args.assets).append_index_html_on_directories(true))
        .layer(TraceLayer::new_for_http())
        .with_state(state);
    let listener = tokio::net::TcpListener::bind(args.listen)
        .await
        .with_context(|| format!("binding {}", args.listen))?;
    tracing::info!(listen = %args.listen, "AlphaZero Playground is ready");
    axum::serve(listener, app)
        .with_graceful_shutdown(begin_shutdown(shutdown_sender, shutdown_finished.clone()))
        .await
        .context("serving interactive demo")?;

    let (model, stats) = executor_task.await.context("joining network executor")?;
    tracing::info!(
        requests = stats.requests,
        batches = stats.invocations,
        average_batch = stats.average_batch_size(),
        "network executor stopped"
    );
    drop(model);
    drop(var_store);
    shutdown_finished.store(true, Ordering::Release);
    Ok(())
}

fn validate_args(args: &Args) -> Result<()> {
    ensure!(args.max_sessions > 0, "max sessions must be positive");
    ensure!(
        args.inference_batch_size > 0,
        "inference batch size must be positive"
    );
    ensure!(args.batch_timeout_us > 0, "batch timeout must be positive");
    Ok(())
}

fn resolve_device(choice: DeviceChoice, cuda_index: usize) -> Result<Device> {
    match choice {
        DeviceChoice::Auto if tch::utils::has_mps() => Ok(Device::Mps),
        DeviceChoice::Auto => Ok(Device::cuda_if_available()),
        DeviceChoice::Cpu => Ok(Device::Cpu),
        DeviceChoice::Mps => {
            ensure!(tch::utils::has_mps(), "MPS is not available");
            Ok(Device::Mps)
        }
        DeviceChoice::Cuda => {
            ensure!(Cuda::is_available(), "CUDA is not available");
            ensure!(
                cuda_index < Cuda::device_count() as usize,
                "CUDA device index {cuda_index} is outside the available range"
            );
            Ok(Device::Cuda(cuda_index))
        }
    }
}

fn device_label(device: Device) -> String {
    match device {
        Device::Cpu => "CPU".to_owned(),
        Device::Cuda(index) => format!("CUDA {index}"),
        Device::Mps => "MPS".to_owned(),
        Device::Vulkan => "Vulkan".to_owned(),
    }
}

async fn health() -> Json<serde_json::Value> {
    Json(json!({ "status": "ok" }))
}

async fn websocket(ws: WebSocketUpgrade, State(state): State<AppState>) -> Response {
    if *state.shutdown.borrow() {
        return (StatusCode::SERVICE_UNAVAILABLE, "server is shutting down").into_response();
    }
    let Ok(permit) = state.sessions.clone().try_acquire_owned() else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            "interactive session limit reached",
        )
            .into_response();
    };
    let seed = state.next_seed.fetch_add(1, Ordering::Relaxed);
    let evaluator = if state.inference_symmetries {
        NetworkPositionEvaluator::<GomokuModel, GomokuCodec>::with_random_symmetry(
            state.evaluator.clone(),
            seed,
        )
    } else {
        NetworkPositionEvaluator::<GomokuModel, GomokuCodec>::new(state.evaluator.clone())
    };
    let config = state.session_config.clone();
    let shutdown = state.shutdown.clone();
    ws.max_message_size(16 * 1024)
        .max_frame_size(16 * 1024)
        .on_upgrade(move |socket| {
            run_until_shutdown(
                run_websocket(socket, evaluator, config, seed, permit),
                shutdown,
            )
        })
}

async fn run_until_shutdown<F>(future: F, mut shutdown: watch::Receiver<bool>)
where
    F: Future<Output = ()>,
{
    tokio::pin!(future);
    tokio::select! {
        biased;
        _ = shutdown.wait_for(|requested| *requested) => {
            tracing::debug!("closing interactive session for server shutdown");
        }
        () = &mut future => {}
    }
}

async fn begin_shutdown(shutdown: watch::Sender<bool>, shutdown_finished: Arc<AtomicBool>) {
    let reason = match wait_for_shutdown_signal().await {
        Ok(reason) => reason,
        Err(error) => {
            tracing::error!(error = ?error, "failed to install shutdown signal handlers");
            return;
        }
    };
    tracing::info!(
        signal = reason.name(),
        "shutdown requested; closing active sessions"
    );
    let _ = shutdown.send(true);
    arm_shutdown_watchdog(reason, shutdown_finished);

    tokio::spawn(async move {
        match wait_for_shutdown_signal().await {
            Ok(second) => {
                tracing::warn!(
                    signal = second.name(),
                    "second shutdown signal received; exiting immediately"
                );
                std::process::exit(reason.forced_exit_code());
            }
            Err(error) => {
                tracing::warn!(error = ?error, "failed to await a second shutdown signal");
            }
        }
    });
}

fn arm_shutdown_watchdog(reason: ShutdownReason, shutdown_finished: Arc<AtomicBool>) {
    let result = thread::Builder::new()
        .name("demo-shutdown-watchdog".to_owned())
        .spawn(move || {
            thread::sleep(SHUTDOWN_GRACE_PERIOD);
            if !shutdown_finished.load(Ordering::Acquire) {
                tracing::error!(
                    timeout_seconds = SHUTDOWN_GRACE_PERIOD.as_secs(),
                    "graceful shutdown timed out; exiting immediately"
                );
                std::process::exit(reason.forced_exit_code());
            }
        });
    if let Err(error) = result {
        tracing::warn!(error = ?error, "failed to start shutdown watchdog");
    }
}

async fn wait_for_shutdown_signal() -> Result<ShutdownReason> {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};

        let mut terminate =
            signal(SignalKind::terminate()).context("installing SIGTERM handler")?;
        tokio::select! {
            result = tokio::signal::ctrl_c() => {
                result.context("installing Ctrl-C handler")?;
                Ok(ShutdownReason::Interrupt)
            }
            received = terminate.recv() => {
                ensure!(received.is_some(), "SIGTERM handler stopped unexpectedly");
                Ok(ShutdownReason::Terminate)
            }
        }
    }

    #[cfg(not(unix))]
    {
        tokio::signal::ctrl_c()
            .await
            .context("installing Ctrl-C handler")?;
        Ok(ShutdownReason::Interrupt)
    }
}

#[cfg(test)]
mod tests {
    use std::future::pending;

    use tokio::sync::oneshot;

    use super::*;

    struct DropMarker(Arc<AtomicBool>);

    impl Drop for DropMarker {
        fn drop(&mut self) {
            self.0.store(true, Ordering::Release);
        }
    }

    #[tokio::test]
    async fn shutdown_cancels_active_connection_tasks() {
        let (shutdown_sender, shutdown_receiver) = watch::channel(false);
        let (started_sender, started_receiver) = oneshot::channel();
        let dropped = Arc::new(AtomicBool::new(false));
        let dropped_in_task = dropped.clone();
        let task = tokio::spawn(run_until_shutdown(
            async move {
                let _marker = DropMarker(dropped_in_task);
                let _ = started_sender.send(());
                pending::<()>().await;
            },
            shutdown_receiver,
        ));

        started_receiver.await.unwrap();
        shutdown_sender.send(true).unwrap();
        tokio::time::timeout(Duration::from_secs(1), task)
            .await
            .expect("active task did not stop after shutdown")
            .unwrap();
        assert!(dropped.load(Ordering::Acquire));
    }

    #[test]
    fn compute_device_labels_are_user_facing() {
        assert_eq!(device_label(Device::Cpu), "CPU");
        assert_eq!(device_label(Device::Cuda(2)), "CUDA 2");
        assert_eq!(device_label(Device::Mps), "MPS");
    }
}
