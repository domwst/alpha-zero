//! Repeatable producer lifecycle/batch-grid workload, including real CUDA output parity.
use alz::{
    engine::{AlphaZeroNet, BatcherCommand, NetworkBatchedExecutor, masked_policy_probabilities},
    gomoku::GomokuModel,
    training_snapshot::resolve_model_checkpoint,
};
use anyhow::{Context, Result, ensure};
use clap::Args;
use serde_json::json;
use std::{
    path::PathBuf,
    time::{Duration, Instant},
};
use tch::{Device, Kind, Tensor, nn};

#[derive(Debug, Args, serde::Serialize)]
pub struct ExecutorBenchmarkArgs {
    #[arg(long)]
    checkpoint: PathBuf,
    #[command(flatten)]
    device: crate::cli::DeviceArgs,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    #[arg(long, default_value_t = 400)]
    producers: usize,
    #[arg(long, default_value_t = 256)]
    batch_size: usize,
    #[arg(long, default_value_t = 1000)]
    timeout_us: u64,
    #[arg(long, default_value_t = 256)]
    requests: usize,
    #[arg(long, value_delimiter = ',')]
    grid: Vec<usize>,
    /// Unequal request counts create a declining-concurrency tail.
    #[arg(long)]
    tail: bool,
    /// Reproduce the old 5/6 task-count batch reduction inside this otherwise identical workload.
    #[arg(long)]
    task_count_baseline: bool,
    /// Drop and reacquire the activity guard between requests.
    #[arg(long)]
    reacquire: bool,
    /// Collect allocator counters in a separate profiling run (requires the local probe).
    #[arg(long)]
    profile_allocator: bool,
    #[arg(long)]
    output: PathBuf,
}
fn inputs(index: usize) -> (Tensor, Tensor) {
    let mut board = vec![0f32; 722];
    let mut legal = vec![1i64; 361];
    // Deterministic varied inputs with masked occupied points; not random noise.
    for ply in 0..(index % 24) {
        let action = (ply * 37 + index * 13) % 361;
        board[(ply % 2) * 361 + action] = 1.;
        legal[action] = 0;
    }
    (
        Tensor::from_slice(&board).view([2, 19, 19]),
        Tensor::from_slice(&legal)
            .view([19, 19])
            .to_kind(Kind::Bool),
    )
}
pub async fn run(args: ExecutorBenchmarkArgs, batch_grid: &[usize]) -> Result<()> {
    ensure!(
        args.producers > 0
            && args.producers <= 4096
            && args.requests > 0
            && args.requests <= 100000
    );
    ensure!(args.batch_size > 0 && args.batch_size <= 4096);
    ensure!(
        args.producers
            .checked_mul(args.requests)
            .is_some_and(|n| n <= 4_000_000),
        "benchmark timing storage is limited to four million requests"
    );
    tch::set_num_threads(1);
    let device = super::common::resolve_device(&args.device)?;
    alz::engine::telemetry::event("configured", serde_json::to_value(&args)?)?;
    let snapshot = resolve_model_checkpoint(&args.checkpoint)?.context("checkpoint missing")?;
    let mut store = nn::VarStore::new(device);
    let network = GomokuModel::new(store.root(), snapshot.model_spec());
    snapshot.load_model(&mut store)?;
    // Single-row reference catches row/mask mapping errors independently of batching.
    let mut references = Vec::new();
    for index in 0..args.producers {
        let (input, mask) = inputs(index.wrapping_add(args.seed as usize));
        let out = tch::no_grad(|| network.forward_t(&input.unsqueeze(0).to(device), false));
        let policy = masked_policy_probabilities(&out.policy_logits, &mask.unsqueeze(0).to(device))
            .to(Device::Cpu)
            .squeeze_dim(0);
        references.push((out.values.to(Device::Cpu).squeeze_dim(0), policy));
    }
    // Equal maximum-shape warmup in every arm; initialization is excluded from throughput.
    let (input, _) = inputs(args.seed as usize);
    let input = input
        .unsqueeze(0)
        .repeat([args.batch_size as i64, 1, 1, 1])
        .to(device);
    for _ in 0..10 {
        let output = tch::no_grad(|| network.forward_t(&input, false));
        let _ = output.values.to(Device::Cpu);
    }
    drop(input);
    let mut executor = NetworkBatchedExecutor::new(network, args.producers * 2)
        .with_active_producer_dispatch(!args.task_count_baseline);
    let grid = if args.grid.is_empty() {
        batch_grid
    } else {
        &args.grid
    };
    executor = executor.with_batch_grid(grid.to_vec())?;
    let handles = (0..args.producers)
        .map(|_| executor.mint_handle())
        .collect::<Vec<_>>();
    let (commands, rx) = tokio::sync::mpsc::channel(1);
    let batch = args.batch_size;
    let timeout = args.timeout_us;
    let server = tokio::spawn(executor.serve(
        batch,
        Duration::from_micros(timeout),
        rx,
        (Kind::Float, device),
    ));
    let barrier = std::sync::Arc::new(tokio::sync::Barrier::new(args.producers));
    let started = Instant::now();
    let mut jobs = tokio::task::JoinSet::new();
    for (index, (mut handle, (value, policy))) in handles.into_iter().zip(references).enumerate() {
        let barrier = barrier.clone();
        let n = if args.tail {
            (args.requests * (index + 1) / args.producers).max(1)
        } else {
            args.requests
        };
        let reacquire = args.reacquire;
        let seed = args.seed;
        jobs.spawn(async move {
            let (input, mask) = inputs(index.wrapping_add(seed as usize));
            let mut guard = handle.submission();
            barrier.wait().await;
            let mut timings = Vec::with_capacity(n);
            let mut max_value_error = 0f64;
            let mut max_policy_error = 0f64;
            for step in 0..n {
                if reacquire && step > 0 {
                    drop(guard);
                    tokio::task::yield_now().await;
                    guard = handle.submission();
                }
                let now = Instant::now();
                let (v, p) = guard
                    .execute(input.shallow_clone(), mask.shallow_clone())
                    .await?;
                timings.push(now.elapsed().as_micros() as u64);
                // Check first and last output per producer; exclude validation from latency.
                if step == 0 || step == n - 1 {
                    max_value_error =
                        max_value_error.max(f64::try_from((&v - &value).abs().max())?);
                    max_policy_error =
                        max_policy_error.max(f64::try_from((&p - &policy).abs().max())?);
                    ensure!(
                        bool::try_from(p.isfinite().all())? && bool::try_from(v.isfinite().all())?
                    );
                    ensure!(
                        f64::try_from(p.masked_select(&mask.logical_not()).abs().sum(Kind::Float))?
                            == 0.,
                        "illegal output probability"
                    );
                }
            }
            Ok::<_, anyhow::Error>((timings, max_value_error, max_policy_error))
        });
    }
    let mut timings = Vec::new();
    let mut remaining = args.producers;
    let mut legacy_batch = batch;
    let mut value_error = 0f64;
    let mut policy_error = 0f64;
    while let Some(result) = jobs.join_next().await {
        let (t, v, p) = result??;
        timings.extend(t);
        value_error = value_error.max(v);
        policy_error = policy_error.max(p);
        remaining -= 1;
        if args.task_count_baseline && remaining > 0 && remaining < legacy_batch && legacy_batch > 1
        {
            legacy_batch = (legacy_batch * 5 / 6).max(1);
            // Other tasks may already have finished before JoinSet yields their results.
            // A closed command channel then means the executor has drained normally;
            // its join below still reports an executor failure.
            let _ = commands
                .send(BatcherCommand::SetBatchSize(legacy_batch))
                .await;
        }
    }
    drop(commands);
    let (_, stats) = tokio::time::timeout(Duration::from_secs(30), server).await??;
    let seconds = started.elapsed().as_secs_f64();
    timings.sort_unstable();
    let quantile = |q: f64| timings[((timings.len() - 1) as f64 * q).round() as usize];
    let passed = value_error <= 2e-4 && policy_error <= 2e-5;
    let result = json!({"schema_version":1,"benchmark":"executor","model_sha256":snapshot.descriptor().model_sha256,
        "config":{"seed":args.seed,"warmup_maximum_batches":10,"task_count_baseline":args.task_count_baseline,"device":args.device,"producers":args.producers,"batch_size":batch,"timeout_us":timeout,"requests_per_producer":args.requests,"grid":args.grid,"tail":args.tail,"reacquire":args.reacquire},
        "duration_seconds":seconds,"requests":timings.len(),"requests_per_second":timings.len() as f64/seconds,
        "latency_us":{"p50":quantile(0.5),"p95":quantile(0.95),"p99":quantile(0.99),"max":timings.last()},
        "parity":{"passed":passed,"maximum_value_error":value_error,"maximum_policy_error":policy_error,"value_tolerance":2e-4,"policy_tolerance":2e-5},"network":stats});
    std::fs::write(&args.output, serde_json::to_vec_pretty(&result)?)?;
    if args.profile_allocator {
        ensure!(
            matches!(device, Device::Cuda(_)),
            "allocator profiling requires CUDA"
        );
        allocator_snapshot(&args.output.with_extension("allocator.json"))?;
    }
    alz::engine::telemetry::event("benchmark_completed", result.clone())?;
    tracing::info!(result=%result,"BENCHMARK_RESULT");
    ensure!(passed, "output parity failed");
    Ok(())
}

/// Deliberately opt-in: allocator introspection stays out of normal throughput runs.
#[cfg(target_os = "linux")]
fn allocator_snapshot(output: &std::path::Path) -> Result<()> {
    use std::ffi::{CString, c_char, c_int, c_void};
    unsafe extern "C" {
        fn dlopen(path: *const c_char, flags: c_int) -> *mut c_void;
        fn dlsym(handle: *mut c_void, name: *const c_char) -> *mut c_void;
    }
    let path = std::env::var_os("ALZ_ALLOCATOR_PROBE")
        .context("ALZ_ALLOCATOR_PROBE must name the trusted local probe")?;
    let library = CString::new(path.as_encoded_bytes())?;
    let destination = CString::new(output.as_os_str().as_encoded_bytes())?;
    // Trusted local benchmark-only library; never accepted by the job-service API.
    unsafe {
        let handle = dlopen(library.as_ptr(), 2);
        ensure!(!handle.is_null(), "loading allocator probe failed");
        let symbol = dlsym(handle, c"alz_dump_cuda_allocator_stats".as_ptr());
        ensure!(!symbol.is_null(), "allocator probe symbol missing");
        let dump: unsafe extern "C" fn(*const c_char) -> c_int = std::mem::transmute(symbol);
        ensure!(dump(destination.as_ptr()) == 0, "allocator probe failed");
    }
    Ok(())
}
#[cfg(not(target_os = "linux"))]
fn allocator_snapshot(_: &std::path::Path) -> Result<()> {
    Ok(())
}
