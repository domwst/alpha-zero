//! Deployment smoke check: selected trained weights agree across CPU and CUDA.
use alz::{engine::AlphaZeroNet, gomoku::GomokuModel, training_snapshot::resolve_model_checkpoint};
use tch::{Device, Kind, Tensor, nn};

fn main() -> anyhow::Result<()> {
    use clap::Parser;
    #[derive(Parser)]
    struct Args {
        /// Directory containing checkpoint directories to compare on CPU and CUDA.
        #[arg(long)]
        models_dir: std::path::PathBuf,
        /// Optionally verify the number of checkpoint directories supplied.
        #[arg(long)]
        expected_models: Option<usize>,
    }
    let args = Args::parse();
    anyhow::ensure!(tch::Cuda::is_available(), "CUDA unavailable");
    let _guard = tch::no_grad_guard();
    tch::manual_seed(20260908);
    let mut checked = 0;
    for entry in std::fs::read_dir(&args.models_dir)? {
        let path = entry?.path();
        if !path.is_dir() {
            continue;
        }
        let snapshot = resolve_model_checkpoint(&path)?.expect("model checkpoint");
        let mut cpu_vs = nn::VarStore::new(Device::Cpu);
        let cpu = GomokuModel::new(cpu_vs.root(), snapshot.model_spec());
        snapshot.load_model(&mut cpu_vs)?;
        let mut cuda_vs = nn::VarStore::new(Device::Cuda(0));
        let cuda = GomokuModel::new(cuda_vs.root(), snapshot.model_spec());
        snapshot.load_model(&mut cuda_vs)?;
        for batch in [1, 4, 64] {
            let input = Tensor::randint(3, [batch, 1, 19, 19], (Kind::Int64, Device::Cpu));
            let input = Tensor::cat(&[input.eq(1), input.eq(2)], 1).to_kind(Kind::Float);
            let expected = cpu.forward_t(&input, false);
            let actual = cuda.forward_t(&input.to_device(Device::Cuda(0)), false);
            for (head, got, wanted) in [
                ("policy", actual.policy_logits, expected.policy_logits),
                ("value", actual.values, expected.values),
            ] {
                let got = got.to_device(Device::Cpu);
                let error = f64::try_from((&got - &wanted).abs().max())?;
                println!(
                    "{} batch={batch} head={head} max_error={error}",
                    path.display()
                );
                anyhow::ensure!(
                    got.allclose(&wanted, 2e-3, 2e-3, false),
                    "CPU/CUDA mismatch"
                );
            }
        }
        checked += 1;
    }
    anyhow::ensure!(checked > 0, "No model checkpoints found");
    if let Some(expected) = args.expected_models {
        anyhow::ensure!(
            checked == expected,
            "Expected {expected} models, found {checked}"
        );
    }
    Ok(())
}
