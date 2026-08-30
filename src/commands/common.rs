use std::path::Path;

use alz::tictactoe::TicTacToeResNet;
use anyhow::{Context, Result, ensure};
use tch::{Cuda, Device, nn};

use crate::{
    cli::{DeviceArgs, DeviceChoice},
    training_snapshot::resolve_snapshot,
};

pub fn resolve_device(args: &DeviceArgs) -> Result<Device> {
    let device = match args.device {
        DeviceChoice::Auto if tch::utils::has_mps() => Device::Mps,
        DeviceChoice::Auto => Device::cuda_if_available(),
        DeviceChoice::Cpu => Device::Cpu,
        DeviceChoice::Mps => {
            ensure!(tch::utils::has_mps(), "MPS is not available");
            Device::Mps
        }
        DeviceChoice::Cuda => {
            ensure!(Cuda::is_available(), "CUDA is not available");
            let device_count = Cuda::device_count() as usize;
            ensure!(
                args.cuda_index < device_count,
                "CUDA device index {} is outside the available range 0..{device_count}",
                args.cuda_index
            );
            Device::Cuda(args.cuda_index)
        }
    };
    eprintln!("Using device {device:?}");
    Ok(device)
}

pub fn load_network(
    snapshot_or_run_dir: &Path,
    device: Device,
) -> Result<(nn::VarStore, TicTacToeResNet, usize)> {
    let mut var_store = nn::VarStore::new(device);
    let network = TicTacToeResNet::new(var_store.root());
    let snapshot = resolve_snapshot(snapshot_or_run_dir)?.with_context(|| {
        format!(
            "no complete training snapshot found in {}",
            snapshot_or_run_dir.display()
        )
    })?;
    let epoch = snapshot.epoch();
    snapshot.load_model(&mut var_store)?;
    eprintln!(
        "Loaded snapshot {epoch} from {}",
        snapshot_or_run_dir.display()
    );
    Ok((var_store, network, epoch))
}
