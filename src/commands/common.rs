use std::path::Path;

use alz::{
    gomoku::{GomokuModel, ModelSpec},
    training_snapshot::{TrainingSnapshot, resolve_model_checkpoint},
};
use anyhow::{Context, Result, ensure};
use tch::{Cuda, Device, nn};

use crate::cli::{ArchitectureChoice, DeviceArgs, DeviceChoice};

pub fn validate_requested_architecture(
    requested: Option<ArchitectureChoice>,
    actual: &ModelSpec,
) -> Result<()> {
    if let Some(requested) = requested {
        let requested = ModelSpec::from(requested);
        ensure!(
            requested == *actual,
            "requested architecture {} does not match checkpoint architecture {}",
            requested.architecture_id(),
            actual.architecture_id()
        );
    }
    Ok(())
}

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
    tracing::info!(device = ?device, "using compute device");
    Ok(device)
}

pub fn load_network(
    snapshot_or_run_dir: &Path,
    requested_architecture: Option<ArchitectureChoice>,
    device: Device,
) -> Result<(nn::VarStore, GomokuModel, TrainingSnapshot)> {
    let snapshot = resolve_model_checkpoint(snapshot_or_run_dir)?.with_context(|| {
        format!(
            "no model checkpoint found in {}",
            snapshot_or_run_dir.display()
        )
    })?;
    validate_requested_architecture(requested_architecture, snapshot.model_spec())?;
    let mut var_store = nn::VarStore::new(device);
    let network = GomokuModel::new(var_store.root(), snapshot.model_spec());
    snapshot.load_model(&mut var_store)?;
    tracing::info!(
        snapshot_epoch = snapshot.epoch(),
        architecture = snapshot.model_spec().architecture_id(),
        path = %snapshot_or_run_dir.display(),
        "loaded snapshot"
    );
    Ok((var_store, network, snapshot))
}
