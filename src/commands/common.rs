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

pub fn resolve_training_architecture(
    requested: Option<ArchitectureChoice>,
    existing: Option<&ModelSpec>,
) -> Result<ModelSpec> {
    if let Some(existing) = existing {
        validate_requested_architecture(requested, existing)?;
        Ok(existing.clone())
    } else {
        Ok(requested.unwrap_or_default().into())
    }
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

pub(super) fn build_adam(
    backend: crate::cli::AdamBackendChoice,
    variables: &tch::nn::VarStore,
    learning_rate: f64,
    weight_decay: f64,
) -> anyhow::Result<tch::nn::Optimizer> {
    use tch::nn::OptimizerConfig;
    let config = tch::nn::Adam::default().wd(weight_decay);
    Ok(match backend {
        crate::cli::AdamBackendChoice::Standard => config.build(variables, learning_rate)?,
        crate::cli::AdamBackendChoice::Fused => {
            anyhow::ensure!(
                matches!(variables.device(), tch::Device::Cpu | tch::Device::Cuda(_)),
                "fused Adam requires a CPU or CUDA device"
            );
            config.fused().build(variables, learning_rate)?
        }
    })
}

#[cfg(test)]
mod tests {
    use clap::ValueEnum;

    use super::*;

    #[test]
    fn architecture_default_applies_only_to_fresh_training() {
        assert_eq!(
            resolve_training_architecture(None, None).unwrap(),
            ModelSpec::KataGeluBoardMaskValue64x2V1
        );
        for choice in ArchitectureChoice::value_variants() {
            let spec = ModelSpec::from(*choice);
            assert_eq!(
                resolve_training_architecture(Some(*choice), None).unwrap(),
                spec
            );
            assert_eq!(
                resolve_training_architecture(None, Some(&spec)).unwrap(),
                spec
            );
            assert_eq!(
                resolve_training_architecture(Some(*choice), Some(&spec)).unwrap(),
                spec
            );
        }
        assert!(
            resolve_training_architecture(
                Some(ArchitectureChoice::KataGeluValue64x2V1),
                Some(&ModelSpec::KataV1)
            )
            .is_err()
        );
    }
}
