//! Optional analysis-only tensors. Semantic axes also accommodate future attention maps.
use anyhow::Result;
use serde::Serialize;
use tch::{Device, Tensor};

#[derive(Debug, Serialize)]
pub struct ActivationMap {
    pub name: String,
    pub shape: Vec<i64>,
    pub axes: Vec<String>,
    pub values: Vec<f32>,
}
impl ActivationMap {
    pub fn capture(name: String, tensor: &Tensor) -> Result<Self> {
        let mut map = Self::describe(name, tensor);
        map.values = Vec::<f32>::try_from(tensor.detach().to(Device::Cpu).flatten(0, -1))?;
        Ok(map)
    }
    pub fn describe(name: String, tensor: &Tensor) -> Self {
        let shape = tensor.size();
        let axes: Vec<String> = match shape.len() {
            4 => vec!["batch", "channel", "row", "column"]
                .into_iter()
                .map(String::from)
                .collect(),
            3 => vec!["batch", "row", "column"]
                .into_iter()
                .map(String::from)
                .collect(),
            2 => vec!["batch", "feature"]
                .into_iter()
                .map(String::from)
                .collect(),
            _ => (0..shape.len()).map(|i| format!("axis_{i}")).collect(),
        };
        Self {
            name,
            shape,
            axes,
            values: vec![],
        }
    }
}
