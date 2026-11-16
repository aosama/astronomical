//! `decision_adapters.safetensors`: small add-on weights trained inside every
//! layer of a decision model, kept beside the original weights.
//!
//! Each add-on belongs to one of the model's linear layers (its target) and
//! is a pair of thin matrices: `down`, `[rank, inputs]`, and `up`,
//! `[outputs, rank]`. The layer then computes its shipped output plus
//! `up · (down · input)`. Training starts `up` at zero, so a fresh add-on
//! changes nothing, and any scale training applies is already folded into
//! `up`, so serving needs no other setting.
//!
//! The original 8-bit weights are never rewritten: merging an update into a
//! quantized weight re-rounds it and blurs exactly the small changes training
//! made. The add-ons stay in 32-bit floats and are applied as they are.
//!
//! The file names each tensor `<target>.adapter_down` or
//! `<target>.adapter_up`, where `<target>` is the shipped weight's prefix
//! (the name before `.weight`). Whether a target exists and its widths match
//! is checked by the engine that serves the model, because only it knows
//! the model's layers.

use thiserror::Error;

use crate::float32_safetensors::{
    Float32TensorsError, TunedTensor, float32_tensors_bytes, read_float32_tensors,
};

/// The trained add-ons of a decision model folder, beside its original weights.
pub const DECISION_ADAPTERS_FILE_NAME: &str = "decision_adapters.safetensors";
/// Laya's add-ons at rank 8 are about 14 MB; a 9-billion-weight model's are
/// a few hundred megabytes. A file far beyond that is not one of these.
pub const MAXIMUM_DECISION_ADAPTERS_FILE_BYTES: u64 = 1024 * 1024 * 1024;

const DOWN_TENSOR_SUFFIX: &str = ".adapter_down";
const UP_TENSOR_SUFFIX: &str = ".adapter_up";

/// One linear layer's add-on.
#[derive(Clone, Debug, PartialEq)]
pub struct LinearAdapter {
    /// The shipped weight's prefix, such as `layers.3.attn.Wqkv`.
    pub target: String,
    pub rank: usize,
    pub input_width: usize,
    pub output_width: usize,
    /// `[rank, input_width]`, row by row.
    pub down_values: Vec<f32>,
    /// `[output_width, rank]`, row by row, with the scale folded in.
    pub up_values: Vec<f32>,
}

/// Every add-on of one folder, in target name order.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DecisionAdapters {
    pub adapters: Vec<LinearAdapter>,
}

impl DecisionAdapters {
    /// Whether the folder has no add-ons, so it serves as shipped.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.adapters.is_empty()
    }
}

/// Why an add-on file cannot be used.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum DecisionAdaptersFileError {
    #[error("{0}")]
    Unreadable(Float32TensorsError),
    #[error("{0} is not named as an add-on's down or up matrix")]
    UnknownTensor(String),
    #[error("{0} is not a two-dimensional matrix")]
    NotAMatrix(String),
    #[error("the add-on for {0} lacks its down or its up matrix")]
    MissingHalf(String),
    #[error("the add-on for {0} has down and up matrices of different ranks")]
    RankMismatch(String),
}

/// Reads and checks an add-on file's bytes.
pub fn parse_decision_adapters(
    file_bytes: &[u8],
) -> Result<DecisionAdapters, DecisionAdaptersFileError> {
    let tensors =
        read_float32_tensors(file_bytes).map_err(DecisionAdaptersFileError::Unreadable)?;
    let mut adapters = Vec::with_capacity(tensors.len() / 2);
    let mut pending_down: Option<TunedTensor> = None;
    // In name order every target's `adapter_down` sorts directly before its
    // `adapter_up`, so the pairs are adjacent.
    for tuned_tensor in tensors {
        if tuned_tensor.shape.len() != 2 {
            return Err(DecisionAdaptersFileError::NotAMatrix(tuned_tensor.name));
        }
        if tuned_tensor.name.ends_with(DOWN_TENSOR_SUFFIX) {
            if let Some(unpaired_down) = pending_down.take() {
                return Err(missing_half(&unpaired_down.name));
            }
            if tuned_tensor.name.len() == DOWN_TENSOR_SUFFIX.len() {
                return Err(DecisionAdaptersFileError::UnknownTensor(tuned_tensor.name));
            }
            pending_down = Some(tuned_tensor);
            continue;
        }
        let Some(target) = tuned_tensor.name.strip_suffix(UP_TENSOR_SUFFIX) else {
            return Err(DecisionAdaptersFileError::UnknownTensor(tuned_tensor.name));
        };
        let target = target.to_owned();
        let Some(down_tensor) = pending_down.take() else {
            return Err(DecisionAdaptersFileError::MissingHalf(target));
        };
        adapters.push(paired_adapter(&target, down_tensor, tuned_tensor)?);
    }
    if let Some(unpaired_down) = pending_down {
        return Err(missing_half(&unpaired_down.name));
    }
    Ok(DecisionAdapters { adapters })
}

fn missing_half(down_tensor_name: &str) -> DecisionAdaptersFileError {
    let target = down_tensor_name
        .strip_suffix(DOWN_TENSOR_SUFFIX)
        .unwrap_or(down_tensor_name);
    DecisionAdaptersFileError::MissingHalf(target.to_owned())
}

fn paired_adapter(
    target: &str,
    down_tensor: TunedTensor,
    up_tensor: TunedTensor,
) -> Result<LinearAdapter, DecisionAdaptersFileError> {
    if down_tensor.name != format!("{target}{DOWN_TENSOR_SUFFIX}") {
        return Err(missing_half(&down_tensor.name));
    }
    let (rank, input_width) = (down_tensor.shape[0], down_tensor.shape[1]);
    let (output_width, up_rank) = (up_tensor.shape[0], up_tensor.shape[1]);
    if rank != up_rank || rank == 0 {
        return Err(DecisionAdaptersFileError::RankMismatch(target.to_owned()));
    }
    Ok(LinearAdapter {
        target: target.to_owned(),
        rank,
        input_width,
        output_width,
        down_values: down_tensor.values,
        up_values: up_tensor.values,
    })
}

/// The file's bytes for `decision_adapters`.
///
/// # Errors
/// Returns the serializer's reason when an add-on's values do not fill its shape.
pub fn decision_adapters_bytes(decision_adapters: &DecisionAdapters) -> Result<Vec<u8>, String> {
    let mut tensors = Vec::with_capacity(decision_adapters.adapters.len() * 2);
    for adapter in &decision_adapters.adapters {
        tensors.push(TunedTensor {
            name: format!("{}{DOWN_TENSOR_SUFFIX}", adapter.target),
            shape: vec![adapter.rank, adapter.input_width],
            values: adapter.down_values.clone(),
        });
        tensors.push(TunedTensor {
            name: format!("{}{UP_TENSOR_SUFFIX}", adapter.target),
            shape: vec![adapter.output_width, adapter.rank],
            values: adapter.up_values.clone(),
        });
    }
    float32_tensors_bytes(&tensors)
}

#[cfg(test)]
#[path = "decision_adapters_file_tests.rs"]
mod tests;
