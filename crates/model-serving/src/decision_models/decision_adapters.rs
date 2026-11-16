//! Reads a decision model folder's `decision_adapters.safetensors`, the
//! add-on weights a whole-model training writes beside the original weights,
//! and checks every add-on against the layer it belongs to.
//!
//! A folder without the file serves its shipped layers exactly as before. A
//! folder whose file cannot be used refuses to load: serving the shipped
//! model while the person believes their training is in force would be a
//! silent wrong answer.

use std::path::Path;

use astronomical_decision_contract::DecisionFailure;
use astronomical_decision_contract::decision_adapters_file::{
    DECISION_ADAPTERS_FILE_NAME, DecisionAdapters, MAXIMUM_DECISION_ADAPTERS_FILE_BYTES,
    parse_decision_adapters,
};
use astronomical_runtime_integration::MlxSafetensors;

/// A linear layer that may carry an add-on, with its widths.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct AdapterTarget {
    /// The shipped weight's prefix, the name before `.weight`.
    pub(crate) name: String,
    pub(crate) input_width: usize,
    pub(crate) output_width: usize,
}

impl AdapterTarget {
    /// The target of an affine-quantized linear layer, its widths read from
    /// the stored tensors: `weight` has one row per output and `scales` one
    /// column per group of inputs. Only the names are looked up; no weight
    /// bytes are read.
    pub(crate) fn of_quantized_linear(
        tensors: &MlxSafetensors,
        name: String,
        quantization_group_size: usize,
    ) -> Result<Self, DecisionFailure> {
        let (Ok(packed_weight), Ok(scales)) = (
            tensors.tensor(&format!("{name}.weight")),
            tensors.tensor(&format!("{name}.scales")),
        ) else {
            return Err(DecisionFailure::fatal_execution(&format!(
                "the model has no quantized layer {name}"
            )));
        };
        let (weight_shape, scales_shape) = (packed_weight.shape(), scales.shape());
        if weight_shape.len() != 2 || scales_shape.len() != 2 {
            return Err(DecisionFailure::fatal_execution(&format!(
                "the quantized layer {name} is not a matrix"
            )));
        }
        Ok(Self {
            output_width: weight_shape[0] as usize,
            input_width: scales_shape[1] as usize * quantization_group_size,
            name,
        })
    }
}

/// The folder's add-ons, or none when the folder has no file.
pub(crate) fn read_decision_adapters(
    model_directory: &Path,
) -> Result<DecisionAdapters, DecisionFailure> {
    let adapters_path = model_directory.join(DECISION_ADAPTERS_FILE_NAME);
    let adapters_size = match std::fs::metadata(&adapters_path) {
        Ok(adapters_metadata) => adapters_metadata.len(),
        Err(metadata_error) if metadata_error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(DecisionAdapters::default());
        }
        Err(_) => return Err(unusable_adapters("it cannot be read")),
    };
    if adapters_size > MAXIMUM_DECISION_ADAPTERS_FILE_BYTES {
        return Err(unusable_adapters("it is too large"));
    }
    let Ok(adapters_bytes) = std::fs::read(&adapters_path) else {
        return Err(unusable_adapters("it cannot be read"));
    };
    match parse_decision_adapters(&adapters_bytes) {
        Ok(decision_adapters) => Ok(decision_adapters),
        Err(adapters_error) => Err(unusable_adapters(&adapters_error.to_string())),
    }
}

/// Checks that every add-on belongs to one of `targets` and has its widths.
/// Any difference means the file was trained for another model.
pub(crate) fn check_adapter_targets(
    decision_adapters: &DecisionAdapters,
    targets: &[AdapterTarget],
) -> Result<(), DecisionFailure> {
    for adapter in &decision_adapters.adapters {
        // The repository's lints require `find` here over a hand-written loop.
        let Some(target) = targets
            .iter()
            .find(|candidate_target| candidate_target.name == adapter.target)
        else {
            return Err(unusable_adapters(&format!(
                "the model has no layer {} that takes an add-on",
                adapter.target
            )));
        };
        if adapter.input_width != target.input_width || adapter.output_width != target.output_width
        {
            return Err(unusable_adapters(&format!(
                "the add-on for {} maps {} inputs to {} outputs but the layer maps {} to {}",
                adapter.target,
                adapter.input_width,
                adapter.output_width,
                target.input_width,
                target.output_width
            )));
        }
    }
    Ok(())
}

pub(crate) fn unusable_adapters(reason: &str) -> DecisionFailure {
    DecisionFailure::fatal_execution(&format!(
        "{DECISION_ADAPTERS_FILE_NAME} cannot be used: {reason}"
    ))
}

#[cfg(test)]
#[path = "decision_adapters_tests.rs"]
mod tests;
