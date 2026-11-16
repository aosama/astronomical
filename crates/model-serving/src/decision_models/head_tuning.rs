//! Reads a decision model folder's `decision_head_tuning.safetensors`, the
//! trained final layers that `decision train` writes beside the original
//! weights, and checks each tensor against the one it replaces.
//!
//! A folder without the file serves its shipped head exactly as before. A
//! folder whose file cannot be used refuses to load: serving the shipped head
//! while the person believes their training is in force would be a silent
//! wrong answer.

use std::path::Path;

use astronomical_decision_contract::DecisionFailure;
use astronomical_decision_contract::head_tuning_file::{
    DECISION_HEAD_TUNING_FILE_NAME, HeadTuning, MAXIMUM_HEAD_TUNING_FILE_BYTES, TunableHeadFamily,
    TunedTensor, parse_head_tuning,
};

/// The folder's trained head, or an empty one when the folder has none.
pub(crate) fn read_head_tuning(
    model_directory: &Path,
    family: TunableHeadFamily,
) -> Result<HeadTuning, DecisionFailure> {
    let tuning_path = model_directory.join(DECISION_HEAD_TUNING_FILE_NAME);
    let tuning_size = match std::fs::metadata(&tuning_path) {
        Ok(tuning_metadata) => tuning_metadata.len(),
        Err(metadata_error) if metadata_error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(HeadTuning::default());
        }
        Err(_) => return Err(unusable_tuning("it cannot be read")),
    };
    if tuning_size > MAXIMUM_HEAD_TUNING_FILE_BYTES {
        return Err(unusable_tuning("it is too large"));
    }
    let Ok(tuning_bytes) = std::fs::read(&tuning_path) else {
        return Err(unusable_tuning("it cannot be read"));
    };
    match parse_head_tuning(&tuning_bytes, family) {
        Ok(head_tuning) => Ok(head_tuning),
        Err(tuning_error) => Err(unusable_tuning(&tuning_error.to_string())),
    }
}

/// Checks that a trained tensor has exactly the shape of the shipped tensor
/// it replaces. Any difference means the file was trained for another model,
/// and using it would read the wrong numbers.
pub(crate) fn check_tuned_shape(
    tuned_tensor: &TunedTensor,
    shipped_shape: &[usize],
) -> Result<(), DecisionFailure> {
    if tuned_tensor.shape == shipped_shape {
        return Ok(());
    }
    Err(unusable_tuning(&format!(
        "{} has shape {:?} but the model's has {:?}",
        tuned_tensor.name, tuned_tensor.shape, shipped_shape
    )))
}

pub(crate) fn unusable_tuning(reason: &str) -> DecisionFailure {
    DecisionFailure::fatal_execution(&format!(
        "{DECISION_HEAD_TUNING_FILE_NAME} cannot be used: {reason}"
    ))
}

#[cfg(test)]
#[path = "head_tuning_tests.rs"]
mod tests;
