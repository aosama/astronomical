//! Reading back the trained tensors the worker wrote to a file, because one
//! message between processes cannot hold a 9B model's add-ons
//! (`trained_tensors_file` in the decision contract says why).
//!
//! The message names only a file, never a folder: the name must be one the
//! worker makes, and it is looked up in this tool's own temporary folder,
//! which the worker shares as the same user. A message can therefore never
//! point the tool at another file, to read or to delete.

use std::path::Path;

use decision_contract::decision_training_job::DecisionTrainingOutcome;
use decision_contract::trained_tensors_file::{
    HandedOverTensors, handover_file_path, read_trained_tensors_file,
};

use super::decision_tool::DecisionToolError;

/// `outcome` with the tensors from its hand-over file in
/// `handover_directory` back in place, and the file deleted once read; an
/// outcome that carries its tensors in place is returned as it is.
///
/// # Errors
/// Returns a training failure when the name is not a hand-over file's, or
/// the file cannot be read or is not the trained tensors, since nothing
/// could be written without them.
pub(super) fn take_handed_over_tensors(
    mut outcome: DecisionTrainingOutcome,
    handover_directory: &Path,
) -> Result<DecisionTrainingOutcome, DecisionToolError> {
    let Some(file_name) = outcome.trained_tensors_file.take() else {
        return Ok(outcome);
    };
    let Some(handover_path) = handover_file_path(handover_directory, &file_name) else {
        return Err(unreadable_handover(&format!(
            "{file_name} is not a name the worker gives its files"
        )));
    };
    let handed_over_tensors = read_handover_file(&handover_path)?;
    // Read once, for this run alone; deleted only once it proved to be one.
    let _ = std::fs::remove_file(&handover_path);
    outcome.trained_tensors = handed_over_tensors.trained_tensors;
    outcome.adapter_tensors = handed_over_tensors.adapter_tensors;
    Ok(outcome)
}

/// The tensors in the file at `handover_path`; its bytes are dropped on
/// return, so the tool holds one copy of a 9B model's add-ons, not two.
fn read_handover_file(handover_path: &Path) -> Result<HandedOverTensors, DecisionToolError> {
    let file_bytes = match std::fs::read(handover_path) {
        Ok(file_bytes) => file_bytes,
        Err(read_error) => return Err(unreadable_handover(&read_error.to_string())),
    };
    match read_trained_tensors_file(&file_bytes) {
        Ok(handed_over_tensors) => Ok(handed_over_tensors),
        Err(format_error) => Err(unreadable_handover(&format_error.to_string())),
    }
}

fn unreadable_handover(reason: &str) -> DecisionToolError {
    DecisionToolError::TrainingFailed(format!(
        "the trained weights the worker wrote could not be read ({reason}); nothing was written"
    ))
}

#[cfg(test)]
#[path = "trained_tensors_handover_reading_tests.rs"]
mod tests;
