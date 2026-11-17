//! Deleting the trained-tensors file of a training result nobody received.
//!
//! The worker writes a finished training's tensors to a file in the
//! temporary folder and the result carries its name (the decision
//! contract's `trained_tensors_file` says why). When the command-line tool
//! has stopped waiting, the result cannot be delivered and nothing would
//! ever read that file, so the daemon deletes it here instead of leaving
//! about 86 MB behind.

use std::path::Path;

use astronomical_decision_contract::trained_tensors_file::handover_file_path;

use crate::serving::decisions_executor::DecisionsOutput;

/// Deletes the hand-over file an undelivered `undelivered_output` names,
/// if any, from `handover_directory`.
pub(crate) fn discard_undelivered_handover_file(
    undelivered_output: &DecisionsOutput,
    handover_directory: &Path,
) {
    let DecisionsOutput::ModelTrained { outcome, .. } = undelivered_output else {
        return;
    };
    let Some(file_name) = &outcome.trained_tensors_file else {
        return;
    };
    if let Some(handover_path) = handover_file_path(handover_directory, file_name) {
        let _ = std::fs::remove_file(handover_path);
    }
}

#[cfg(test)]
#[path = "undelivered_training_result_tests.rs"]
mod tests;
