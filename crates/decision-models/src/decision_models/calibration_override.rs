//! Reads a decision model folder's `decision_calibration.json`, shared by
//! every family.
//!
//! A folder without the file serves with the family's own temperatures,
//! exactly as before. A folder with a file that cannot be read refuses to
//! load: serving with the shipped temperatures while the person believes
//! their calibration is in force would be a silent wrong answer.

use std::path::Path;

use astronomical_decision_contract::DecisionFailure;
use astronomical_decision_contract::calibration_file::{
    CalibrationOverride, DECISION_CALIBRATION_FILE_NAME, MAXIMUM_DECISION_CALIBRATION_FILE_BYTES,
    parse_calibration_document,
};

/// The folder's calibration, or an empty one when the folder has none.
pub fn read_calibration_override(
    model_directory: &Path,
) -> Result<CalibrationOverride, DecisionFailure> {
    let calibration_path = model_directory.join(DECISION_CALIBRATION_FILE_NAME);
    let calibration_size = match std::fs::metadata(&calibration_path) {
        Ok(calibration_metadata) => calibration_metadata.len(),
        Err(metadata_error) if metadata_error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(CalibrationOverride::default());
        }
        Err(_) => return Err(unusable_calibration("it cannot be read")),
    };
    if calibration_size > MAXIMUM_DECISION_CALIBRATION_FILE_BYTES {
        return Err(unusable_calibration("it is too large"));
    }
    let Ok(calibration_bytes) = std::fs::read(&calibration_path) else {
        return Err(unusable_calibration("it cannot be read"));
    };
    match parse_calibration_document(&calibration_bytes) {
        Ok(calibration_override) => Ok(calibration_override),
        Err(calibration_error) => Err(DecisionFailure::fatal_execution(
            &calibration_error.to_string(),
        )),
    }
}

fn unusable_calibration(reason: &str) -> DecisionFailure {
    DecisionFailure::fatal_execution(&format!(
        "{DECISION_CALIBRATION_FILE_NAME} cannot be used: {reason}"
    ))
}

#[cfg(test)]
#[path = "calibration_override_tests.rs"]
mod tests;
