//! Reading a folder's `decision_training_record.json` from disk.
//!
//! The command-line tool reads it to build on a source folder's record, and
//! every engine reads it to put the record into each answer. One reader
//! keeps them agreeing on what a missing, oversized or unreadable file
//! means; the format itself lives in the decision contract.

use std::path::Path;

use astronomical_decision_contract::training_record::{
    DECISION_TRAINING_RECORD_FILE_NAME, MAXIMUM_DECISION_TRAINING_RECORD_FILE_BYTES,
    TrainingRecord, parse_training_record,
};
use thiserror::Error;

/// Why a record file that is present cannot be used.
#[derive(Debug, Error)]
pub enum TrainingRecordFileError {
    #[error("{DECISION_TRAINING_RECORD_FILE_NAME} cannot be read: {0}")]
    Unreadable(String),
    #[error("{DECISION_TRAINING_RECORD_FILE_NAME} is too large to be a training record")]
    TooLarge,
    #[error("{0}")]
    Malformed(String),
}

/// The folder's training record, or `None` when the folder has none.
pub fn read_training_record_file(
    model_directory: &Path,
) -> Result<Option<TrainingRecord>, TrainingRecordFileError> {
    let record_path = model_directory.join(DECISION_TRAINING_RECORD_FILE_NAME);
    let record_size = match std::fs::metadata(&record_path) {
        Ok(record_metadata) => record_metadata.len(),
        Err(metadata_error) if metadata_error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(None);
        }
        Err(metadata_error) => {
            return Err(TrainingRecordFileError::Unreadable(
                metadata_error.to_string(),
            ));
        }
    };
    if record_size > MAXIMUM_DECISION_TRAINING_RECORD_FILE_BYTES {
        return Err(TrainingRecordFileError::TooLarge);
    }
    let record_bytes = match std::fs::read(&record_path) {
        Ok(record_bytes) => record_bytes,
        Err(read_error) => return Err(TrainingRecordFileError::Unreadable(read_error.to_string())),
    };
    match parse_training_record(&record_bytes) {
        Ok(training_record) => Ok(Some(training_record)),
        Err(record_error) => Err(TrainingRecordFileError::Malformed(record_error.to_string())),
    }
}

#[cfg(test)]
#[path = "training_record_file_tests.rs"]
mod tests;
