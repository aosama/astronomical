//! The trained tensors of a finished training as one safetensors file: how
//! the worker hands them to the command-line tool.
//!
//! The worker and the daemon exchange messages of at most
//! `MAX_IPC_FRAME_BYTES`, 32,000,000 bytes. A whole-model training of a 9B
//! backbone returns about 21.6 million add-on values, about 115,000,000
//! bytes once encoded in a message, so it could never arrive and every such
//! training failed after it had finished (found with Clef 9B, issue #2299).
//! The worker therefore writes the tensors to a file in its temporary folder
//! and the message carries only the file's name; each side joins that name
//! to its own temporary folder, so a message can never point at any other
//! file.
//!
//! The final layers and the add-ons go in one file, each name marked with
//! its list and its zero-padded position: reading follows the file's names
//! in order, and the add-on pairing expects each layer's `down` right
//! before its `up`.
//!
//! Naming, validating and encoding live here, apart from any filesystem
//! access, so the worker, the daemon and the command-line tool agree.

use std::path::{Path, PathBuf};

use thiserror::Error;

use crate::decision_training_job::TrainedTensor;
use crate::float32_safetensors::{
    Float32TensorsError, TunedTensor, float32_tensors_bytes, read_float32_tensors,
};
use crate::head_inputs::FloatVector;

const TRAINED_TENSOR_LIST: &str = "trained";
const ADAPTER_TENSOR_LIST: &str = "adapter";
/// Every hand-over file's name starts with this, so stale ones can be found.
pub const HANDOVER_FILE_PREFIX: &str = "astronomical-trained-tensors-";
const HANDOVER_FILE_SUFFIX: &str = ".safetensors";

/// The final layers' tensors and the add-ons, in the order they were trained.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct HandedOverTensors {
    pub trained_tensors: Vec<TrainedTensor>,
    pub adapter_tensors: Vec<TrainedTensor>,
}

/// Why a hand-over file cannot be written or read.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum TrainedTensorsFileError {
    #[error("{0}")]
    Tensors(#[from] Float32TensorsError),
    #[error("{0} is not marked as a trained tensor or an add-on")]
    UnmarkedTensor(String),
    #[error("the training produced a value in {0} that is not a finite number")]
    NonFiniteValue(String),
    #[error("{0}")]
    Unwritable(String),
}

/// The name of the hand-over file a worker with `process_id` writes at
/// `nanoseconds` past the epoch: unique while one worker runs one training
/// at a time.
#[must_use]
pub fn handover_file_name(process_id: u32, nanoseconds: u128) -> String {
    format!("{HANDOVER_FILE_PREFIX}{process_id}-{nanoseconds}{HANDOVER_FILE_SUFFIX}")
}

/// `file_name` in `handover_directory`, or `None` when the name is not one
/// [`handover_file_name`] makes: no folder separators, no other file.
#[must_use]
pub fn handover_file_path(handover_directory: &Path, file_name: &str) -> Option<PathBuf> {
    let middle = file_name
        .strip_prefix(HANDOVER_FILE_PREFIX)?
        .strip_suffix(HANDOVER_FILE_SUFFIX)?;
    if middle.is_empty() {
        return None;
    }
    for name_byte in middle.bytes() {
        if !name_byte.is_ascii_digit() && name_byte != b'-' {
            return None;
        }
    }
    Some(handover_directory.join(file_name))
}

/// The file's bytes for `handed_over_tensors`, which it consumes so the
/// values are moved rather than copied: a 9B model's add-ons are about 86 MB.
///
/// # Errors
/// Refuses a value that is not a finite number, naming its tensor, and
/// returns the serializer's reason when a tensor's values do not fill its
/// shape.
pub fn trained_tensors_file_bytes(
    handed_over_tensors: HandedOverTensors,
) -> Result<Vec<u8>, TrainedTensorsFileError> {
    let mut marked_tensors = Vec::with_capacity(
        handed_over_tensors.trained_tensors.len() + handed_over_tensors.adapter_tensors.len(),
    );
    for (list_name, tensors) in [
        (TRAINED_TENSOR_LIST, handed_over_tensors.trained_tensors),
        (ADAPTER_TENSOR_LIST, handed_over_tensors.adapter_tensors),
    ] {
        for (position, trained_tensor) in tensors.into_iter().enumerate() {
            for value in &trained_tensor.values.0 {
                if !value.is_finite() {
                    return Err(TrainedTensorsFileError::NonFiniteValue(trained_tensor.name));
                }
            }
            marked_tensors.push(TunedTensor {
                name: format!("{list_name}/{position:08}/{}", trained_tensor.name),
                shape: trained_tensor.shape,
                values: trained_tensor.values.0,
            });
        }
    }
    float32_tensors_bytes(&marked_tensors).map_err(TrainedTensorsFileError::Unwritable)
}

/// The tensors a file written by [`trained_tensors_file_bytes`] holds, in
/// the order they were written.
///
/// # Errors
/// Refuses a file that is not finite 32-bit safetensors, or a tensor whose
/// name lacks the list and position marks.
pub fn read_trained_tensors_file(
    file_bytes: &[u8],
) -> Result<HandedOverTensors, TrainedTensorsFileError> {
    let mut handed_over_tensors = HandedOverTensors::default();
    // Names sort by list, then by zero-padded position.
    for tuned_tensor in read_float32_tensors(file_bytes)? {
        let Some((list_name, positioned_name)) = tuned_tensor.name.split_once('/') else {
            return Err(TrainedTensorsFileError::UnmarkedTensor(tuned_tensor.name));
        };
        let Some((_position, shipped_name)) = positioned_name.split_once('/') else {
            return Err(TrainedTensorsFileError::UnmarkedTensor(tuned_tensor.name));
        };
        let trained_tensor = TrainedTensor {
            name: shipped_name.to_owned(),
            shape: tuned_tensor.shape,
            values: FloatVector(tuned_tensor.values),
        };
        match list_name {
            ADAPTER_TENSOR_LIST => handed_over_tensors.adapter_tensors.push(trained_tensor),
            TRAINED_TENSOR_LIST => handed_over_tensors.trained_tensors.push(trained_tensor),
            _ => {
                return Err(TrainedTensorsFileError::UnmarkedTensor(
                    tuned_tensor.name.clone(),
                ));
            }
        }
    }
    Ok(handed_over_tensors)
}

#[cfg(test)]
#[path = "trained_tensors_file_tests.rs"]
mod tests;
