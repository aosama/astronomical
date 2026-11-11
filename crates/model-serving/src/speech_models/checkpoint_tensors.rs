//! Reading a speech checkpoint's files and tensors.
//!
//! Every tensor is cast to float32 and evaluated as it is read. The speech
//! models are small (tens of millions of parameters), so float32 costs little
//! memory, and their audio paths are sensitive to the rounding bfloat16
//! activations would add; a checkpoint published in bfloat16 would otherwise
//! silently run parts of the model at lower precision. Evaluating at load also
//! means a forward pass never re-reads or re-casts a weight.

use std::fs::File;
use std::path::Path;

use astronomical_runtime_integration::{MlxRuntime, MlxSafetensors};
use mlx_bridge::{MlxArray, MlxDtype};

use super::speech_model_error::SpeechModelError;

/// One loaded safetensors file and the runtime that reads it.
pub(crate) struct CheckpointTensors<'checkpoint> {
    pub(crate) runtime: &'checkpoint MlxRuntime,
    pub(crate) tensors: &'checkpoint MlxSafetensors,
}

impl CheckpointTensors<'_> {
    /// The named tensor as float32, or an artifact error naming it.
    pub(crate) fn tensor(&self, tensor_name: &str) -> Result<MlxArray, SpeechModelError> {
        let stored_tensor = self.tensors.tensor(tensor_name).map_err(|missing| {
            SpeechModelError::Artifact(format!("tensor {tensor_name}: {missing}"))
        })?;
        let float_tensor = self.runtime.astype(&stored_tensor, MlxDtype::Float32)?;
        self.runtime.evaluate_arrays(&[&float_tensor])?;
        Ok(float_tensor)
    }

    /// The named tensor when the checkpoint has it, such as an optional bias.
    pub(crate) fn optional_tensor(
        &self,
        tensor_name: &str,
    ) -> Result<Option<MlxArray>, SpeechModelError> {
        if self.tensors.tensor(tensor_name).is_err() {
            return Ok(None);
        }
        self.tensor(tensor_name).map(Some)
    }
}

/// Opens one safetensors file of a checkpoint. The returned map keeps no
/// file handle the arrays need: each array holds its reader alive itself.
pub(crate) fn load_safetensors_file(
    runtime: &MlxRuntime,
    safetensors_path: &Path,
) -> Result<MlxSafetensors, SpeechModelError> {
    let safetensors_file = File::open(safetensors_path).map_err(|open_error| {
        SpeechModelError::Artifact(format!("{}: {open_error}", safetensors_path.display()))
    })?;
    Ok(runtime.load_safetensors(safetensors_file, None)?)
}

/// Reads a text artifact such as `config.json`.
pub(crate) fn read_text_artifact(artifact_path: &Path) -> Result<String, SpeechModelError> {
    std::fs::read_to_string(artifact_path).map_err(|read_error| {
        SpeechModelError::Artifact(format!("{}: {read_error}", artifact_path.display()))
    })
}

#[cfg(test)]
#[path = "checkpoint_tensors_tests.rs"]
mod tests;
