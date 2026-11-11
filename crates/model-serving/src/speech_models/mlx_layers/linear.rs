//! A linear layer: `x @ W^T`, plus a bias when the checkpoint has one.

use astronomical_runtime_integration::MlxRuntime;
use mlx_bridge::MlxArray;

use crate::speech_models::checkpoint_tensors::CheckpointTensors;
use crate::speech_models::speech_model_error::SpeechModelError;

pub(crate) struct Linear {
    /// `[in, out]`, transposed once at load so the forward pass is one product.
    transposed_weight: MlxArray,
    bias: Option<MlxArray>,
}

impl Linear {
    pub(crate) fn load(
        checkpoint: &CheckpointTensors<'_>,
        prefix: &str,
    ) -> Result<Self, SpeechModelError> {
        let weight = checkpoint.tensor(&format!("{prefix}.weight"))?;
        Ok(Self {
            transposed_weight: checkpoint.runtime.transpose_axes(&weight, &[1, 0])?,
            bias: checkpoint.optional_tensor(&format!("{prefix}.bias"))?,
        })
    }

    pub(crate) fn forward(
        &self,
        runtime: &MlxRuntime,
        input: &MlxArray,
    ) -> Result<MlxArray, SpeechModelError> {
        let projected = runtime.matmul(input, &self.transposed_weight)?;
        match &self.bias {
            Some(bias) => Ok(runtime.add(&projected, bias)?),
            None => Ok(projected),
        }
    }
}
