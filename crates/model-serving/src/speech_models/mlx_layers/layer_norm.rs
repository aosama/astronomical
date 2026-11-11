//! Normalisation layers.
//!
//! [`LayerNorm`] normalises over channels with a learned (or unit) scale and
//! shift; [`normalized_over_axis`] is the parameter-free form both instance
//! norm (over the length) and an unscaled layer norm (over channels) reduce to.

use astronomical_runtime_integration::MlxRuntime;
use mlx_bridge::{MlxArray, MlxDtype};

use super::scalar_arithmetic::add_scalar;
use crate::speech_models::checkpoint_tensors::CheckpointTensors;
use crate::speech_models::speech_model_error::SpeechModelError;

pub(crate) struct LayerNorm {
    scale: MlxArray,
    shift: MlxArray,
    epsilon: f32,
}

impl LayerNorm {
    /// Scale and shift stored as `weight` and `bias`.
    pub(crate) fn load(
        checkpoint: &CheckpointTensors<'_>,
        prefix: &str,
        epsilon: f32,
    ) -> Result<Self, SpeechModelError> {
        Ok(Self {
            scale: checkpoint.tensor(&format!("{prefix}.weight"))?,
            shift: checkpoint.tensor(&format!("{prefix}.bias"))?,
            epsilon,
        })
    }

    /// Scale and shift stored as `gamma` and `beta`, StyleTTS2's names.
    pub(crate) fn load_gamma_beta(
        checkpoint: &CheckpointTensors<'_>,
        prefix: &str,
        epsilon: f32,
    ) -> Result<Self, SpeechModelError> {
        Ok(Self {
            scale: checkpoint.tensor(&format!("{prefix}.gamma"))?,
            shift: checkpoint.tensor(&format!("{prefix}.beta"))?,
            epsilon,
        })
    }

    /// Scale 1 and shift 0, for a layer trained without an affine part.
    pub(crate) fn plain(
        runtime: &MlxRuntime,
        channel_count: i32,
        epsilon: f32,
    ) -> Result<Self, SpeechModelError> {
        Ok(Self {
            scale: runtime.ones(&[channel_count], MlxDtype::Float32)?,
            shift: runtime.zeros(&[channel_count], MlxDtype::Float32)?,
            epsilon,
        })
    }

    pub(crate) fn forward(
        &self,
        runtime: &MlxRuntime,
        input: &MlxArray,
    ) -> Result<MlxArray, SpeechModelError> {
        Ok(runtime.layer_norm(input, &self.scale, &self.shift, self.epsilon)?)
    }
}

/// `(x - mean) / sqrt(variance + epsilon)` along `axis`, with the biased
/// variance PyTorch's norms use.
pub(crate) fn normalized_over_axis(
    runtime: &MlxRuntime,
    input: &MlxArray,
    axis: i32,
    epsilon: f32,
) -> Result<MlxArray, SpeechModelError> {
    let mean = runtime.mean_axis(input, axis, true)?;
    let variance = runtime.var_axis(input, axis, true, 0)?;
    let centered = runtime.subtract(input, &mean)?;
    let inverse_deviation = runtime.rsqrt(&add_scalar(runtime, &variance, epsilon)?)?;
    Ok(runtime.multiply(&centered, &inverse_deviation)?)
}
