//! Conditioning on the voice's style: a linear layer turns the 128-wide style
//! vector into a per-channel scale and shift, applied after normalising.
//! StyleTTS2's AdaIN normalises each channel over the length (instance norm);
//! its AdaLayerNorm normalises each step over the channels.

use astronomical_runtime_integration::MlxRuntime;
use mlx_bridge::MlxArray;

use crate::speech_models::checkpoint_tensors::CheckpointTensors;
use crate::speech_models::mlx_layers::activations::modulate;
use crate::speech_models::mlx_layers::layer_norm::normalized_over_axis;
use crate::speech_models::mlx_layers::linear::Linear;
use crate::speech_models::speech_model_error::SpeechModelError;

/// PyTorch's default epsilon for both norms.
const STYLE_NORM_EPSILON: f32 = 1e-5;
const LENGTH_AXIS: i32 = 0;
const CHANNEL_AXIS: i32 = -1;

pub(crate) struct StyleModulation {
    style_projection: Linear,
    channel_count: i32,
}

impl StyleModulation {
    /// The `prefix.fc` projection, for `channel_count` channels.
    pub(crate) fn load(
        checkpoint: &CheckpointTensors<'_>,
        prefix: &str,
        channel_count: i32,
    ) -> Result<Self, SpeechModelError> {
        Ok(Self {
            style_projection: Linear::load(checkpoint, &format!("{prefix}.fc"))?,
            channel_count,
        })
    }

    /// AdaIN: instance norm, then the style's scale and shift.
    pub(crate) fn instance_modulate(
        &self,
        runtime: &MlxRuntime,
        input: &MlxArray,
        style: &MlxArray,
    ) -> Result<MlxArray, SpeechModelError> {
        let normalized = normalized_over_axis(runtime, input, LENGTH_AXIS, STYLE_NORM_EPSILON)?;
        self.modulate(runtime, &normalized, style)
    }

    /// AdaLayerNorm: channel norm, then the style's scale and shift.
    pub(crate) fn channel_modulate(
        &self,
        runtime: &MlxRuntime,
        input: &MlxArray,
        style: &MlxArray,
    ) -> Result<MlxArray, SpeechModelError> {
        let normalized = normalized_over_axis(runtime, input, CHANNEL_AXIS, STYLE_NORM_EPSILON)?;
        self.modulate(runtime, &normalized, style)
    }

    /// The projection's first half is the scale, its second the shift.
    fn modulate(
        &self,
        runtime: &MlxRuntime,
        normalized: &MlxArray,
        style: &MlxArray,
    ) -> Result<MlxArray, SpeechModelError> {
        let scale_and_shift = self.style_projection.forward(runtime, style)?;
        let channel_count = self.channel_count;
        let scale = runtime.slice(&scale_and_shift, &[0, 0], &[1, channel_count], &[1, 1])?;
        let shift = runtime.slice(
            &scale_and_shift,
            &[0, channel_count],
            &[1, 2 * channel_count],
            &[1, 1],
        )?;
        modulate(runtime, normalized, &shift, &scale)
    }
}

#[cfg(test)]
#[path = "style_modulation_tests.rs"]
mod tests;
