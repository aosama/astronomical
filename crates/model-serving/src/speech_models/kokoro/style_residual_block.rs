//! StyleTTS2's `AdainResBlk1d`, the residual block of Kokoro's prosody
//! predictor and decoder: AdaIN, activation, (upsample), convolution, twice,
//! plus a shortcut, the sum scaled by `1/sqrt(2)` to keep its variance.

use astronomical_runtime_integration::MlxRuntime;
use mlx_bridge::MlxArray;

use super::convolutions::{ConvolutionGeometry, KokoroConv1d, KokoroConvTranspose1d};
use super::style_modulation::StyleModulation;
use crate::speech_models::checkpoint_tensors::CheckpointTensors;
use crate::speech_models::mlx_layers::activations::leaky_relu;
use crate::speech_models::speech_model_error::SpeechModelError;

const BLOCK_ACTIVATION_SLOPE: f32 = 0.2;

/// What one block looks like: channels in and out, and whether it doubles the length.
#[derive(Clone, Copy)]
pub(crate) struct StyleBlockShape {
    pub(crate) input_channels: i32,
    pub(crate) output_channels: i32,
    pub(crate) doubles_length: bool,
}

pub(crate) struct StyleResidualBlock {
    first_norm: StyleModulation,
    second_norm: StyleModulation,
    first_convolution: KokoroConv1d,
    second_convolution: KokoroConv1d,
    shortcut_projection: Option<KokoroConv1d>,
    /// The learned depthwise upsampler on the residual path (`pool`).
    residual_upsampler: Option<KokoroConvTranspose1d>,
}

impl StyleResidualBlock {
    pub(crate) fn load(
        checkpoint: &CheckpointTensors<'_>,
        prefix: &str,
        shape: StyleBlockShape,
    ) -> Result<Self, SpeechModelError> {
        let three_wide = ConvolutionGeometry::same_length(3, 1);
        Ok(Self {
            first_norm: StyleModulation::load(
                checkpoint,
                &format!("{prefix}.norm1"),
                shape.input_channels,
            )?,
            second_norm: StyleModulation::load(
                checkpoint,
                &format!("{prefix}.norm2"),
                shape.output_channels,
            )?,
            first_convolution: KokoroConv1d::load(
                checkpoint,
                &format!("{prefix}.conv1"),
                three_wide,
            )?,
            second_convolution: KokoroConv1d::load(
                checkpoint,
                &format!("{prefix}.conv2"),
                three_wide,
            )?,
            shortcut_projection: load_shortcut_projection(checkpoint, prefix, shape)?,
            residual_upsampler: load_residual_upsampler(checkpoint, prefix, shape)?,
        })
    }

    pub(crate) fn doubles_length(&self) -> bool {
        self.residual_upsampler.is_some()
    }

    pub(crate) fn forward(
        &self,
        runtime: &MlxRuntime,
        input: &MlxArray,
        style: &MlxArray,
    ) -> Result<MlxArray, SpeechModelError> {
        let residual = self.residual(runtime, input, style)?;
        let shortcut = self.shortcut(runtime, input)?;
        let summed_paths = runtime.add(&residual, &shortcut)?;
        Ok(runtime.multiply_scalar(&summed_paths, std::f32::consts::FRAC_1_SQRT_2)?)
    }

    fn residual(
        &self,
        runtime: &MlxRuntime,
        input: &MlxArray,
        style: &MlxArray,
    ) -> Result<MlxArray, SpeechModelError> {
        let normalized = self.first_norm.instance_modulate(runtime, input, style)?;
        let mut activated = leaky_relu(runtime, &normalized, BLOCK_ACTIVATION_SLOPE)?;
        if let Some(residual_upsampler) = &self.residual_upsampler {
            activated = residual_upsampler.forward(runtime, &activated)?;
        }
        let convolved = self.first_convolution.forward(runtime, &activated)?;
        let renormalized = self
            .second_norm
            .instance_modulate(runtime, &convolved, style)?;
        let reactivated = leaky_relu(runtime, &renormalized, BLOCK_ACTIVATION_SLOPE)?;
        self.second_convolution.forward(runtime, &reactivated)
    }

    /// Nearest-neighbour doubling (when the block upsamples), then a 1x1
    /// projection (when the channel count changes).
    fn shortcut(
        &self,
        runtime: &MlxRuntime,
        input: &MlxArray,
    ) -> Result<MlxArray, SpeechModelError> {
        let resized = if self.doubles_length() {
            runtime.repeat_axis(input, 2, 0)?
        } else {
            input.retain()?
        };
        match &self.shortcut_projection {
            Some(shortcut_projection) => shortcut_projection.forward(runtime, &resized),
            None => Ok(resized),
        }
    }
}

fn load_shortcut_projection(
    checkpoint: &CheckpointTensors<'_>,
    prefix: &str,
    shape: StyleBlockShape,
) -> Result<Option<KokoroConv1d>, SpeechModelError> {
    if shape.input_channels == shape.output_channels {
        return Ok(None);
    }
    let pointwise = ConvolutionGeometry::same_length(1, 1);
    KokoroConv1d::load(checkpoint, &format!("{prefix}.conv1x1"), pointwise).map(Some)
}

/// The depthwise stride-2 transposed convolution that doubles the length.
fn load_residual_upsampler(
    checkpoint: &CheckpointTensors<'_>,
    prefix: &str,
    shape: StyleBlockShape,
) -> Result<Option<KokoroConvTranspose1d>, SpeechModelError> {
    if !shape.doubles_length {
        return Ok(None);
    }
    let depthwise = ConvolutionGeometry {
        stride: 2,
        padding: 1,
        dilation: 1,
        groups: shape.input_channels,
    };
    KokoroConvTranspose1d::load(checkpoint, &format!("{prefix}.pool"), depthwise, 1).map(Some)
}
