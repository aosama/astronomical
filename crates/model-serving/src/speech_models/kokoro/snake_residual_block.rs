//! HiFi-GAN's `AdaINResBlock1` with Snake activations, the residual block of
//! Kokoro's waveform generator: three stages of AdaIN, Snake, dilated
//! convolution, AdaIN, Snake, convolution, each added back to its input.

use astronomical_runtime_integration::MlxRuntime;
use mlx_bridge::MlxArray;

use super::convolutions::{ConvolutionGeometry, KokoroConv1d};
use super::style_modulation::StyleModulation;
use crate::speech_models::checkpoint_tensors::CheckpointTensors;
use crate::speech_models::speech_model_error::SpeechModelError;

/// The block's channels, kernel and the dilation of each stage.
#[derive(Clone, Copy)]
pub(crate) struct SnakeBlockShape {
    pub(crate) channel_count: i32,
    pub(crate) kernel_size: i32,
    pub(crate) dilations: [i32; 3],
}

struct SnakeStage {
    first_norm: StyleModulation,
    second_norm: StyleModulation,
    dilated_convolution: KokoroConv1d,
    plain_convolution: KokoroConv1d,
    /// Snake frequencies, `[1, channels]`.
    first_frequency: MlxArray,
    second_frequency: MlxArray,
}

pub(crate) struct SnakeResidualBlock {
    stages: Vec<SnakeStage>,
}

impl SnakeResidualBlock {
    pub(crate) fn load(
        checkpoint: &CheckpointTensors<'_>,
        prefix: &str,
        shape: SnakeBlockShape,
    ) -> Result<Self, SpeechModelError> {
        let mut stages = Vec::with_capacity(shape.dilations.len());
        for (stage_index, dilation) in shape.dilations.into_iter().enumerate() {
            stages.push(SnakeStage::load(
                checkpoint,
                prefix,
                stage_index,
                shape,
                dilation,
            )?);
        }
        Ok(Self { stages })
    }

    pub(crate) fn forward(
        &self,
        runtime: &MlxRuntime,
        input: &MlxArray,
        style: &MlxArray,
    ) -> Result<MlxArray, SpeechModelError> {
        let mut block_states = input.retain()?;
        for stage in &self.stages {
            let stage_output = stage.forward(runtime, &block_states, style)?;
            block_states = runtime.add(&stage_output, &block_states)?;
        }
        Ok(block_states)
    }
}

impl SnakeStage {
    fn load(
        checkpoint: &CheckpointTensors<'_>,
        prefix: &str,
        stage_index: usize,
        shape: SnakeBlockShape,
        dilation: i32,
    ) -> Result<Self, SpeechModelError> {
        let channel_count = shape.channel_count;
        Ok(Self {
            first_norm: StyleModulation::load(
                checkpoint,
                &format!("{prefix}.adain1.{stage_index}"),
                channel_count,
            )?,
            second_norm: StyleModulation::load(
                checkpoint,
                &format!("{prefix}.adain2.{stage_index}"),
                channel_count,
            )?,
            dilated_convolution: KokoroConv1d::load(
                checkpoint,
                &format!("{prefix}.convs1.{stage_index}"),
                ConvolutionGeometry::same_length(shape.kernel_size, dilation),
            )?,
            plain_convolution: KokoroConv1d::load(
                checkpoint,
                &format!("{prefix}.convs2.{stage_index}"),
                ConvolutionGeometry::same_length(shape.kernel_size, 1),
            )?,
            first_frequency: snake_frequency(
                checkpoint,
                &format!("{prefix}.alpha1.{stage_index}"),
            )?,
            second_frequency: snake_frequency(
                checkpoint,
                &format!("{prefix}.alpha2.{stage_index}"),
            )?,
        })
    }

    fn forward(
        &self,
        runtime: &MlxRuntime,
        input: &MlxArray,
        style: &MlxArray,
    ) -> Result<MlxArray, SpeechModelError> {
        let normalized = self.first_norm.instance_modulate(runtime, input, style)?;
        let snaked = snake(runtime, &normalized, &self.first_frequency)?;
        let convolved = self.dilated_convolution.forward(runtime, &snaked)?;
        let renormalized = self
            .second_norm
            .instance_modulate(runtime, &convolved, style)?;
        let resnaked = snake(runtime, &renormalized, &self.second_frequency)?;
        self.plain_convolution.forward(runtime, &resnaked)
    }
}

/// The checkpoint's `[1, channels, 1]` Snake frequency as `[1, channels]`.
fn snake_frequency(
    checkpoint: &CheckpointTensors<'_>,
    tensor_name: &str,
) -> Result<MlxArray, SpeechModelError> {
    let stored_frequency = checkpoint.tensor(tensor_name)?;
    let channel_count = stored_frequency.shape()[1];
    Ok(checkpoint
        .runtime
        .reshape(&stored_frequency, &[1, channel_count])?)
}

/// Snake: `x + sin(a x)^2 / a`, a periodic activation that helps a vocoder
/// model pitch.
pub(crate) fn snake(
    runtime: &MlxRuntime,
    input: &MlxArray,
    frequency: &MlxArray,
) -> Result<MlxArray, SpeechModelError> {
    let sine = runtime.sin(&runtime.multiply(input, frequency)?)?;
    let squared_sine = runtime.multiply(&sine, &sine)?;
    Ok(runtime.add(input, &runtime.divide(&squared_sine, frequency)?)?)
}

#[cfg(test)]
#[path = "snake_residual_block_tests.rs"]
mod tests;
