//! Kokoro's iSTFTNet generator: decoder features up to a 120x finer frame
//! rate, mixed with the harmonic excitation, then out as a short-time
//! spectrum that the inverse transform turns into 24 kHz samples.

use astronomical_runtime_integration::MlxRuntime;
use mlx_bridge::MlxArray;

use super::convolutions::{ConvolutionGeometry, KokoroConv1d, KokoroConvTranspose1d};
use super::harmonic_source::HarmonicSource;
use super::short_time_fourier::{FREQUENCY_BIN_COUNT, ShortTimeFourier};
use super::snake_residual_block::{SnakeBlockShape, SnakeResidualBlock};
use crate::speech_models::checkpoint_tensors::CheckpointTensors;
use crate::speech_models::mlx_layers::activations::leaky_relu;
use crate::speech_models::speech_model_error::SpeechModelError;

const INITIAL_CHANNELS: i32 = 512;
pub(crate) const UPSAMPLE_RATES: [i32; 2] = [10, 6];
pub(crate) const UPSAMPLE_KERNEL_SIZES: [i32; 2] = [20, 12];
pub(crate) const RESIDUAL_KERNEL_SIZES: [i32; 3] = [3, 7, 11];
pub(crate) const RESIDUAL_DILATIONS: [i32; 3] = [1, 3, 5];
/// The source blocks' kernels: 7 at the coarser stage, 11 at the finest.
const SOURCE_BLOCK_KERNEL_SIZES: [i32; 2] = [7, 11];
const STAGE_ACTIVATION_SLOPE: f32 = 0.1;
/// `F.leaky_relu`'s default slope, which Kokoro uses before the last layer.
const OUTPUT_ACTIVATION_SLOPE: f32 = 0.01;
const PREFIX: &str = "decoder.generator";

struct UpsamplingStage {
    upsampler: KokoroConvTranspose1d,
    source_convolution: KokoroConv1d,
    source_block: SnakeResidualBlock,
    residual_blocks: Vec<SnakeResidualBlock>,
}

pub(crate) struct WaveformGenerator {
    harmonic_source: HarmonicSource,
    fourier: ShortTimeFourier,
    stages: Vec<UpsamplingStage>,
    output_convolution: KokoroConv1d,
}

impl WaveformGenerator {
    pub(crate) fn load(checkpoint: &CheckpointTensors<'_>) -> Result<Self, SpeechModelError> {
        let mut stages = Vec::with_capacity(UPSAMPLE_RATES.len());
        for stage_index in 0..UPSAMPLE_RATES.len() {
            stages.push(UpsamplingStage::load(checkpoint, stage_index)?);
        }
        Ok(Self {
            harmonic_source: HarmonicSource::load(checkpoint)?,
            fourier: ShortTimeFourier::new(checkpoint.runtime)?,
            stages,
            output_convolution: KokoroConv1d::load(
                checkpoint,
                &format!("{PREFIX}.conv_post"),
                ConvolutionGeometry::same_length(7, 1),
            )?,
        })
    }

    /// `[2 * frames, 512]` decoder features and `[2 * frames, 1]` pitch to
    /// `[600 * frames]` samples.
    pub(crate) fn generate(
        &self,
        runtime: &MlxRuntime,
        decoder_features: &MlxArray,
        decoder_style: &MlxArray,
        pitch_curve: &MlxArray,
        noise_key: Option<&MlxArray>,
    ) -> Result<MlxArray, SpeechModelError> {
        let excitation = self
            .harmonic_source
            .excitation(runtime, pitch_curve, noise_key)?;
        let excitation_spectrum = self.fourier.magnitude_and_phase(runtime, &excitation)?;
        let mut features = decoder_features.retain()?;
        let last_stage_index = self.stages.len() - 1;
        for (stage_index, stage) in self.stages.iter().enumerate() {
            let is_last_stage = stage_index == last_stage_index;
            features = stage.forward(
                runtime,
                &features,
                &excitation_spectrum,
                decoder_style,
                is_last_stage,
            )?;
        }
        let activated = leaky_relu(runtime, &features, OUTPUT_ACTIVATION_SLOPE)?;
        let spectrum = self.output_convolution.forward(runtime, &activated)?;
        self.spectrum_to_samples(runtime, &spectrum)
    }

    /// The output convolution's first 11 channels are log magnitudes, the
    /// other 11 the phases' sines.
    fn spectrum_to_samples(
        &self,
        runtime: &MlxRuntime,
        spectrum: &MlxArray,
    ) -> Result<MlxArray, SpeechModelError> {
        let frame_count = spectrum.shape()[0];
        let bins = FREQUENCY_BIN_COUNT as i32;
        let log_magnitude = runtime.slice(spectrum, &[0, 0], &[frame_count, bins], &[1, 1])?;
        let phase_source =
            runtime.slice(spectrum, &[0, bins], &[frame_count, 2 * bins], &[1, 1])?;
        let magnitude = runtime.exp(&log_magnitude)?;
        let phase = runtime.sin(&phase_source)?;
        self.fourier.inverse(runtime, &magnitude, &phase)
    }
}

impl UpsamplingStage {
    fn load(
        checkpoint: &CheckpointTensors<'_>,
        stage_index: usize,
    ) -> Result<Self, SpeechModelError> {
        let output_channels = INITIAL_CHANNELS >> (stage_index + 1);
        let source_shape = SnakeBlockShape {
            channel_count: output_channels,
            kernel_size: SOURCE_BLOCK_KERNEL_SIZES[stage_index],
            dilations: RESIDUAL_DILATIONS,
        };
        Ok(Self {
            upsampler: KokoroConvTranspose1d::load(
                checkpoint,
                &format!("{PREFIX}.ups.{stage_index}"),
                upsampler_geometry(stage_index),
                0,
            )?,
            source_convolution: KokoroConv1d::load(
                checkpoint,
                &format!("{PREFIX}.noise_convs.{stage_index}"),
                source_convolution_geometry(stage_index),
            )?,
            source_block: SnakeResidualBlock::load(
                checkpoint,
                &format!("{PREFIX}.noise_res.{stage_index}"),
                source_shape,
            )?,
            residual_blocks: load_residual_blocks(checkpoint, stage_index, output_channels)?,
        })
    }

    fn forward(
        &self,
        runtime: &MlxRuntime,
        features: &MlxArray,
        excitation_spectrum: &MlxArray,
        decoder_style: &MlxArray,
        is_last_stage: bool,
    ) -> Result<MlxArray, SpeechModelError> {
        let activated = leaky_relu(runtime, features, STAGE_ACTIVATION_SLOPE)?;
        let source_input = self
            .source_convolution
            .forward(runtime, excitation_spectrum)?;
        let source_features = self
            .source_block
            .forward(runtime, &source_input, decoder_style)?;
        let mut upsampled = self.upsampler.forward(runtime, &activated)?;
        if is_last_stage {
            upsampled = reflect_one_sample_at_start(runtime, &upsampled)?;
        }
        let mixed = runtime.add(&upsampled, &source_features)?;
        average_of_blocks(runtime, &self.residual_blocks, &mixed, decoder_style)
    }
}

fn upsampler_geometry(stage_index: usize) -> ConvolutionGeometry {
    let rate = UPSAMPLE_RATES[stage_index];
    let kernel_size = UPSAMPLE_KERNEL_SIZES[stage_index];
    ConvolutionGeometry {
        stride: rate,
        padding: (kernel_size - rate) / 2,
        dilation: 1,
        groups: 1,
    }
}

/// The excitation spectrum runs at the finest rate; each stage's source
/// convolution strides it down to that stage's rate.
fn source_convolution_geometry(stage_index: usize) -> ConvolutionGeometry {
    if stage_index + 1 == UPSAMPLE_RATES.len() {
        return ConvolutionGeometry::same_length(1, 1);
    }
    let remaining_rate: i32 = UPSAMPLE_RATES[stage_index + 1..].iter().product();
    ConvolutionGeometry {
        stride: remaining_rate,
        padding: (remaining_rate + 1) / 2,
        dilation: 1,
        groups: 1,
    }
}

fn load_residual_blocks(
    checkpoint: &CheckpointTensors<'_>,
    stage_index: usize,
    channel_count: i32,
) -> Result<Vec<SnakeResidualBlock>, SpeechModelError> {
    let mut residual_blocks = Vec::with_capacity(RESIDUAL_KERNEL_SIZES.len());
    for (kernel_index, kernel_size) in RESIDUAL_KERNEL_SIZES.into_iter().enumerate() {
        let block_index = stage_index * RESIDUAL_KERNEL_SIZES.len() + kernel_index;
        let block_shape = SnakeBlockShape {
            channel_count,
            kernel_size,
            dilations: RESIDUAL_DILATIONS,
        };
        residual_blocks.push(SnakeResidualBlock::load(
            checkpoint,
            &format!("{PREFIX}.resblocks.{block_index}"),
            block_shape,
        )?);
    }
    Ok(residual_blocks)
}

/// `ReflectionPad1d((1, 0))`: the second sample copied in front.
pub(crate) fn reflect_one_sample_at_start(
    runtime: &MlxRuntime,
    features: &MlxArray,
) -> Result<MlxArray, SpeechModelError> {
    let channel_count = features.shape()[1];
    let second_sample = runtime.slice(features, &[1, 0], &[2, channel_count], &[1, 1])?;
    Ok(runtime.concatenate_axis(&[&second_sample, features], 0)?)
}

/// The mean of every block's output on the same input (HiFi-GAN's multi-receptive-field fusion).
pub(crate) fn average_of_blocks(
    runtime: &MlxRuntime,
    blocks: &[SnakeResidualBlock],
    features: &MlxArray,
    decoder_style: &MlxArray,
) -> Result<MlxArray, SpeechModelError> {
    let mut summed_outputs = blocks[0].forward(runtime, features, decoder_style)?;
    for block in &blocks[1..] {
        summed_outputs = runtime.add(
            &summed_outputs,
            &block.forward(runtime, features, decoder_style)?,
        )?;
    }
    Ok(runtime.multiply_scalar(&summed_outputs, 1.0 / blocks.len() as f32)?)
}

#[cfg(test)]
#[path = "waveform_generator_tests.rs"]
mod tests;
