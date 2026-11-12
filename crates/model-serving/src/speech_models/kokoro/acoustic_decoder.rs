//! Kokoro's decoder: frame-aligned text states, pitch and energy in, the
//! waveform generator's input features out, conditioned on the voice's
//! 128-wide timbre style.

use astronomical_runtime_integration::MlxRuntime;
use mlx_bridge::MlxArray;

use super::convolutions::{ConvolutionGeometry, KokoroConv1d};
use super::style_residual_block::{StyleBlockShape, StyleResidualBlock};
use super::waveform_generator::WaveformGenerator;
use crate::speech_models::checkpoint_tensors::CheckpointTensors;
use crate::speech_models::speech_model_error::SpeechModelError;

/// Text states (512), plus one pitch and one energy channel.
const ENCODER_INPUT_CHANNELS: i32 = 514;
const DECODER_WIDTH: i32 = 1024;
/// Decoder width plus pitch, energy and the 64-wide text residual.
const DECODE_BLOCK_INPUT_CHANNELS: i32 = 1090;
const DECODE_BLOCK_COUNT: usize = 4;

pub(crate) struct AcousticDecoder {
    pitch_downsampler: KokoroConv1d,
    energy_downsampler: KokoroConv1d,
    encode_block: StyleResidualBlock,
    text_residual: KokoroConv1d,
    decode_blocks: Vec<StyleResidualBlock>,
    generator: WaveformGenerator,
}

/// What the decoder reads for one utterance.
pub(crate) struct DecoderInputs<'utterance> {
    /// `[frames, 512]` frame-aligned text states.
    pub(crate) aligned_text: &'utterance MlxArray,
    /// `[2 * frames, 1]` pitch in hertz.
    pub(crate) pitch_curve: &'utterance MlxArray,
    /// `[2 * frames, 1]` energy.
    pub(crate) energy_curve: &'utterance MlxArray,
    /// `[1, 128]` timbre style.
    pub(crate) decoder_style: &'utterance MlxArray,
    /// Seeds the excitation noise; `None` leaves the noise out.
    pub(crate) noise_key: Option<&'utterance MlxArray>,
}

impl AcousticDecoder {
    pub(crate) fn load(checkpoint: &CheckpointTensors<'_>) -> Result<Self, SpeechModelError> {
        let halving = ConvolutionGeometry {
            stride: 2,
            padding: 1,
            dilation: 1,
            groups: 1,
        };
        let encode_shape = StyleBlockShape {
            input_channels: ENCODER_INPUT_CHANNELS,
            output_channels: DECODER_WIDTH,
            doubles_length: false,
        };
        Ok(Self {
            pitch_downsampler: KokoroConv1d::load(checkpoint, "decoder.F0_conv", halving)?,
            energy_downsampler: KokoroConv1d::load(checkpoint, "decoder.N_conv", halving)?,
            encode_block: StyleResidualBlock::load(checkpoint, "decoder.encode", encode_shape)?,
            text_residual: KokoroConv1d::load(
                checkpoint,
                "decoder.asr_res.0",
                ConvolutionGeometry::same_length(1, 1),
            )?,
            decode_blocks: load_decode_blocks(checkpoint)?,
            generator: WaveformGenerator::load(checkpoint)?,
        })
    }

    /// The utterance's samples, `[600 * frames]`.
    pub(crate) fn decode(
        &self,
        runtime: &MlxRuntime,
        inputs: &DecoderInputs<'_>,
    ) -> Result<MlxArray, SpeechModelError> {
        let frame_pitch = self
            .pitch_downsampler
            .forward(runtime, inputs.pitch_curve)?;
        let frame_energy = self
            .energy_downsampler
            .forward(runtime, inputs.energy_curve)?;
        let encoder_input =
            runtime.concatenate_axis(&[inputs.aligned_text, &frame_pitch, &frame_energy], 1)?;
        let mut features =
            self.encode_block
                .forward(runtime, &encoder_input, inputs.decoder_style)?;
        let text_residual = self.text_residual.forward(runtime, inputs.aligned_text)?;
        // Blocks before the upsampling one read the residual inputs again;
        // after it the length has doubled and they no longer line up.
        let mut should_append_residual_inputs = true;
        for block in &self.decode_blocks {
            if should_append_residual_inputs {
                features = runtime.concatenate_axis(
                    &[&features, &text_residual, &frame_pitch, &frame_energy],
                    1,
                )?;
            }
            features = block.forward(runtime, &features, inputs.decoder_style)?;
            if block.doubles_length() {
                should_append_residual_inputs = false;
            }
        }
        self.generator.generate(
            runtime,
            &features,
            inputs.decoder_style,
            inputs.pitch_curve,
            inputs.noise_key,
        )
    }
}

/// Three blocks at full width, then one that halves the width and doubles the length.
fn load_decode_blocks(
    checkpoint: &CheckpointTensors<'_>,
) -> Result<Vec<StyleResidualBlock>, SpeechModelError> {
    let mut decode_blocks = Vec::with_capacity(DECODE_BLOCK_COUNT);
    for block_index in 0..DECODE_BLOCK_COUNT {
        let is_last_block = block_index + 1 == DECODE_BLOCK_COUNT;
        let block_shape = StyleBlockShape {
            input_channels: DECODE_BLOCK_INPUT_CHANNELS,
            output_channels: if is_last_block {
                DECODER_WIDTH / 2
            } else {
                DECODER_WIDTH
            },
            doubles_length: is_last_block,
        };
        decode_blocks.push(StyleResidualBlock::load(
            checkpoint,
            &format!("decoder.decode.{block_index}"),
            block_shape,
        )?);
    }
    Ok(decode_blocks)
}
