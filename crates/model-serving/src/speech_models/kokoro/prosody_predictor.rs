//! Kokoro's prosody predictor: how many frames each phoneme lasts, then the
//! pitch (F0) and energy (N) curves over those frames, all conditioned on the
//! voice's 128-wide prosody style.

use astronomical_runtime_integration::MlxRuntime;
use mlx_bridge::MlxArray;

use super::bidirectional_lstm::BidirectionalLstm;
use super::convolutions::{ConvolutionGeometry, KokoroConv1d};
use super::style_modulation::StyleModulation;
use super::style_residual_block::{StyleBlockShape, StyleResidualBlock};
use crate::speech_models::checkpoint_tensors::CheckpointTensors;
use crate::speech_models::mlx_layers::linear::Linear;
use crate::speech_models::speech_model_error::SpeechModelError;

const DURATION_ENCODER_LAYER_COUNT: usize = 3;
const PREDICTOR_WIDTH: i32 = 512;

/// One `DurationEncoder` layer: a bidirectional LSTM, then AdaLayerNorm.
struct DurationEncoderLayer {
    recurrence: BidirectionalLstm,
    style_norm: StyleModulation,
}

pub(crate) struct ProsodyPredictor {
    /// `bert_encoder`: ALBERT's 768-wide states down to the predictor's 512.
    context_projection: Linear,
    duration_encoder_layers: Vec<DurationEncoderLayer>,
    duration_recurrence: BidirectionalLstm,
    duration_buckets: Linear,
    shared_recurrence: BidirectionalLstm,
    pitch_blocks: Vec<StyleResidualBlock>,
    energy_blocks: Vec<StyleResidualBlock>,
    pitch_projection: KokoroConv1d,
    energy_projection: KokoroConv1d,
}

/// The duration stage's results: the style-conditioned phoneme states the
/// alignment spreads over frames, and each phoneme's frame count.
pub(crate) struct PhonemeDurations {
    pub(crate) duration_states: MlxArray,
    pub(crate) frame_counts: Vec<i32>,
}

impl ProsodyPredictor {
    pub(crate) fn load(checkpoint: &CheckpointTensors<'_>) -> Result<Self, SpeechModelError> {
        let pointwise = ConvolutionGeometry::same_length(1, 1);
        Ok(Self {
            context_projection: Linear::load(checkpoint, "bert_encoder")?,
            duration_encoder_layers: load_duration_encoder_layers(checkpoint)?,
            duration_recurrence: BidirectionalLstm::load(checkpoint, "predictor.lstm")?,
            duration_buckets: Linear::load(checkpoint, "predictor.duration_proj.linear_layer")?,
            shared_recurrence: BidirectionalLstm::load(checkpoint, "predictor.shared")?,
            pitch_blocks: load_curve_blocks(checkpoint, "predictor.F0")?,
            energy_blocks: load_curve_blocks(checkpoint, "predictor.N")?,
            pitch_projection: KokoroConv1d::load(checkpoint, "predictor.F0_proj", pointwise)?,
            energy_projection: KokoroConv1d::load(checkpoint, "predictor.N_proj", pointwise)?,
        })
    }

    /// Frame counts per phoneme. Each of the 50 duration buckets votes with a
    /// sigmoid; their sum, divided by `speed`, rounds half to even (as
    /// `torch.round` does) to at least one frame.
    pub(crate) fn durations(
        &self,
        runtime: &MlxRuntime,
        albert_states: &MlxArray,
        prosody_style: &MlxArray,
        speed: f32,
    ) -> Result<PhonemeDurations, SpeechModelError> {
        let projected = self.context_projection.forward(runtime, albert_states)?;
        let duration_states = self.encode_durations(runtime, &projected, prosody_style)?;
        let recurrent = self
            .duration_recurrence
            .forward(runtime, &duration_states)?;
        let bucket_votes = runtime.sigmoid(&self.duration_buckets.forward(runtime, &recurrent)?)?;
        let expected_frames = runtime.sum_axis(&bucket_votes, -1, false)?;
        let frame_counts = runtime
            .array_to_vec_f32(&expected_frames)?
            .into_iter()
            .map(|frame_estimate| ((frame_estimate / speed).round_ties_even() as i32).max(1))
            .collect();
        Ok(PhonemeDurations {
            duration_states,
            frame_counts,
        })
    }

    /// StyleTTS2's `DurationEncoder`: three LSTM and AdaLayerNorm pairs, the
    /// style appended after each, giving `[length, 512 + 128]`.
    fn encode_durations(
        &self,
        runtime: &MlxRuntime,
        projected: &MlxArray,
        prosody_style: &MlxArray,
    ) -> Result<MlxArray, SpeechModelError> {
        let sequence_length = projected.shape()[0];
        let style_per_step =
            runtime.broadcast_to(prosody_style, &[sequence_length, prosody_style.shape()[1]])?;
        let mut encoded = runtime.concatenate_axis(&[projected, &style_per_step], 1)?;
        for layer in &self.duration_encoder_layers {
            let recurrent = layer.recurrence.forward(runtime, &encoded)?;
            let modulated =
                layer
                    .style_norm
                    .channel_modulate(runtime, &recurrent, prosody_style)?;
            encoded = runtime.concatenate_axis(&[&modulated, &style_per_step], 1)?;
        }
        Ok(encoded)
    }

    /// Pitch and energy, `[2 * frames, 1]` each: the curves run at twice the
    /// frame rate.
    pub(crate) fn pitch_and_energy(
        &self,
        runtime: &MlxRuntime,
        aligned_states: &MlxArray,
        prosody_style: &MlxArray,
    ) -> Result<(MlxArray, MlxArray), SpeechModelError> {
        let shared = self.shared_recurrence.forward(runtime, aligned_states)?;
        let pitch = run_blocks(runtime, &self.pitch_blocks, &shared, prosody_style)?;
        let energy = run_blocks(runtime, &self.energy_blocks, &shared, prosody_style)?;
        Ok((
            self.pitch_projection.forward(runtime, &pitch)?,
            self.energy_projection.forward(runtime, &energy)?,
        ))
    }
}

fn run_blocks(
    runtime: &MlxRuntime,
    blocks: &[StyleResidualBlock],
    input: &MlxArray,
    prosody_style: &MlxArray,
) -> Result<MlxArray, SpeechModelError> {
    let mut curve_states = input.retain()?;
    for block in blocks {
        curve_states = block.forward(runtime, &curve_states, prosody_style)?;
    }
    Ok(curve_states)
}

/// The LSTMs are the even entries of `predictor.text_encoder.lstms` and the
/// AdaLayerNorms the odd ones, as StyleTTS2 interleaves them in one list.
fn load_duration_encoder_layers(
    checkpoint: &CheckpointTensors<'_>,
) -> Result<Vec<DurationEncoderLayer>, SpeechModelError> {
    let mut layers = Vec::with_capacity(DURATION_ENCODER_LAYER_COUNT);
    for layer_index in 0..DURATION_ENCODER_LAYER_COUNT {
        let lstm_prefix = format!("predictor.text_encoder.lstms.{}", 2 * layer_index);
        let norm_prefix = format!("predictor.text_encoder.lstms.{}", 2 * layer_index + 1);
        layers.push(DurationEncoderLayer {
            recurrence: BidirectionalLstm::load(checkpoint, &lstm_prefix)?,
            style_norm: StyleModulation::load(checkpoint, &norm_prefix, PREDICTOR_WIDTH)?,
        });
    }
    Ok(layers)
}

/// 512 to 512, 512 to 256 at double length, 256 to 256.
fn load_curve_blocks(
    checkpoint: &CheckpointTensors<'_>,
    prefix: &str,
) -> Result<Vec<StyleResidualBlock>, SpeechModelError> {
    let half_width = PREDICTOR_WIDTH / 2;
    let block_shapes = [
        StyleBlockShape {
            input_channels: PREDICTOR_WIDTH,
            output_channels: PREDICTOR_WIDTH,
            doubles_length: false,
        },
        StyleBlockShape {
            input_channels: PREDICTOR_WIDTH,
            output_channels: half_width,
            doubles_length: true,
        },
        StyleBlockShape {
            input_channels: half_width,
            output_channels: half_width,
            doubles_length: false,
        },
    ];
    let mut blocks = Vec::with_capacity(block_shapes.len());
    for (block_index, block_shape) in block_shapes.into_iter().enumerate() {
        blocks.push(StyleResidualBlock::load(
            checkpoint,
            &format!("{prefix}.{block_index}"),
            block_shape,
        )?);
    }
    Ok(blocks)
}
