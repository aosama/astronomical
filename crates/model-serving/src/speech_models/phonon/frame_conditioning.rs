//! What Phonon adds to every generated frame's input: the summed speaker
//! count and duration conditions, each through its projection.

use mlx_bridge::MlxArray;

use super::configuration::DurationConditioner;
use super::sinusoidal_embedding::sinusoidal_embedding;
use crate::speech_models::checkpoint_tensors::CheckpointTensors;
use crate::speech_models::mlx_layers::linear::Linear;
use crate::speech_models::speech_model_error::SpeechModelError;

pub(crate) const CONDITIONERS: &str = "flow_lm.condition_provider.conditioners";
/// The speaker-count conditioner's setting for one voice.
const ONE_SPEAKER: &str = "1";
/// The duration delta for the voice's natural pace.
const NATURAL_PACE_DURATION_DELTA: f32 = 0.0;

/// `[1, width]`: one speaker, natural pace.
pub(crate) fn frame_conditioning(
    checkpoint: &CheckpointTensors<'_>,
    speaker_count_values: &[String],
    duration_conditioner: &DurationConditioner,
) -> Result<MlxArray, SpeechModelError> {
    let runtime = checkpoint.runtime;
    let speaker_condition = speaker_count_condition(checkpoint, speaker_count_values)?;
    let duration_condition = duration_condition(checkpoint, duration_conditioner)?;
    let frame_conditioning = runtime.add(&speaker_condition, &duration_condition)?;
    runtime.evaluate_arrays(&[&frame_conditioning])?;
    Ok(frame_conditioning)
}

fn speaker_count_condition(
    checkpoint: &CheckpointTensors<'_>,
    speaker_count_values: &[String],
) -> Result<MlxArray, SpeechModelError> {
    let runtime = checkpoint.runtime;
    let speaker_index = speaker_count_values
        .iter()
        .position(|speaker_count| speaker_count == ONE_SPEAKER)
        .ok_or_else(|| {
            SpeechModelError::Artifact(format!("num_speakers does not offer {ONE_SPEAKER:?}"))
        })?;
    let speaker_table = checkpoint.tensor(&format!("{CONDITIONERS}.num_speakers.embed.weight"))?;
    let speaker_row_index = runtime.array_from_i32(&[speaker_index as i32], &[1])?;
    let speaker_row = runtime.take_axis(&speaker_table, &speaker_row_index, 0)?;
    Linear::load(
        checkpoint,
        &format!("{CONDITIONERS}.num_speakers.output_proj"),
    )?
    .forward(runtime, &speaker_row)
}

fn duration_condition(
    checkpoint: &CheckpointTensors<'_>,
    duration_conditioner: &DurationConditioner,
) -> Result<MlxArray, SpeechModelError> {
    let runtime = checkpoint.runtime;
    let embedded_delta = duration_conditioner.scale_factor * NATURAL_PACE_DURATION_DELTA;
    let duration_values = sinusoidal_embedding(
        embedded_delta,
        duration_conditioner.width,
        duration_conditioner.max_period,
    )
    .ok_or_else(|| {
        SpeechModelError::Artifact(format!(
            "duration_delta width {} is not even and at least 4",
            duration_conditioner.width
        ))
    })?;
    let duration_row =
        runtime.array_from_f32(&duration_values, &[1, duration_conditioner.width as i32])?;
    Linear::load(
        checkpoint,
        &format!("{CONDITIONERS}.duration_delta.output_proj"),
    )?
    .forward(runtime, &duration_row)
}
