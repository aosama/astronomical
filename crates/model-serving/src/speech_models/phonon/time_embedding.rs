//! The flow head's time conditioning: a time through a sinusoid, a two-layer
//! MLP and an RMS norm (`TimestepEmbedder`).
//!
//! Phonon samples in a single step from time 0 to time 1, so its time
//! conditioning never changes and is computed once at load: the average of
//! the start-time and end-time embeddings.

use astronomical_runtime_integration::MlxRuntime;
use mlx_bridge::MlxArray;

use crate::speech_models::checkpoint_tensors::CheckpointTensors;
use crate::speech_models::mlx_layers::linear::Linear;
use crate::speech_models::mlx_layers::scalar_arithmetic::add_scalar;
use crate::speech_models::speech_model_error::SpeechModelError;

const TIME_NORM_EPSILON: f32 = 1e-5;
const START_TIME: f32 = 0.0;
const END_TIME: f32 = 1.0;

/// `[1, width]`: the conditioning of the one step from time 0 to time 1.
pub(crate) fn single_step_time_embedding(
    checkpoint: &CheckpointTensors<'_>,
    prefix: &str,
) -> Result<MlxArray, SpeechModelError> {
    let runtime = checkpoint.runtime;
    let start_time_embedding = time_embedding(checkpoint, &format!("{prefix}.0"), START_TIME)?;
    let end_time_embedding = time_embedding(checkpoint, &format!("{prefix}.1"), END_TIME)?;
    let summed_embeddings = runtime.add(&start_time_embedding, &end_time_embedding)?;
    let averaged = runtime.multiply_scalar(&summed_embeddings, 0.5)?;
    runtime.evaluate_arrays(&[&averaged])?;
    Ok(averaged)
}

fn time_embedding(
    checkpoint: &CheckpointTensors<'_>,
    embedder_prefix: &str,
    time: f32,
) -> Result<MlxArray, SpeechModelError> {
    let runtime = checkpoint.runtime;
    let frequencies = checkpoint.tensor(&format!("{embedder_prefix}.freqs"))?;
    let arguments = runtime.expand_dims(&runtime.multiply_scalar(&frequencies, time)?, 0)?;
    let sinusoid =
        runtime.concatenate_axis(&[&runtime.cos(&arguments)?, &runtime.sin(&arguments)?], 1)?;
    let first_projection = Linear::load(checkpoint, &format!("{embedder_prefix}.mlp.0"))?;
    let second_projection = Linear::load(checkpoint, &format!("{embedder_prefix}.mlp.2"))?;
    let hidden = runtime.silu(&first_projection.forward(runtime, &sinusoid)?)?;
    let projected = second_projection.forward(runtime, &hidden)?;
    let norm_scale = checkpoint.tensor(&format!("{embedder_prefix}.mlp.3.alpha"))?;
    uncentred_rms_norm(runtime, &projected, &norm_scale)
}

/// `x * scale / sqrt(mean(x^2) + eps)`: Phonon's `rms` time norm, which,
/// unlike Pocket TTS's variance form, does not subtract the mean.
pub(crate) fn uncentred_rms_norm(
    runtime: &MlxRuntime,
    input: &MlxArray,
    norm_scale: &MlxArray,
) -> Result<MlxArray, SpeechModelError> {
    let mean_square = runtime.mean_axis(&runtime.multiply(input, input)?, -1, true)?;
    let inverse_root = runtime.rsqrt(&add_scalar(runtime, &mean_square, TIME_NORM_EPSILON)?)?;
    Ok(runtime.multiply(&runtime.multiply(input, &inverse_root)?, norm_scale)?)
}

#[cfg(test)]
#[path = "time_embedding_tests.rs"]
mod tests;
