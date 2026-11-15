//! The distribution the sampler draws each row from, for callers that need
//! the probabilities rather than a draw: speculative verification compares a
//! drafter's proposals against exactly this law, so a drafted reply is
//! distributed as an undrafted one. Keep it in step with the sampler.

use astronomical_runtime_integration::MlxRuntime;
use mlx_bridge::MlxArray;

use super::{
    masked_logits_after_top_k_and_top_p, sampling_runtime_error, softmax_last_axis,
    validate_sampled_logits,
};
use crate::InferenceEngineError;

/// The probabilities the sampler draws from, per row: the same top-k and
/// top-p mask on the untempered logits, then the temperature, then softmax.
/// Speculative verification compares drafts against exactly this law, so a
/// drafted reply is distributed as the undrafted one; keep the two in step.
fn sampling_probability_rows(
    runtime: &MlxRuntime,
    logits: &MlxArray,
    temperature_thousandths: u16,
    top_p_thousandths: u16,
    top_k: u16,
) -> Result<MlxArray, InferenceEngineError> {
    let vocabulary_size =
        validate_sampled_logits(logits, temperature_thousandths, top_p_thousandths)?;
    let constrained_logits = masked_logits_after_top_k_and_top_p(
        runtime,
        logits,
        vocabulary_size,
        top_k,
        top_p_thousandths,
    )?;
    let temperature = f32::from(temperature_thousandths) / 1_000.0;
    let tempered_logits = runtime
        .multiply_scalar(&constrained_logits, temperature.recip())
        .map_err(sampling_runtime_error)?;
    softmax_last_axis(runtime, &tempered_logits)
}

/// `sampling_probability_rows` cut to each row's `top_k` most likely tokens:
/// ([1, rows, top_k] uint32 ids, [1, rows, top_k] float32 probabilities).
/// The sampler's top-k mask gives every other token probability zero, so
/// this is the whole distribution in a few values a row; speculative
/// verification reads back these instead of full vocabulary rows.
pub(crate) fn sampling_distribution_rows(
    runtime: &MlxRuntime,
    logits: &MlxArray,
    temperature_thousandths: u16,
    top_p_thousandths: u16,
    top_k: u16,
) -> Result<(MlxArray, MlxArray), InferenceEngineError> {
    let probabilities = sampling_probability_rows(
        runtime,
        logits,
        temperature_thousandths,
        top_p_thousandths,
        top_k,
    )?;
    let shape = probabilities.shape();
    let (row_count, vocabulary_size) = (shape[1], shape[2]);
    let kept_count = i32::from(top_k).min(vocabulary_size);
    let first_kept_rank = vocabulary_size - kept_count;
    let partitioned = runtime
        .argpartition_axis(&probabilities, first_kept_rank, -1)
        .map_err(sampling_runtime_error)?;
    let token_ids = runtime
        .slice(
            &partitioned,
            &[0, 0, first_kept_rank],
            &[1, row_count, vocabulary_size],
            &[1, 1, 1],
        )
        .map_err(sampling_runtime_error)?;
    let kept_probabilities = runtime
        .take_along_axis(&probabilities, &token_ids, -1)
        .map_err(sampling_runtime_error)?;
    let single_precision = runtime
        .astype(&kept_probabilities, mlx_bridge::MlxDtype::Float32)
        .map_err(sampling_runtime_error)?;
    Ok((token_ids, single_precision))
}
