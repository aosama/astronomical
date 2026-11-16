//! The distribution the sampler draws each row from, for callers that need
//! the probabilities rather than a draw: speculative verification compares a
//! drafter's proposals against exactly this law, so a drafted reply is
//! distributed as an undrafted one. Keep it in step with the sampler.

use astronomical_runtime_integration::MlxRuntime;
use mlx_bridge::{MlxArray, MlxDtype};

use super::vocabulary_top_k::MAXIMUM_TOP_K;
use super::{
    apply_top_p_mask, sampling_failure, sampling_runtime_error, softmax_last_axis,
    validate_sampled_logits, vocabulary_top_k,
};
use crate::InferenceEngineError;

/// The distribution the sampler draws each row from, restricted to the
/// row's `top_k` most likely tokens: ([1, rows, top_k] uint32 ids,
/// [1, rows, top_k] float32 probabilities). The sampler's law is a top-k and
/// top-p mask on the untempered logits, then the temperature, then softmax
/// (`masked_logits_after_top_k_and_top_p`); its top-k mask gives every other
/// token probability zero, so this is the whole distribution. Speculative
/// verification compares drafts against exactly this law, so a drafted reply
/// is distributed as the undrafted one: keep the two in step.
///
/// It runs the law's own steps in the law's own dtype, on the kept tokens
/// only after the one full-vocabulary softmax the top-p mask reads. The law
/// as written scatters the masked tokens back over the vocabulary, takes a
/// second softmax and a second partition there, which cost 6.6 ms for 16
/// rows of the 27B's 248,320-token vocabulary on an M5 Pro.
pub(crate) fn sampling_distribution_rows(
    runtime: &MlxRuntime,
    logits: &MlxArray,
    temperature_thousandths: u16,
    top_p_thousandths: u16,
    top_k: u16,
) -> Result<(MlxArray, MlxArray), InferenceEngineError> {
    let vocabulary_size =
        validate_sampled_logits(logits, temperature_thousandths, top_p_thousandths)?;
    if top_k == 0 {
        return Err(sampling_failure("sampled top-k must be positive"));
    }
    let kept = kept_top_k(runtime, logits, i32::from(top_k).min(vocabulary_size))?;
    let masked_logits = apply_top_p_mask(
        runtime,
        &kept.full_vocabulary_probabilities,
        &kept.logits,
        top_p_thousandths,
    )?;
    let temperature = f32::from(temperature_thousandths) / 1_000.0;
    let tempered_logits = runtime
        .multiply_scalar(&masked_logits, temperature.recip())
        .map_err(sampling_runtime_error)?;
    let probabilities = runtime
        .astype(
            &softmax_last_axis(runtime, &tempered_logits)?,
            MlxDtype::Float32,
        )
        .map_err(sampling_runtime_error)?;
    Ok((kept.token_ids, probabilities))
}

/// Each row's `kept_count` largest-logit ids: the one-pass top-k kernel for
/// the sampler's usual few tokens, MLX's argpartition for a larger top-k.
fn largest_logit_ids(
    runtime: &MlxRuntime,
    logits: &MlxArray,
    kept_count: i32,
) -> Result<MlxArray, InferenceEngineError> {
    if kept_count <= MAXIMUM_TOP_K {
        return vocabulary_top_k(runtime, logits, kept_count);
    }
    let row_count = logits.shape()[1];
    let partitioned = runtime
        .argpartition_axis(
            &runtime.negative(logits).map_err(sampling_runtime_error)?,
            kept_count - 1,
            -1,
        )
        .map_err(sampling_runtime_error)?;
    runtime
        .slice(
            &partitioned,
            &[0, 0, 0],
            &[1, row_count, kept_count],
            &[1, 1, 1],
        )
        .map_err(sampling_runtime_error)
}

/// One row's `top_k` most likely tokens, chosen as the law chooses them.
struct KeptTopK {
    token_ids: MlxArray,
    logits: MlxArray,
    /// Their probabilities under the whole vocabulary's softmax.
    full_vocabulary_probabilities: MlxArray,
}

fn kept_top_k(
    runtime: &MlxRuntime,
    logits: &MlxArray,
    kept_count: i32,
) -> Result<KeptTopK, InferenceEngineError> {
    let probabilities = softmax_last_axis(runtime, logits)?;
    let token_ids = largest_logit_ids(runtime, logits, kept_count)?;
    Ok(KeptTopK {
        logits: runtime
            .take_along_axis(logits, &token_ids, -1)
            .map_err(sampling_runtime_error)?,
        full_vocabulary_probabilities: runtime
            .take_along_axis(&probabilities, &token_ids, -1)
            .map_err(sampling_runtime_error)?,
        token_ids,
    })
}

#[cfg(test)]
#[path = "sampling_distribution_tests.rs"]
mod sampling_distribution_tests;
