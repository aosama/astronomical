use mlx_bridge::MlxArray;

use crate::InferenceEngineError;
use crate::gpu_token_sampling;

use crate::model::TextServingModel;

pub fn random_state_for_seed(
    model: &impl TextServingModel,
    seed: u64,
) -> Result<MlxArray, InferenceEngineError> {
    model
        .runtime()
        .random_key(seed)
        .map_err(sampler_runtime_error)
}

pub fn validate_sampled_strategy(
    temperature_thousandths: u16,
    top_k: u16,
    top_p_thousandths: u16,
) -> Result<(), InferenceEngineError> {
    if temperature_thousandths == 0 || top_k == 0 || top_p_thousandths > 1_000 {
        return Err(sampler_fatal_error(
            "sampled strategy must use positive temperature, positive top-k, and top-p at most 1.0",
        ));
    }
    Ok(())
}

/// Builds one lazy sample using the shared GPU top-k + top-p pipeline.
pub fn build_sampled_token_for_model(
    model: &impl TextServingModel,
    final_logits: &MlxArray,
    temperature_thousandths: u16,
    top_p_thousandths: u16,
    top_k: u16,
    random_state: &mut MlxArray,
) -> Result<MlxArray, InferenceEngineError> {
    validate_sampling_arguments(
        final_logits,
        temperature_thousandths,
        top_p_thousandths,
        top_k,
        model.vocabulary_size(),
    )?;
    gpu_token_sampling::build_sampled_token(
        model.runtime(),
        final_logits,
        temperature_thousandths,
        top_p_thousandths,
        Some(top_k),
        random_state,
    )
    .map_err(InferenceEngineError::from)
}

fn validate_sampling_arguments(
    final_logits: &MlxArray,
    temperature_thousandths: u16,
    top_p_thousandths: u16,
    top_k: u16,
    vocabulary_size: u32,
) -> Result<(), InferenceEngineError> {
    let vocabulary_size = i32::try_from(vocabulary_size)
        .map_err(|_| sampler_fatal_error("model vocabulary exceeds the MLX shape range"))?;
    if final_logits.shape() != [1, 1, vocabulary_size] {
        return Err(sampler_fatal_error(format!(
            "sampled logits must have shape [1, 1, {vocabulary_size}]"
        )));
    }
    if temperature_thousandths == 0 {
        return Err(sampler_fatal_error("sampled temperature must be positive"));
    }
    if top_p_thousandths > 1_000 {
        return Err(sampler_fatal_error("sampled top-p must not exceed 1.0"));
    }
    if top_k == 0 {
        return Err(sampler_fatal_error("sampled top-k must be positive"));
    }
    Ok(())
}

fn sampler_runtime_error(runtime_error: impl std::fmt::Display) -> InferenceEngineError {
    sampler_fatal_error(runtime_error.to_string())
}

fn sampler_fatal_error(reason: impl Into<String>) -> InferenceEngineError {
    InferenceEngineError::Fatal {
        reason: reason.into(),
    }
}
