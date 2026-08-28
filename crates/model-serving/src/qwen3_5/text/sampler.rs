use astronomical_runtime_integration::MlxArray;

use crate::InferenceEngineError;
use crate::gpu_token_sampling::build_sampled_token;

use super::super::inference_execution::{fatal_engine_error, qwen3_5_runtime_error};
use super::Qwen3_5Model;

pub use crate::gpu_token_sampling::apply_top_p_mask as qwen3_5_apply_top_p_mask;

pub(in crate::qwen3_5) fn random_state_for_seed(
    model: &Qwen3_5Model,
    seed: u64,
) -> Result<MlxArray, InferenceEngineError> {
    model
        .runtime
        .random_key(seed)
        .map_err(qwen3_5_runtime_error)
}

pub(in crate::qwen3_5) fn validate_sampled_strategy(
    temperature_thousandths: u16,
    top_k: u16,
    top_p_thousandths: u16,
) -> Result<(), InferenceEngineError> {
    if temperature_thousandths == 0 || top_k == 0 || top_p_thousandths > 1_000 {
        return Err(fatal_engine_error(
            "sampled Qwen3.5 strategy must use positive temperature, positive top-k, and top-p at most 1.0",
        ));
    }
    Ok(())
}

/// Builds one lazy Qwen3.5 sample using the shared GPU top-k + top-p pipeline.
pub(in crate::qwen3_5) fn build_qwen3_5_sampled_token(
    model: &Qwen3_5Model,
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
        model.config.vocabulary_size(),
    )?;
    build_sampled_token(
        &model.runtime,
        final_logits,
        temperature_thousandths,
        top_p_thousandths,
        Some(top_k),
        random_state,
    )
}

fn validate_sampling_arguments(
    final_logits: &MlxArray,
    temperature_thousandths: u16,
    top_p_thousandths: u16,
    top_k: u16,
    vocabulary_size: u32,
) -> Result<(), InferenceEngineError> {
    let vocabulary_size = i32::try_from(vocabulary_size)
        .map_err(|_| fatal_engine_error("model vocabulary exceeds the MLX shape range"))?;
    if final_logits.shape() != [1, 1, vocabulary_size] {
        return Err(fatal_engine_error(format!(
            "sampled Qwen3.5 logits must have shape [1, 1, {vocabulary_size}]"
        )));
    }
    if temperature_thousandths == 0 {
        return Err(fatal_engine_error(
            "sampled Qwen3.5 temperature must be positive",
        ));
    }
    if top_p_thousandths > 1_000 {
        return Err(fatal_engine_error(
            "sampled Qwen3.5 top-p must not exceed 1.0",
        ));
    }
    if top_k == 0 {
        return Err(fatal_engine_error("sampled Qwen3.5 top-k must be positive"));
    }
    Ok(())
}
