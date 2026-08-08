//! Multi-token prediction feedback-injection handling.

use crate::InferenceEngineError;
use crate::qwen3_5::inference_execution::engine_request::Qwen3_5EngineRequest;
use crate::qwen3_5::inference_execution::{fatal_engine_error, qwen3_5_runtime_error};
use crate::qwen3_5::model::Qwen3_5Model;
use astronomical_runtime_integration::MlxArray;

pub(in crate::qwen3_5) fn restore_queued_prediction_prefix_before_injection(
    active_request: &mut Qwen3_5EngineRequest,
) -> Result<bool, InferenceEngineError> {
    let has_queued_verified_token_ids = active_request
        .optional_prediction_session_mut()
        .is_some_and(|multi_token_prediction_request| {
            multi_token_prediction_request.has_verified_generated_token_ids()
        });
    if !has_queued_verified_token_ids {
        return Ok(false);
    }
    let accepted_draft_rollback = active_request
        .optional_prediction_session_mut()
        .and_then(|multi_token_prediction_request| {
            multi_token_prediction_request.accepted_draft_rollback()
        })
        .ok_or_else(|| {
            fatal_engine_error("queued prediction draft lost its target rollback checkpoint")
        })?;
    active_request
        .request_decoder_state_mut()
        .restore_verified_prefix(
            accepted_draft_rollback.verified_prefix_position_tokens,
            accepted_draft_rollback.verified_prefix_boundary_checkpoint,
        )
        .map_err(qwen3_5_runtime_error)?;
    active_request
        .set_next_position_tokens(accepted_draft_rollback.verified_prefix_position_tokens);
    Ok(true)
}

pub(in crate::qwen3_5) fn reset_prediction_after_injection(
    active_request: &mut Qwen3_5EngineRequest,
    full_attention_kv_state_growth_tokens: i32,
) -> Result<bool, InferenceEngineError> {
    let Some(multi_token_prediction_request) = active_request.optional_prediction_session_mut()
    else {
        return Ok(false);
    };
    multi_token_prediction_request.clear_verified_generated_token_ids();
    multi_token_prediction_request.set_target_hidden_states(None);
    multi_token_prediction_request
        .reset_history(full_attention_kv_state_growth_tokens)
        .map_err(qwen3_5_runtime_error)?;
    Ok(true)
}

pub(in crate::qwen3_5) fn disable_prediction_after_optional_injection_failure(
    active_request: &mut Qwen3_5EngineRequest,
) {
    active_request.clear_optional_prediction_session();
}

pub(in crate::qwen3_5) fn projected_injected_prediction_growth_bytes(
    model: &Qwen3_5Model,
    active_request: &Qwen3_5EngineRequest,
    update_token_count: usize,
) -> Result<usize, InferenceEngineError> {
    let Some(multi_token_prediction_request) = active_request.optional_prediction_session() else {
        return Ok(0);
    };
    let full_attention_bytes_per_layer_token = model
        .config()
        .full_attention_key_value_state_bytes_per_layer_token()
        .ok_or_else(|| {
            fatal_engine_error("prediction full-attention bytes per layer token overflowed")
        })?;
    multi_token_prediction_request
        .projected_full_attention_growth_bytes(
            full_attention_bytes_per_layer_token,
            update_token_count,
        )
        .map_err(qwen3_5_runtime_error)
}

pub(in crate::qwen3_5) fn reseed_prediction_after_injected_prefix(
    model: &Qwen3_5Model,
    active_request: &mut Qwen3_5EngineRequest,
    feedback_prefix_token_ids: &[u32],
    shifted_feedback_token_ids: &[u32],
) -> Result<(), InferenceEngineError> {
    let starting_position_tokens = active_request.next_position_tokens();
    let target_prefill_output = active_request
        .with_decoder_state_and_performance_attribution(
            |request_decoder_state, performance_attribution| {
                model.forward_chunk_with_pre_final_normalization_hidden_states_and_performance_attribution(
                    feedback_prefix_token_ids,
                    starting_position_tokens,
                    request_decoder_state,
                    performance_attribution,
                )
            },
        )
        .map_err(InferenceEngineError::from)?;
    crate::qwen3_5::multi_token_prediction::initialize_prompt_history_from_token_ids_with_performance_attribution(
        model,
        target_prefill_output.pre_final_normalization_hidden_states(),
        shifted_feedback_token_ids,
        active_request,
    )
    .map_err(InferenceEngineError::from)?;
    active_request
        .performance_attribution_mut()
        .record_counter(crate::PerformanceCounter::MtpFeedbackHistoryReseedCount, 1);
    Ok(())
}

pub(in crate::qwen3_5) fn forward_final_injected_prediction_token(
    model: &Qwen3_5Model,
    active_request: &mut Qwen3_5EngineRequest,
    final_input_token_id: u32,
) -> Result<Option<MlxArray>, InferenceEngineError> {
    if !active_request.has_optional_prediction_session() {
        return Ok(None);
    }
    let starting_position_tokens = active_request.next_position_tokens();
    let target_forward_output = active_request
        .with_decoder_state_and_performance_attribution(
            |request_decoder_state, performance_attribution| {
                model.forward_chunk_with_pre_final_normalization_hidden_states_and_performance_attribution(
                    &[final_input_token_id],
                    starting_position_tokens,
                    request_decoder_state,
                    performance_attribution,
                )
            },
        )
        .map_err(InferenceEngineError::from)?;
    active_request.advance_position(1)?;
    let next_generated_token =
        active_request.build_generated_token(model, target_forward_output.final_logits())?;
    if let Some(multi_token_prediction_request) = active_request.optional_prediction_session_mut() {
        multi_token_prediction_request.set_target_hidden_states(Some(
            target_forward_output.into_pre_final_normalization_hidden_states(),
        ));
    }
    Ok(Some(next_generated_token))
}
