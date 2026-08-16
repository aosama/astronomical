//! Multi-token prediction feedback-injection handling.

use crate::InferenceEngineError;
use crate::qwen3_5::inference_execution::engine_request::Qwen3_5EngineRequest;
use crate::qwen3_5::inference_execution::{fatal_engine_error, qwen3_5_runtime_error};
use crate::qwen3_5::model::Qwen3_5Model;

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
    let public_verified_frontier = active_request
        .optional_prediction_session_mut()
        .and_then(|multi_token_prediction_request| {
            multi_token_prediction_request.take_public_verified_frontier()
        })
        .ok_or_else(|| {
            fatal_engine_error("queued prediction drafts lost their public target frontier")
        })?;
    active_request.measure_operation_with_request(
        crate::PerformanceOperation::MtpQueuedFrontierRestoration,
        |active_request| {
            active_request
                .request_decoder_state_mut()
                .restore_verified_prefix(
                    public_verified_frontier.position_tokens,
                    public_verified_frontier.boundary,
                )
                .map_err(qwen3_5_runtime_error)
        },
    )?;
    active_request.set_next_position_tokens(public_verified_frontier.position_tokens);
    active_request.performance_attribution_mut().record_counter(
        crate::PerformanceCounter::MtpQueuedFrontierRestorationCount,
        1,
    );
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
