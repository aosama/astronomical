//! Multi-token prediction prompt-history initialization.

use astronomical_runtime_integration::MlxArray;

use crate::qwen3_5::inference_execution::engine_request::Qwen3_5EngineRequest;
use crate::qwen3_5::model::{Qwen3_5ExecutionError, Qwen3_5Model};

pub(in crate::qwen3_5) fn execute_terminal_optional_history_capture_with_performance_attribution(
    model: &Qwen3_5Model,
    prefill_start: usize,
    prefill_end: usize,
    active_request: &mut Qwen3_5EngineRequest,
) -> Result<usize, Qwen3_5ExecutionError> {
    let target_prefill_output = active_request
        .with_input_token_range_and_decoder_state_and_performance_attribution(
            prefill_start,
            prefill_end,
            |prompt_token_ids, next_position_tokens, request_decoder_state, performance_attribution| {
                model.forward_chunk_with_pre_final_normalization_hidden_states_and_performance_attribution(
                    prompt_token_ids,
                    next_position_tokens,
                    request_decoder_state,
                    performance_attribution,
                )
            },
        )
        .ok_or(Qwen3_5ExecutionError::InvalidInput {
            description: "terminal prompt range disappeared during MTP capture",
        })??;
    let shifted_prompt_start =
        prefill_start
            .checked_add(1)
            .ok_or(Qwen3_5ExecutionError::InvalidInput {
                description: "shifted prompt start overflowed",
            })?;
    let shifted_prompt_end =
        prefill_end
            .checked_add(1)
            .ok_or(Qwen3_5ExecutionError::InvalidInput {
                description: "shifted prompt end overflowed",
            })?;
    let history_initialization_started_at = active_request
        .performance_attribution_mut()
        .begin_operation_span();
    let history_initialization_outcome =
        initialize_prompt_history_for_prompt_range_with_performance_attribution(
            model,
            target_prefill_output.pre_final_normalization_hidden_states(),
            shifted_prompt_start,
            shifted_prompt_end,
            active_request,
        );
    active_request
        .performance_attribution_mut()
        .complete_operation_span(
            crate::PerformanceOperation::MtpPromptHistoryInitializationSpan,
            history_initialization_started_at,
        );
    history_initialization_outcome?;
    Ok(shifted_prompt_end.saturating_sub(shifted_prompt_start))
}

pub(in crate::qwen3_5) fn record_prompt_history_initialization_fallback(
    active_request: &mut Qwen3_5EngineRequest,
) {
    active_request.performance_attribution_mut().record_counter(
        crate::PerformanceCounter::MtpPromptHistoryInitializationFallbackCount,
        1,
    );
}

pub(in crate::qwen3_5) fn record_terminal_history_token_count(
    active_request: &mut Qwen3_5EngineRequest,
    terminal_history_token_count: usize,
) {
    active_request.performance_attribution_mut().record_counter(
        crate::PerformanceCounter::SpeculativePrefillTerminalMtpHistoryTokenCount,
        u64::try_from(terminal_history_token_count).unwrap_or(u64::MAX),
    );
}

pub(in crate::qwen3_5) fn initialize_prompt_history_from_token_ids_with_performance_attribution(
    model: &Qwen3_5Model,
    target_pre_final_normalization_hidden_states: &MlxArray,
    shifted_prompt_token_ids: &[u32],
    active_request: &mut Qwen3_5EngineRequest,
) -> Result<(), Qwen3_5ExecutionError> {
    let initialization_outcome = active_request
        .with_optional_prediction_session_and_performance_attribution(
            |multi_token_prediction_request, performance_attribution| {
                model.prefill_mtp_history_from_token_ids_with_performance_attribution(
                    target_pre_final_normalization_hidden_states,
                    shifted_prompt_token_ids,
                    multi_token_prediction_request.request_state_mut(),
                    performance_attribution,
                )
            },
        );
    initialization_outcome.ok_or(Qwen3_5ExecutionError::InvalidInput {
        description: "MTP request session disappeared during prompt-history initialization",
    })?
}

pub(in crate::qwen3_5) fn initialize_prompt_history_for_prompt_range_with_performance_attribution(
    model: &Qwen3_5Model,
    target_pre_final_normalization_hidden_states: &MlxArray,
    shifted_prompt_start: usize,
    shifted_prompt_end: usize,
    active_request: &mut Qwen3_5EngineRequest,
) -> Result<(), Qwen3_5ExecutionError> {
    let initialization_outcome = active_request
        .with_input_token_range_and_optional_prediction_session_and_performance_attribution(
            shifted_prompt_start,
            shifted_prompt_end,
            |shifted_prompt_token_ids, multi_token_prediction_request, performance_attribution| {
                model.prefill_mtp_history_from_token_ids_with_performance_attribution(
                    target_pre_final_normalization_hidden_states,
                    shifted_prompt_token_ids,
                    multi_token_prediction_request.request_state_mut(),
                    performance_attribution,
                )
            },
        );
    initialization_outcome.ok_or(Qwen3_5ExecutionError::InvalidInput {
        description: "MTP request session or prompt range disappeared during prompt-history initialization",
    })?
}
