//! Injected-token forwarding kept separate from ordinary autoregressive decode orchestration.

use astronomical_ipc_protocol::RequestId;

use crate::{AdaptiveRamGrowthPhase, InferenceEngineError};

use super::execution::LagunaInferenceExecution;
use super::memory::complete_laguna_forward_memory_observation;

pub(super) fn inject_input_tokens(
    execution: &mut LagunaInferenceExecution,
    request_id: RequestId,
    input_token_ids: Vec<u32>,
) -> Result<(), InferenceEngineError> {
    if input_token_ids.is_empty() {
        return Ok(());
    }
    let Some(runtime) = execution.runtime.as_ref() else {
        return Err(InferenceEngineError::Fatal {
            reason: "the Laguna runtime is not loaded".to_owned(),
        });
    };
    let Some(model) = execution.model.as_mut() else {
        return Err(InferenceEngineError::Fatal {
            reason: "the Laguna model is not loaded".to_owned(),
        });
    };
    let mlx_ram_budget = execution
        .mlx_ram_budget
        .as_mut()
        .ok_or(InferenceEngineError::Fatal {
            reason: "the Laguna RAM budget is not loaded".to_owned(),
        })?;
    let adaptive_ram_growth_guard =
        execution
            .adaptive_ram_growth_guard
            .as_mut()
            .ok_or(InferenceEngineError::Fatal {
                reason: "the Laguna adaptive RAM growth guard is not loaded".to_owned(),
            })?;
    let active_request =
        execution
            .active_request
            .as_mut()
            .ok_or(InferenceEngineError::InvalidRequest {
                reason: "no Laguna generation is active".to_owned(),
            })?;
    if active_request.request_id != request_id {
        return Err(InferenceEngineError::InvalidRequest {
            reason: "Laguna generation request identifiers do not match".to_owned(),
        });
    }
    let injected_token_array = runtime
        .array_from_u32(
            &input_token_ids,
            &[i32::try_from(input_token_ids.len()).unwrap_or(i32::MAX)],
        )
        .map_err(|_| InferenceEngineError::Fatal {
            reason: "Laguna injected tokens could not be placed on the runtime".to_owned(),
        })?;
    let context_token_count_after_forward = active_request
        .context_token_count
        .saturating_add(u64::try_from(input_token_ids.len()).unwrap_or(u64::MAX));
    let (adaptive_ram_growth_context, memory_baseline) =
        super::memory::admit_laguna_forward_memory(
            runtime,
            model,
            adaptive_ram_growth_guard,
            &active_request.decoder_state,
            input_token_ids.len(),
            0,
            AdaptiveRamGrowthPhase::Prefill,
            context_token_count_after_forward,
            &mut active_request.performance_attribution,
        )?;
    let injected_output = model
        .forward_prefill(
            runtime,
            &injected_token_array,
            &mut active_request.decoder_state,
            &mut active_request.performance_attribution,
        )
        .map_err(|injection_error| InferenceEngineError::Fatal {
            reason: format!("Laguna injected-token processing failed: {injection_error}"),
        })?;
    runtime
        .evaluate_arrays(&[&injected_output])
        .map_err(|evaluation_error| InferenceEngineError::Fatal {
            reason: format!("Laguna injected-token materialization failed: {evaluation_error}"),
        })?;
    complete_laguna_forward_memory_observation(
        runtime,
        model,
        adaptive_ram_growth_guard,
        adaptive_ram_growth_context,
        mlx_ram_budget,
        memory_baseline,
        context_token_count_after_forward,
        &mut active_request.performance_attribution,
    )?;
    if let Some(last_injected_token_id) = input_token_ids.last().copied() {
        active_request.next_input_token_ids = vec![last_injected_token_id];
    }
    active_request.context_token_count = context_token_count_after_forward;
    Ok(())
}
