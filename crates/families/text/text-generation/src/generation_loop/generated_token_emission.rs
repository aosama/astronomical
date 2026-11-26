//! Turning one sampled token id into the reply token the worker receives.
//!
//! Every text engine counts the token, moves the thinking budget and structured
//! generation on, decides whether the reply ends here and attaches the memory
//! telemetry of the forward that produced it. The steps are generic over the
//! model and the request's engine extension, so each engine's copy compiles to
//! the same code it ran before.

use mlx_bridge::MlxArray;
use runtime_integration::MlxMemorySnapshot;

use crate::engine_errors::{
    fatal_runtime_error, generated_token_counter_overflowed_error,
    mlx_active_memory_bytes_exceed_u64_range_error, mlx_buffer_cache_bytes_exceed_u64_range_error,
    mlx_peak_memory_bytes_exceed_u64_range_error,
};
use crate::engine_request::TextEngineRequest;
use crate::memory::telemetry::MlxMemoryTelemetry;
use crate::{GeneratedToken, InferenceEngineError, PerformanceCounter, PerformanceOperation};

use crate::model::text_model::TextModel;

pub struct GeneratedTokenEmission {
    pub generated_token: GeneratedToken,
    pub is_terminal: bool,
}

impl GeneratedTokenEmission {
    /// The decode step's answer for this emission: complete when the token
    /// ends the reply, otherwise continue.
    #[must_use]
    pub fn into_active_request_advance(self) -> super::decode_step::ActiveRequestAdvance {
        if self.is_terminal {
            super::decode_step::ActiveRequestAdvance::Complete(self.generated_token)
        } else {
            super::decode_step::ActiveRequestAdvance::Continue(self.generated_token)
        }
    }
}

/// Whether emitting `generated_token_id` next would end the reply, judged
/// before the token is counted.
pub fn generated_token_will_be_terminal<RequestExtension>(
    end_of_sequence_token_ids: &[u32],
    active_request: &TextEngineRequest<RequestExtension>,
    generated_token_id: u32,
) -> bool {
    end_of_sequence_token_ids.contains(&generated_token_id)
        || active_request.generated_token_count.saturating_add(1)
            >= active_request.maximum_output_tokens
}

/// Counts one emitted token and builds the reply token for it, with the
/// memory telemetry of the forward that produced it when there was one.
pub fn build_generated_token_emission<Model: TextModel, RequestExtension>(
    model: &Model,
    end_of_sequence_token_ids: &[u32],
    active_request: &mut TextEngineRequest<RequestExtension>,
    generated_token_id: u32,
    completed_forward_memory_snapshot: Option<&MlxMemorySnapshot>,
) -> Result<GeneratedTokenEmission, InferenceEngineError> {
    active_request.generated_token_count = active_request
        .generated_token_count
        .checked_add(1)
        .ok_or_else(generated_token_counter_overflowed_error)?;
    active_request
        .performance_attribution
        .record_counter(PerformanceCounter::GeneratedTokenCount, 1);

    let was_already_in_visible_answer = !active_request.is_inside_thinking();
    let is_reasoning_token = active_request.observe_committed_thinking_token(generated_token_id)?;
    if was_already_in_visible_answer
        && !is_reasoning_token
        && let Some(structured_generation) = active_request.structured_generation.as_mut()
    {
        structured_generation.accept_visible_token(generated_token_id);
    }

    let is_terminal = end_of_sequence_token_ids.contains(&generated_token_id)
        || active_request.generated_token_count >= active_request.maximum_output_tokens;
    let mlx_memory_telemetry = match completed_forward_memory_snapshot {
        Some(mlx_memory_snapshot) => Some(decode_mlx_memory_telemetry_from_snapshot(
            model,
            active_request,
            mlx_memory_snapshot,
        )?),
        None => None,
    };
    let generated_token = GeneratedToken::TokenId {
        token_id: generated_token_id,
        is_reasoning_token,
        expert_memory_mode: Some(model.observed_expert_memory_mode()),
        mlx_memory_telemetry,
        first_decode_forward_elapsed_millis: active_request
            .first_decode_forward_elapsed_millis
            .take(),
        generation_finalization: None,
    };
    Ok(GeneratedTokenEmission {
        generated_token,
        is_terminal,
    })
}

/// Reads a sampled token's id on the host, the one wait per reply token, and
/// records it for the repetition penalty when one is in use.
pub fn synchronize_generated_token_id<RequestExtension>(
    active_request: &mut TextEngineRequest<RequestExtension>,
    generated_token: &MlxArray,
) -> Result<u32, InferenceEngineError> {
    let synchronized_token_id = active_request
        .performance_attribution
        .measure_operation(
            PerformanceOperation::GeneratedTokenItemSynchronizationWait,
            |_performance_attribution| generated_token.item_u32(),
        )
        .map_err(fatal_runtime_error)?;
    if active_request.repetition_penalty_thousandths != 1_000 {
        active_request.observe_penalized_token_id(synchronized_token_id);
    }
    Ok(synchronized_token_id)
}

/// Builds the decode-phase MLX memory telemetry from one completed forward's
/// memory snapshot.
///
/// Extracted from the emission step so each snapshot conversion reads as a
/// named value instead of a nested `try_from`/`map_err` chain inside an
/// `Option::map` closure.
fn decode_mlx_memory_telemetry_from_snapshot<Model: TextModel, RequestExtension>(
    model: &Model,
    active_request: &TextEngineRequest<RequestExtension>,
    mlx_memory_snapshot: &MlxMemorySnapshot,
) -> Result<MlxMemoryTelemetry, InferenceEngineError> {
    let mlx_active_memory_bytes = u64::try_from(mlx_memory_snapshot.active_memory_bytes())
        .map_err(mlx_active_memory_bytes_exceed_u64_range_error)?;
    let active_memory_breakdown = model.active_memory_breakdown(
        &active_request.request_decoder_state,
        0,
        mlx_active_memory_bytes,
        0,
    );
    let mlx_buffer_cache_bytes = u64::try_from(mlx_memory_snapshot.mlx_buffer_cache_bytes())
        .map_err(mlx_buffer_cache_bytes_exceed_u64_range_error)?;
    let peak_memory_bytes = u64::try_from(mlx_memory_snapshot.peak_memory_bytes())
        .map_err(mlx_peak_memory_bytes_exceed_u64_range_error)?;
    // The breakdown reconciles the snapshot captured at one instant by
    // the observation owner, and the residency claim is derived from
    // that same breakdown, so this pair can never straddle a promotion
    // or demote (issue #337).
    Ok(MlxMemoryTelemetry::new(
        mlx_active_memory_bytes,
        mlx_buffer_cache_bytes,
        peak_memory_bytes,
        active_memory_breakdown,
    )
    .with_expert_residency_telemetry(
        model.expert_residency_telemetry_for_breakdown(&active_memory_breakdown),
    )
    // Issue #510: publish the same utilization split the decode
    // attribution records, so the live status shows why the
    // ceiling is not fully in use while generation runs.
    .with_memory_ceiling_utilization(
        model.memory_ceiling_utilization_for_breakdown(
            mlx_active_memory_bytes,
            active_memory_breakdown,
        ),
    )
    .with_memory_ledger_reading(model.latest_memory_ledger_reading()))
}

#[cfg(test)]
#[path = "generated_token_emission_tests.rs"]
mod generated_token_emission_tests;
