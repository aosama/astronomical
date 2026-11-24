//! The folded-ternary engine's answers to the shared Qwen 3.5 decode step and
//! finalization (issue #2473, Prism step 7).
//!
//! The shared step owns the reply rules: the end-of-sequence token is emitted
//! and finalizes the request, the output and thinking budgets are counted
//! there, and a failed step finalizes at once. What stays here is how a
//! folded-ternary step reads its prompt, runs a reply token's forward, and
//! keeps its decode speed: when nothing the host learns from reading a reply
//! token can change its successor, the successor's forward is submitted
//! before the read, so the graphics processor works while the host waits.

use std::time::Instant;

use ipc_protocol::{ExpertMemoryMode, RequestId};
use mlx_bridge::MlxArray;
use runtime_integration::MlxMemorySnapshot;

use super::{PrismHadamardQwen35ActiveGeneration, PrismHadamardQwen35InferenceExecution};
use crate::prism_hadamard_qwen35::model::PrismHadamardTextModel;
use crate::qwen3_5_core::model_math::error::Qwen3_5ExecutionError;
use crate::qwen3_5_core::shared_execution::decode_step::{
    ActiveRequestAdvance, Qwen3_5DecodeStepHooks,
};
use crate::qwen3_5_core::shared_execution::generation_finalization_steps::{
    GenerationAttributionRecord, Qwen3_5FinalizationHooks, collect_current_mlx_memory_telemetry,
    record_generation_performance_attribution,
};
use crate::qwen3_5_core::shared_execution::text_model::Qwen3_5TextModel;
use crate::{
    GeneratedToken, GenerationFinalization, InferenceEngineError, PerformanceAttribution,
    PerformanceAttributionOutcome, PerformanceOperation,
};

impl Qwen3_5FinalizationHooks for PrismHadamardQwen35InferenceExecution {
    /// The folded-ternary prefill measures no prompt working buffers.
    fn measured_prompt_mlx_working_buffer_bytes(&self) -> Option<u64> {
        None
    }

    /// Records the request's attribution with a memory reading taken while
    /// its decoder state is still held, as the folded-ternary engine always
    /// has, then releases the request and reports the idle memory.
    fn finalize_generation_request(
        &mut self,
        mut active_generation: PrismHadamardQwen35ActiveGeneration,
        performance_attribution_outcome: PerformanceAttributionOutcome,
        failure_description: Option<&'static str>,
    ) -> GenerationFinalization {
        let performance_attribution = std::mem::replace(
            &mut active_generation.performance_attribution,
            PerformanceAttribution::disabled(),
        );
        let mlx_memory_snapshot = self
            .execution_core
            .model
            .as_ref()
            .and_then(|model| model.runtime.memory_snapshot().ok());
        let attribution_record = GenerationAttributionRecord {
            performance_attribution,
            outcome: performance_attribution_outcome,
            request_id: active_generation.request_id,
            configured_maximum_output_tokens: active_generation.maximum_output_tokens,
            mlx_memory_snapshot,
            failure_description,
        };
        record_generation_performance_attribution(self, attribution_record);
        drop(active_generation);
        GenerationFinalization::new(
            Some(ExpertMemoryMode::Resident),
            idle_mlx_memory_telemetry(self),
            None,
        )
    }
}

impl Qwen3_5DecodeStepHooks for PrismHadamardQwen35InferenceExecution {
    fn advance_prompt_prefill_if_pending(
        &mut self,
        request_id: RequestId,
        active_generation: &mut PrismHadamardQwen35ActiveGeneration,
    ) -> Result<Option<GeneratedToken>, InferenceEngineError> {
        self.advance_prompt_prefill(request_id, active_generation)
    }

    /// The folded-ternary model is fully resident, so there is nothing to
    /// prepare between the prompt and the reply.
    fn begin_generation_preparation(
        &mut self,
        _request_id: RequestId,
        _active_generation: &mut PrismHadamardQwen35ActiveGeneration,
    ) -> Result<Option<ActiveRequestAdvance>, InferenceEngineError> {
        Ok(None)
    }

    fn submit_successor_before_reply_token_read(
        &mut self,
        active_generation: &mut PrismHadamardQwen35ActiveGeneration,
        current_generated_token: &MlxArray,
        is_forced_thinking_transition: bool,
    ) -> Result<Option<(MlxArray, MlxMemorySnapshot)>, InferenceEngineError> {
        if is_forced_thinking_transition
            || !successor_may_be_submitted_before_read(active_generation)
        {
            return Ok(None);
        }
        let model = self.execution_core.loaded_model()?;
        forward_and_submit_successor(model, active_generation, current_generated_token).map(Some)
    }

    /// The prefill's last chunk always samples the first reply token, so a
    /// decode step never starts without one.
    fn decode_first_token_from_final_prompt(
        &mut self,
        _request_id: RequestId,
        _active_generation: &mut PrismHadamardQwen35ActiveGeneration,
        _final_prompt_index: usize,
        _first_decode_forward_started_at: &mut Option<Instant>,
    ) -> Result<MlxArray, InferenceEngineError> {
        Err(InferenceEngineError::Fatal {
            reason: "folded-ternary decode step found no reply token sampled by the prefill"
                .to_owned(),
        })
    }

    fn decode_next_token(
        &mut self,
        active_generation: &mut PrismHadamardQwen35ActiveGeneration,
        current_generated_token: &MlxArray,
        _current_generated_token_id: u32,
    ) -> Result<(MlxArray, MlxMemorySnapshot), InferenceEngineError> {
        let model = self.execution_core.loaded_model()?;
        forward_and_submit_successor(model, active_generation, current_generated_token)
    }
}

/// Whether the successor may be submitted before the reply token is read:
/// no repetition history or structured mask the token would change, and the
/// token is not the last the output budget allows, so the successor is
/// wasted only when the token ends the reply by itself.
pub(super) fn successor_may_be_submitted_before_read(
    active_generation: &PrismHadamardQwen35ActiveGeneration,
) -> bool {
    active_generation.repetition_penalty_thousandths == 1_000
        && active_generation.structured_generation.is_none()
        && active_generation.generated_token_count.saturating_add(1)
            < active_generation.maximum_output_tokens
}

/// Runs the forward of `current_generated_token` (`[1, 1]`, on the device
/// or the host), samples its successor with the shared sampler, submits the
/// step with every retained state array, and reads the memory after it.
/// Nothing here waits for the graphics processor.
fn forward_and_submit_successor(
    model: &PrismHadamardTextModel,
    active_generation: &mut PrismHadamardQwen35ActiveGeneration,
    current_generated_token: &MlxArray,
) -> Result<(MlxArray, MlxMemorySnapshot), InferenceEngineError> {
    let successor_logits = active_generation
        .forward_pass(model, current_generated_token)
        .map_err(Qwen3_5ExecutionError::from)?;
    let successor = active_generation.build_generated_token(model, &successor_logits)?;
    active_generation
        .performance_attribution
        .measure_operation(
            PerformanceOperation::DecodeAsyncEvaluationSubmission,
            |_performance_attribution| {
                model
                    .async_evaluate_generation(&successor, &active_generation.request_decoder_state)
            },
        )?;
    let completed_forward_memory_snapshot = model.memory_snapshot_for_telemetry()?;
    Ok((successor, completed_forward_memory_snapshot))
}

/// The memory a finished request leaves, as the shared finalization reports
/// it; an unreadable reading is reported as none rather than failing the
/// finalization.
fn idle_mlx_memory_telemetry(
    execution: &PrismHadamardQwen35InferenceExecution,
) -> Option<crate::MlxMemoryTelemetry> {
    match collect_current_mlx_memory_telemetry(execution) {
        Ok(mlx_memory_telemetry) => mlx_memory_telemetry,
        Err(telemetry_error) => {
            tracing::warn!(
                error = %telemetry_error,
                "folded-ternary execution could not sample memory after a request"
            );
            None
        }
    }
}

#[cfg(test)]
#[path = "decode_hooks_tests.rs"]
mod decode_hooks_tests;
