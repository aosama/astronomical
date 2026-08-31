//! Thin orchestrator for a single prompt-processing chunk.
//!
//! Calls the three phases in order:
//! 1. `plan_prompt_prefill_chunk` — pure decision logic (no side effects).
//! 2. `execute_prefill_memory_admission` — adaptive RAM growth.
//! 3. `dispatch_prefill_forward` — GPU forward dispatch (speculative / visual /
//!    history capture / plain text).
//!
//! Collects results into `PromptPrefillChunkOutcome` for the caller in
//! `prefill_advance.rs`.

use astronomical_ipc_protocol::RequestId;

use super::engine_request::Qwen3_5EngineRequest;
use super::prefill_forward_dispatch::PrefillForwardError;
use super::prompt_prefill_errors::{
    PromptPrefillChunkAttemptError, configured_speculative_prefill_execution_error,
    prefill_execution_error,
};
use super::{Qwen3_5EngineState, fatal_engine_error, qwen3_5_runtime_error};

use crate::{PerformanceCounter, Qwen3_5PersistentPromptCacheBoundaryCheckpoint};

/// Outcome of executing one prompt-processing chunk.
pub(super) struct PromptPrefillChunkOutcome {
    pub(super) active_memory_bytes_before_growth: usize,
    pub(super) retained_expert_payload_bytes_before_growth: u64,
    pub(super) forward_chunk_elapsed_millis: u64,
    pub(super) adaptive_ram_growth_context: crate::AdaptiveRamGrowthContext,
    pub(super) exact_temporary_workspace_bytes: usize,
    pub(super) boundary_checkpoints: Vec<Qwen3_5PersistentPromptCacheBoundaryCheckpoint>,
}

impl Qwen3_5EngineState {
    /// Execute one prompt-processing chunk: plan → admit → forward.
    pub(super) fn execute_prompt_prefill_chunk(
        &mut self,
        request_id: RequestId,
        active_request: &mut Qwen3_5EngineRequest,
        prefill_start: usize,
        prefill_end: usize,
    ) -> Result<PromptPrefillChunkOutcome, PromptPrefillChunkAttemptError> {
        // Phase 1: pure decision logic — no side effects.
        let plan = self.plan_prompt_prefill_chunk(active_request, prefill_start, prefill_end)?;
        active_request
            .performance_attribution
            .record_counter(PerformanceCounter::PrefillChunkCount, 1);

        // Phase 2: adaptive RAM growth admission.
        let admission_outcome = self.execute_prefill_memory_admission(
            active_request,
            plan.adaptive_ram_growth_context,
            plan.additional_persistent_state_growth_bytes,
            plan.exact_temporary_workspace_bytes,
            plan.direct_publication_workspace_bytes,
        )?;

        if plan.speculative_prefill_target_is_active {
            active_request.performance_attribution.record_counter(
                PerformanceCounter::SpeculativePrefillContextTargetExpertReclaimedPayloadBytes,
                admission_outcome.target_expert_payload_bytes_reclaimed_during_context_admission,
            );
        }

        let prefill_request_checkpoint = active_request
            .prefill_request_checkpoint()
            .map_err(qwen3_5_runtime_error)?;

        // Phase 3: forward dispatch (speculative / visual / history / text).
        let forward_started_at = std::time::Instant::now();
        let dispatch_outcome = match self.dispatch_prefill_forward(
            request_id,
            active_request,
            prefill_start,
            prefill_end,
            &plan,
        ) {
            Ok(dispatch_outcome) => dispatch_outcome,
            Err(PrefillForwardError::Execution(execution_error)) => {
                return Err(if plan.speculative_prefill_target_is_active {
                    configured_speculative_prefill_execution_error(
                        request_id,
                        "sparse target execution",
                        execution_error,
                        prefill_request_checkpoint,
                    )
                } else {
                    prefill_execution_error(execution_error, prefill_request_checkpoint)
                });
            }
            Err(PrefillForwardError::Engine(inference_engine_error)) => {
                return Err(PromptPrefillChunkAttemptError::Engine(
                    inference_engine_error,
                ));
            }
        };
        let forward_elapsed = forward_started_at.elapsed();
        if forward_elapsed > std::time::Duration::from_secs(5) {
            tracing::info!(
                prefill_start,
                prefill_end,
                prefill_token_count = plan.prefill_token_count,
                forward_elapsed_millis = forward_elapsed.as_millis(),
                "slow prefill chunk forward"
            );
        }

        let model = self
            .model
            .as_ref()
            .ok_or_else(|| fatal_engine_error("Qwen3.5 engine lost its loaded model"))?;
        let mut boundary_checkpoints = dispatch_outcome.boundary_checkpoints;
        super::prompt_prefill_counters::record_sparse_target_and_mode_counters(
            active_request,
            model,
            plan.speculative_prefill_target_is_active,
            plan.speculative_prefill_target_token_count,
            plan.speculative_prefill_chunk_mode,
            plan.prefill_token_count,
            &plan.all_completed_prefill_chunk_tokens,
            dispatch_outcome.terminal_history_token_count,
            &mut boundary_checkpoints,
        )?;

        // Test-only: force a capacity rejection after the forward succeeds.
        if std::mem::take(&mut active_request.force_next_prefill_capacity_rejection_for_tests) {
            return Err(PromptPrefillChunkAttemptError::ActiveMemoryLimitExceeded {
                active_memory_bytes: 1,
                attempted_allocation_bytes: 1,
                allowed_active_memory_bytes: 1,
                prefill_request_checkpoint,
            });
        }

        Ok(PromptPrefillChunkOutcome {
            active_memory_bytes_before_growth: admission_outcome.active_memory_bytes_before_growth,
            retained_expert_payload_bytes_before_growth: admission_outcome
                .retained_expert_payload_bytes_before_growth,
            forward_chunk_elapsed_millis: forward_elapsed.as_millis() as u64,
            adaptive_ram_growth_context: plan
                .adaptive_ram_growth_context
                .with_sparse_experts_are_paged(model.sparse_experts_are_paged()),
            exact_temporary_workspace_bytes: plan.exact_temporary_workspace_bytes,
            boundary_checkpoints,
        })
    }
}
