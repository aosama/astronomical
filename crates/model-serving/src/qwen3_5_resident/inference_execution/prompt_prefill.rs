//! Thin orchestrator for a single prompt-processing chunk.
//!
//! Calls the three phases in order:
//! 1. `plan_prompt_prefill_chunk` — pure decision logic (no side effects).
//! 2. `execute_prefill_memory_admission` — adaptive RAM growth.
//! 3. `dispatch_prefill_forward` — GPU forward dispatch for visual, history,
//!    or plain-text input.
//!
//! Collects results into `PromptPrefillChunkOutcome` for the caller in
//! `prefill_advance.rs`.

use super::Qwen3_5EngineState;
use super::engine_request::Qwen3_5EngineRequest;
use super::prefill_forward_dispatch::PrefillForwardError;
use super::prompt_prefill_errors;
use super::prompt_prefill_errors::PromptPrefillChunkAttemptError;

use crate::{
    AdaptiveRamGrowthExecutionProfile, PerformanceCounter,
    Qwen3_5PersistentPromptCacheBoundaryCheckpoint,
};

/// Outcome of executing one prompt-processing chunk.
pub(super) struct PromptPrefillChunkOutcome {
    pub(super) active_memory_bytes_before_growth: usize,
    pub(super) retained_expert_payload_bytes_before_growth: u64,
    pub(super) streamed_expert_page_bytes_before_growth: u64,
    pub(super) forward_chunk_elapsed_millis: u64,
    pub(super) adaptive_ram_growth_context: crate::AdaptiveRamGrowthContext,
    pub(super) exact_temporary_workspace_bytes: usize,
    pub(super) boundary_checkpoints: Vec<Qwen3_5PersistentPromptCacheBoundaryCheckpoint>,
}

impl Qwen3_5EngineState {
    /// Execute one prompt-processing chunk: plan → admit → forward.
    pub(super) fn execute_prompt_prefill_chunk(
        &mut self,
        active_request: &mut Qwen3_5EngineRequest,
        prefill_start: usize,
        prefill_end: usize,
    ) -> Result<PromptPrefillChunkOutcome, PromptPrefillChunkAttemptError> {
        // Phase 1: pure decision logic — no side effects.
        let plan = self.plan_prompt_prefill_chunk(active_request, prefill_start, prefill_end)?;

        // Phase 2: adaptive RAM growth admission. A rejection leaves no state
        // behind, so the caller may retry a smaller chunk of the same prompt.
        let admission_outcome = self.execute_prefill_memory_admission(
            active_request,
            plan.adaptive_ram_growth_context,
            0,
            plan.exact_temporary_workspace_bytes,
            plan.direct_publication_workspace_bytes,
        )?;
        // Counted after admission so a rejected attempt that is retried as a
        // smaller chunk counts once, as the chunk that actually executes.
        active_request
            .performance_attribution
            .record_counter(PerformanceCounter::PrefillChunkCount, 1);

        // Phase 3: dispatch the target model's prompt forward.
        let forward_started_at = std::time::Instant::now();
        let dispatch_outcome = match self.dispatch_prefill_forward(
            active_request,
            prefill_start,
            prefill_end,
            &plan,
        ) {
            Ok(dispatch_outcome) => dispatch_outcome,
            Err(PrefillForwardError::Execution(execution_error)) => {
                return Err(prompt_prefill_errors::prefill_execution_error(
                    execution_error,
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

        let mut boundary_checkpoints = dispatch_outcome.boundary_checkpoints;
        super::prompt_prefill_counters::record_persistent_prompt_cache_boundary_checkpoint(
            active_request,
            plan.prefill_token_count,
            &plan.all_completed_prefill_chunk_tokens,
            &mut boundary_checkpoints,
        )?;
        let _model = self
            .model
            .as_ref()
            .ok_or_else(|| super::fatal_engine_error("Qwen3.5 engine lost its loaded model"))?;

        Ok(PromptPrefillChunkOutcome {
            active_memory_bytes_before_growth: admission_outcome.active_memory_bytes_before_growth,
            retained_expert_payload_bytes_before_growth: admission_outcome
                .retained_expert_payload_bytes_before_growth,
            streamed_expert_page_bytes_before_growth: admission_outcome
                .streamed_expert_page_bytes_before_growth,
            forward_chunk_elapsed_millis: forward_elapsed.as_millis() as u64,
            adaptive_ram_growth_context: plan
                .adaptive_ram_growth_context
                .with_execution_profile(AdaptiveRamGrowthExecutionProfile::Resident),
            exact_temporary_workspace_bytes: plan.exact_temporary_workspace_bytes,
            boundary_checkpoints,
        })
    }
}
