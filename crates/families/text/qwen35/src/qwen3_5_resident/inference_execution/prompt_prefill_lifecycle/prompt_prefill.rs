//! Thin orchestrator for a single prompt-processing chunk.
//!
//! Calls the three phases in order:
//! 1. `plan_prompt_prefill_chunk` — pure decision logic (no side effects).
//! 2. `admit_forward` — the memory governor says whether the chunk fits.
//! 3. `dispatch_prefill_forward` — GPU forward dispatch for visual, history,
//!    or plain-text input.
//!
//! Collects results into `PromptPrefillChunkOutcome` for the caller in
//! `prefill_advance.rs`.

use super::Qwen3_5EngineState;
use super::engine_request::ResidentEngineRequest;
use super::prefill_forward_dispatch::PrefillForwardError;
use super::prompt_prefill_errors;
use super::prompt_prefill_errors::PromptPrefillChunkAttemptError;

use super::forward_memory::{AdmittedForward, ForwardShape};
use crate::{AstroSsdPromptCacheBoundaryCheckpoint, PerformanceCounter};

/// Outcome of executing one prompt-processing chunk.
pub(super) struct PromptPrefillChunkOutcome {
    /// The chunk as the memory governor admitted it, to be reported once
    /// the chunk has been measured.
    pub(super) admitted_forward: AdmittedForward,
    pub(super) forward_chunk_elapsed_millis: u64,
    pub(super) boundary_checkpoints: Vec<AstroSsdPromptCacheBoundaryCheckpoint>,
}

impl Qwen3_5EngineState {
    /// Execute one prompt-processing chunk: plan → admit → forward.
    pub(super) fn execute_prompt_prefill_chunk(
        &mut self,
        active_request: &mut ResidentEngineRequest,
        prefill_start: usize,
        prefill_end: usize,
    ) -> Result<PromptPrefillChunkOutcome, PromptPrefillChunkAttemptError> {
        // Phase 1: pure decision logic — no side effects.
        let plan = self.plan_prompt_prefill_chunk(active_request, prefill_start, prefill_end)?;

        // Phase 2: ask whether the chunk fits. A rejection leaves no state
        // behind, so the caller can move the request to the streaming engine.
        let admitted_forward = self.admit_forward(
            ForwardShape::prompt_chunk(plan.prefill_token_count),
            &active_request.request_decoder_state,
            0,
            plan.exact_temporary_workspace_bytes,
        )?;
        // Counted after admission so only a chunk that runs is counted.
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
        text_generation::astro_ssd_prompt_cache_capture::record_astro_ssd_prompt_cache_boundary_checkpoint(
            active_request,
            plan.prefill_token_count,
            &plan.all_completed_prefill_chunk_tokens,
            &mut boundary_checkpoints,
        )?;
        Ok(PromptPrefillChunkOutcome {
            admitted_forward,
            forward_chunk_elapsed_millis: forward_elapsed.as_millis() as u64,
            boundary_checkpoints,
        })
    }
}
