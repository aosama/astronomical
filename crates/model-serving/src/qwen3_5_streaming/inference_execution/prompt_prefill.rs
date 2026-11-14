//! Thin orchestrator for a single prompt-processing chunk.
//!
//! Calls the three phases in order:
//! 1. `plan_prompt_prefill_chunk` — pure decision logic (no side effects).
//! 2. `begin_forward_memory_measurement` — what is held before the forward.
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

use crate::{PerformanceCounter, Qwen3_5PersistentPromptCacheBoundaryCheckpoint};

/// Outcome of executing one prompt-processing chunk.
pub(super) struct PromptPrefillChunkOutcome {
    /// What MLX held when the forward began.
    pub(super) active_memory_bytes_before_forward: usize,
    pub(super) forward_chunk_elapsed_millis: u64,
    /// The tokens the forward's memory is measured under: the chunk's, or
    /// one span's for a group read one layer at a time.
    pub(super) measured_forward_token_count: usize,
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
        let plan =
            self.plan_prompt_prefill_chunk(active_request, prefill_start, prefill_end, None)?;
        // Counted before the forward, so a declined layer-major group counts one
        // attempt; its chunks then count again as they run.
        active_request
            .performance_attribution
            .record_counter(PerformanceCounter::PrefillChunkCount, 1);

        // Phase 2: what is held before the forward, for measuring it.
        let memory_baseline = self.begin_forward_memory_measurement()?;

        // Phase 3: dispatch the target model's prompt forward.
        let forward_started_at = std::time::Instant::now();
        let dispatch_outcome_result =
            self.dispatch_prefill_forward(active_request, prefill_start, prefill_end, &plan);
        let dispatch_outcome = match dispatch_outcome_result {
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
        crate::qwen3_5_core::prompt_cache_capture::record_persistent_prompt_cache_boundary_checkpoint(
            active_request,
            plan.prefill_token_count,
            &plan.all_completed_prefill_chunk_tokens,
            &mut boundary_checkpoints,
        )?;

        Ok(PromptPrefillChunkOutcome {
            active_memory_bytes_before_forward: memory_baseline.active_memory_bytes,
            forward_chunk_elapsed_millis: forward_elapsed.as_millis() as u64,
            measured_forward_token_count: plan.measured_forward_token_count,
            boundary_checkpoints,
        })
    }
}
