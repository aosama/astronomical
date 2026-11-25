//! Redoes one prompt-processing forward that ran out of memory instead of
//! failing the request.
//!
//! The memory governor leases room for another attempt and the astro expert cache
//! gives back just the experts that room displaces (issue #2221). The
//! forward that failed is then run again unchanged. A forward that fits is
//! measured, and that measurement replaces the guess that was too small.
//!
//! The decoder state is rebuilt, not rewound. A failed command buffer can
//! leave some layers' K/V and recurrent state updated and others not, and
//! checkpointing live state before every forward would keep old K/V handles
//! alive, defeating MLX buffer donation and copying the whole `astro_decoder_state_cache` per
//! forward. Every completed forward already published its durable
//! astro SSD prompt cache blocks before the cursor advanced, so re-seeding from the
//! `astro_ssd_prompt_cache` reproduces exactly the verified frontier.
//!
//! The retry never shrinks the forward: smaller chunks re-pay
//! complete-layer storage reads without changing the outcome.

use ipc_protocol::RequestId;
use text_generation::generation_loop::astro_ssd_prompt_cache_restore;

use crate::{GeneratedToken, InferenceEngineError, MemoryPhase};
use text_generation::RequestDecoderStateStack;
use text_generation::engine_errors::invalid_request_error;

use super::engine_request::StreamingEngineRequest;
use super::prompt_prefill_errors::PromptPrefillChunkAttemptError;
use super::{Qwen3_5EngineState, fatal_runtime_error};

/// Whether a prefill memory failure may be redone at all.
///
/// Image requests stay terminal: their uploaded visual embeddings cover only
/// the prompt suffix planned at request start, so re-seeding to a different
/// cursor would misalign the consumed embedding rows.
pub(super) const fn can_redo_prefill_after_memory_failure(has_visual_inputs: bool) -> bool {
    !has_visual_inputs
}

impl Qwen3_5EngineState {
    /// Leases room for another attempt and leaves the request ready to run
    /// the failed forward again. `Ok(false)` once the governor's attempts
    /// are spent: the limit cannot hold the forward.
    pub(super) fn redo_prefill_after_memory_exhaustion(
        &mut self,
        request_id: RequestId,
        active_request: &mut StreamingEngineRequest,
        failed_prefill_start: usize,
        known_shortfall_bytes: u64,
    ) -> Result<bool, InferenceEngineError> {
        let model = self.core.loaded_model()?;
        if !model.govern_redo_after_memory_failure(MemoryPhase::Prefill, known_shortfall_bytes) {
            return Ok(false);
        }
        self.reseed_prefill_from_astro_ssd_prompt_cache(
            request_id,
            active_request,
            failed_prefill_start,
        )?;
        Ok(true)
    }

    /// Drops the possibly half-updated decoder state and rebuilds it from
    /// the `astro_ssd_prompt_cache`, moving the cursor back to the last
    /// published boundary.
    pub(super) fn reseed_prefill_from_astro_ssd_prompt_cache(
        &mut self,
        request_id: RequestId,
        active_request: &mut StreamingEngineRequest,
        failed_prefill_start: usize,
    ) -> Result<(), InferenceEngineError> {
        let model = self.core.loaded_model()?;
        let astro_decoder_state_cache_layout = model.astro_decoder_state_cache_layout().clone();
        // Drop the possibly half-updated decoder state first so its buffers are
        // unreferenced before the MLX buffer cache is cleared below.
        active_request.request_decoder_state =
            RequestDecoderStateStack::empty_from_astro_decoder_state_cache_layout_with_full_attention_kv_state_growth_tokens(
                &astro_decoder_state_cache_layout,
                self.core.full_attention_kv_state_growth_tokens,
            )
            .map_err(fatal_runtime_error)?;
        active_request.pending_generated_token = None;
        model.forget_expert_layer_states();
        model
            .runtime()
            .synchronize_gpu_stream_and_clear_mlx_buffer_cache()
            .map_err(fatal_runtime_error)?;
        let mut restored_prefill_cursor = 0_usize;
        let mut restored_next_position_tokens = 0_u32;
        let mut restored_last_astro_ssd_prompt_cache_block_key = None;
        if self.core.astro_ssd_prompt_cache.is_some()
            && active_request.can_use_astro_ssd_prompt_cache
        {
            // Same split borrow as request start: the store must be reinstalled
            // even when restore fails, or later requests silently lose the `astro_ssd_prompt_cache`.
            let astro_ssd_prompt_cache = self.core.astro_ssd_prompt_cache.take();
            let restore_result =
                if let Some(astro_ssd_prompt_cache) = astro_ssd_prompt_cache.as_ref() {
                    astro_ssd_prompt_cache_restore::restore_astro_ssd_prompt_cache_prefix(
                        self,
                        &astro_ssd_prompt_cache_restore::AstroSsdPromptCacheRestoreRequest {
                            request_id,
                            astro_ssd_prompt_cache,
                            prompt_token_ids: &active_request.input_token_ids,
                            block_causal_inputs: &active_request
                                .astro_ssd_prompt_cache_block_causal_inputs,
                            total_context_tokens: active_request
                                .engine_extension
                                .reserved_context_tokens,
                        },
                        &mut active_request.request_decoder_state,
                        &mut active_request.performance_attribution,
                    )
                    .map(Some)
                } else {
                    Ok(None)
                };
            self.core.astro_ssd_prompt_cache = astro_ssd_prompt_cache;
            if let Some(restore_outcome) = restore_result? {
                restored_prefill_cursor = restore_outcome.restored_token_count;
                restored_next_position_tokens =
                    restore_outcome.astro_ssd_prompt_cache_restored_token_count;
                restored_last_astro_ssd_prompt_cache_block_key =
                    restore_outcome.last_restored_astro_ssd_prompt_cache_block_key;
                // The re-restore replaced the astro SSD prompt cache frontier, so the
                // request's reported diagnostics and reuse counts must describe
                // this restore, not the one from request start.
                active_request.astro_ssd_prompt_cache_diagnostics =
                    Some(restore_outcome.astro_ssd_prompt_cache_diagnostics);
                active_request.prompt_work_reuse.target_restored_token_count =
                    u64::from(restore_outcome.astro_ssd_prompt_cache_restored_token_count);
            }
        }
        active_request.prefill_cursor = restored_prefill_cursor;
        active_request.next_position_tokens = restored_next_position_tokens;
        active_request.last_restored_astro_ssd_prompt_cache_block_key =
            restored_last_astro_ssd_prompt_cache_block_key;
        // Demand evidence recorded by the discarded forward must not steer
        // the reply's first tokens, so the retry opens a fresh window.
        active_request
            .engine_extension
            .has_opened_prefill_demand_window = false;
        tracing::warn!(
            request_id = request_id.value(),
            failed_prefill_start,
            resumed_prefill_cursor = restored_prefill_cursor,
            "re-seeded a prompt from the astro-ssd-prompt-cache to redo a forward that ran out of memory"
        );
        Ok(())
    }
}

/// Where a failed prefill attempt stood.
#[derive(Clone, Copy)]
pub(super) struct PrefillAttemptFailure {
    pub(super) prefill_start: usize,
    pub(super) attempted_prefill_chunk_token_count: usize,
}

/// The request-level error for a failed attempt, logged with where it stood.
fn prefill_memory_failure_error(
    request_id: RequestId,
    prefill_attempt_failure: PrefillAttemptFailure,
    prompt_prefill_chunk_attempt_error: PromptPrefillChunkAttemptError,
) -> InferenceEngineError {
    let prefill_start = prefill_attempt_failure.prefill_start;
    let attempted_prefill_chunk_token_count =
        prefill_attempt_failure.attempted_prefill_chunk_token_count;
    match prompt_prefill_chunk_attempt_error {
        PromptPrefillChunkAttemptError::Engine(generation_error) => generation_error,
        PromptPrefillChunkAttemptError::ActiveMemoryLimitExceeded {
            active_memory_bytes,
            attempted_allocation_bytes,
            allowed_active_memory_bytes,
        } => {
            tracing::warn!(
                request_id = request_id.value(),
                prefill_start,
                attempted_prefill_chunk_token_count,
                active_memory_bytes,
                attempted_allocation_bytes,
                allowed_active_memory_bytes,
                "prefill forward exceeded the MLX active-memory limit"
            );
            invalid_request_error(format!(
                "prompt forward of {attempted_prefill_chunk_token_count} tokens exceeded the MLX active-memory limit: {active_memory_bytes} active bytes, {attempted_allocation_bytes} requested bytes, {allowed_active_memory_bytes} allowed bytes"
            ))
        }
        PromptPrefillChunkAttemptError::GraphicsProcessorMemoryExhausted { reason } => {
            tracing::warn!(
                request_id = request_id.value(),
                prefill_start,
                attempted_prefill_chunk_token_count,
                reason = %reason,
                "prefill forward exhausted graphics-processor memory"
            );
            invalid_request_error(format!(
                "prompt forward of {attempted_prefill_chunk_token_count} tokens exhausted GPU memory: {reason}"
            ))
        }
    }
}

impl Qwen3_5EngineState {
    /// Routes a failed attempt. An engine failure ends the request. A
    /// prompt still in flight runs its failed span again with more room.
    /// A prompt that could not be put back to a span, as a dense model's
    /// cannot, is read again from the `astro_ssd_prompt_cache` with more room. Every
    /// redo stops once the memory governor's attempts are spent.
    pub(super) fn recover_failed_prefill_attempt(
        &mut self,
        request_id: RequestId,
        active_request: &mut super::engine_request::StreamingEngineRequest,
        prefill_attempt_failure: PrefillAttemptFailure,
        prompt_prefill_chunk_attempt_error: PromptPrefillChunkAttemptError,
    ) -> Result<Option<GeneratedToken>, InferenceEngineError> {
        let was_recovered = ran_out_of_memory(&prompt_prefill_chunk_attempt_error)
            && self.make_room_to_run_failed_attempt_again(
                request_id,
                active_request,
                prefill_attempt_failure,
                known_memory_shortfall_bytes(&prompt_prefill_chunk_attempt_error),
            )?;
        if !was_recovered {
            return Err(prefill_memory_failure_error(
                request_id,
                prefill_attempt_failure,
                prompt_prefill_chunk_attempt_error,
            ));
        }
        // Recursion depth is bounded by the governor's redo attempts.
        self.advance_prompt_prefill_if_pending(request_id, active_request)
    }

    /// Leaves the request ready to run what failed again: the failed span
    /// of a prompt still in flight, the dense fallback to chunks, or a
    /// governed redo from the `astro_ssd_prompt_cache`. `Ok(false)` when none applies:
    /// an image request with no prompt in flight, or the redo attempts are
    /// spent.
    fn make_room_to_run_failed_attempt_again(
        &mut self,
        request_id: RequestId,
        active_request: &mut super::engine_request::StreamingEngineRequest,
        prefill_attempt_failure: PrefillAttemptFailure,
        known_shortfall_bytes: u64,
    ) -> Result<bool, InferenceEngineError> {
        if active_request
            .engine_extension
            .layer_at_a_time_prefill
            .is_some()
        {
            return self.make_room_to_run_failed_span_again(
                request_id,
                active_request,
                known_shortfall_bytes,
            );
        }
        if !can_redo_prefill_after_memory_failure(active_request.has_visual_inputs) {
            return Ok(false);
        }
        self.redo_prefill_after_memory_exhaustion(
            request_id,
            active_request,
            prefill_attempt_failure.prefill_start,
            known_shortfall_bytes,
        )
    }
}

impl Qwen3_5EngineState {
    /// Leases room for the failed span of the prompt in flight and returns
    /// the allocator's freed buffers to the system, so the next step runs
    /// that span again. Nothing the prompt already read is read again but
    /// the failed layer's page. `Ok(false)` once the governor's attempts
    /// are spent, and the prompt in flight is dropped.
    fn make_room_to_run_failed_span_again(
        &mut self,
        request_id: RequestId,
        active_request: &mut StreamingEngineRequest,
        known_shortfall_bytes: u64,
    ) -> Result<bool, InferenceEngineError> {
        let model = self.core.loaded_model()?;
        if !model.govern_redo_after_memory_failure(MemoryPhase::Prefill, known_shortfall_bytes) {
            active_request.engine_extension.layer_at_a_time_prefill = None;
            return Ok(false);
        }
        model
            .runtime()
            .synchronize_gpu_stream_and_clear_mlx_buffer_cache()
            .map_err(fatal_runtime_error)?;
        tracing::warn!(
            request_id = request_id.value(),
            "a prompt span ran out of memory; running that span again with more room"
        );
        Ok(true)
    }
}

/// How far over the limit a failed attempt went, when the failure says:
/// what was active plus what it asked for, less what is allowed. Zero when
/// the failure carries no figures, as a GPU command buffer's does not.
pub(super) const fn known_memory_shortfall_bytes(
    prompt_prefill_chunk_attempt_error: &PromptPrefillChunkAttemptError,
) -> u64 {
    // Only an MLX limit refusal carries figures; the other failures, and the
    // admission refusal this engine never makes, have none.
    if let PromptPrefillChunkAttemptError::ActiveMemoryLimitExceeded {
        active_memory_bytes,
        attempted_allocation_bytes,
        allowed_active_memory_bytes,
    } = prompt_prefill_chunk_attempt_error
    {
        (*active_memory_bytes as u64)
            .saturating_add(*attempted_allocation_bytes as u64)
            .saturating_sub(*allowed_active_memory_bytes as u64)
    } else {
        0
    }
}

/// Whether a failed attempt ran out of memory, as opposed to an engine
/// failure that ends the request whatever path it took.
pub(super) const fn ran_out_of_memory(
    prompt_prefill_chunk_attempt_error: &PromptPrefillChunkAttemptError,
) -> bool {
    matches!(
        prompt_prefill_chunk_attempt_error,
        PromptPrefillChunkAttemptError::ActiveMemoryLimitExceeded { .. }
            | PromptPrefillChunkAttemptError::GraphicsProcessorMemoryExhausted { .. }
    )
}

#[cfg(test)]
#[path = "prefill_memory_recovery_tests.rs"]
mod tests;
