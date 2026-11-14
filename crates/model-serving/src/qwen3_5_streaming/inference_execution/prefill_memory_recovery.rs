//! Redoes one prompt-processing forward that ran out of memory instead of
//! failing the request.
//!
//! The memory governor leases room for another attempt and the expert cache
//! gives back just the experts that room displaces (issue #2221). The
//! forward that failed is then run again unchanged. A forward that fits is
//! measured, and that measurement replaces the guess that was too small.
//!
//! The decoder state is rebuilt, not rewound. A failed command buffer can
//! leave some layers' K/V and recurrent state updated and others not, and
//! checkpointing live state before every forward would keep old K/V handles
//! alive, defeating MLX buffer donation and copying the whole cache per
//! forward. Every completed forward already published its durable
//! prompt-cache blocks before the cursor advanced, so re-seeding from the
//! persistent prompt cache reproduces exactly the verified frontier.
//!
//! The retry never shrinks the forward: smaller chunks re-pay
//! complete-layer storage reads without changing the outcome.

use astronomical_ipc_protocol::RequestId;

use crate::qwen3_5_core::decoder::RequestDecoderStateStack;
use crate::qwen3_5_streaming::model::memory_admission;
use crate::{GeneratedToken, InferenceEngineError, MemoryPhase};

use super::engine_request::Qwen3_5EngineRequest;
use super::prompt_prefill_errors::PromptPrefillChunkAttemptError;
use super::{Qwen3_5EngineState, fatal_engine_error, qwen3_5_runtime_error};

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
        active_request: &mut Qwen3_5EngineRequest,
        failed_prefill_start: usize,
        known_shortfall_bytes: u64,
    ) -> Result<bool, InferenceEngineError> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| fatal_engine_error("Qwen3.5 engine lost its loaded model"))?;
        if !model.govern_redo_after_memory_failure(MemoryPhase::Prefill, known_shortfall_bytes) {
            return Ok(false);
        }
        self.reseed_prefill_from_prompt_cache(request_id, active_request, failed_prefill_start)?;
        Ok(true)
    }

    /// Drops the possibly half-updated decoder state and rebuilds it from
    /// the persistent prompt cache, moving the cursor back to the last
    /// published boundary.
    pub(super) fn reseed_prefill_from_prompt_cache(
        &mut self,
        request_id: RequestId,
        active_request: &mut Qwen3_5EngineRequest,
        failed_prefill_start: usize,
    ) -> Result<(), InferenceEngineError> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| fatal_engine_error("Qwen3.5 engine lost its loaded model"))?;
        let decoder_cache_layout = model.decoder_cache_layout().clone();
        // Drop the possibly half-updated decoder state first so its buffers are
        // unreferenced before the allocator cache is cleared below.
        active_request.request_decoder_state =
            RequestDecoderStateStack::empty_from_decoder_cache_layout_with_full_attention_kv_state_growth_tokens(
                &decoder_cache_layout,
                self.full_attention_kv_state_growth_tokens,
            )
            .map_err(qwen3_5_runtime_error)?;
        active_request.pending_generated_token = None;
        model.clear_phase_aware_expert_residency_plan();
        model
            .runtime()
            .synchronize_gpu_stream_and_clear_allocator_cache()
            .map_err(qwen3_5_runtime_error)?;
        let mut restored_prefill_cursor = 0_usize;
        let mut restored_next_position_tokens = 0_u32;
        let mut restored_last_persistent_prompt_cache_block_key = None;
        if self.persistent_prompt_cache.is_some() && active_request.can_use_persistent_prompt_cache
        {
            // Same split borrow as request start: the store must be reinstalled
            // even when restore fails, or later requests silently lose caching.
            let persistent_prompt_cache = self.persistent_prompt_cache.take();
            let restore_result =
                if let Some(persistent_prompt_cache) = persistent_prompt_cache.as_ref() {
                    self.restore_persistent_prompt_cache_prefix(
                        request_id,
                        persistent_prompt_cache,
                        &active_request.input_token_ids,
                        &active_request.persistent_prompt_cache_block_causal_inputs,
                        active_request.engine_extension.reserved_context_tokens,
                        &mut active_request.request_decoder_state,
                        &mut active_request.performance_attribution,
                    )
                    .map(Some)
                } else {
                    Ok(None)
                };
            self.persistent_prompt_cache = persistent_prompt_cache;
            if let Some(restore_outcome) = restore_result? {
                restored_prefill_cursor = restore_outcome.restored_token_count;
                restored_next_position_tokens = restore_outcome.persistent_prompt_cache_token_count;
                restored_last_persistent_prompt_cache_block_key =
                    restore_outcome.last_restored_persistent_prompt_cache_block_key;
                // The re-restore replaced the prompt-cache frontier, so the
                // request's reported diagnostics and reuse counts must describe
                // this restore, not the one from request start.
                active_request.persistent_prompt_cache_diagnostics =
                    Some(restore_outcome.persistent_prompt_cache_diagnostics);
                active_request.prompt_work_reuse.target_restored_token_count =
                    u64::from(restore_outcome.persistent_prompt_cache_token_count);
            }
        }
        active_request.prefill_cursor = restored_prefill_cursor;
        active_request.next_position_tokens = restored_next_position_tokens;
        active_request.last_restored_persistent_prompt_cache_block_key =
            restored_last_persistent_prompt_cache_block_key;
        // Demand evidence recorded by the discarded forward must not steer
        // the reply's first tokens, so the retry opens a fresh window.
        active_request
            .engine_extension
            .has_opened_prefill_demand_window = false;
        tracing::warn!(
            request_id = request_id.value(),
            failed_prefill_start,
            resumed_prefill_cursor = restored_prefill_cursor,
            "re-seeded a prompt from the persistent prompt cache to redo a forward that ran out of memory"
        );
        Ok(())
    }
}

/// Where a failed prefill attempt stood.
#[derive(Clone, Copy)]
pub(super) struct PrefillAttemptFailure {
    pub(super) prefill_start: usize,
    pub(super) attempted_prefill_chunk_token_count: usize,
    pub(super) is_layer_major_attempt: bool,
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
            memory_admission::invalid_request_error(format!(
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
            memory_admission::invalid_request_error(format!(
                "prompt forward of {attempted_prefill_chunk_token_count} tokens exhausted GPU memory: {reason}"
            ))
        }
    }
}

impl Qwen3_5EngineState {
    /// Routes a failed attempt. An engine failure ends the request. A
    /// layer-at-a-time prompt still in flight runs its failed span again
    /// with more room. A dense model's layer-major group that ran out of
    /// memory falls back to chunks. Anything else that ran out of memory is
    /// redone from the prompt cache with more room. Every redo stops once
    /// the memory governor's attempts are spent.
    pub(super) fn recover_failed_prefill_attempt(
        &mut self,
        request_id: RequestId,
        active_request: &mut super::engine_request::Qwen3_5EngineRequest,
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
    /// governed redo from the prompt cache. `Ok(false)` when none applies:
    /// an image request with no prompt in flight, or the redo attempts are
    /// spent.
    fn make_room_to_run_failed_attempt_again(
        &mut self,
        request_id: RequestId,
        active_request: &mut super::engine_request::Qwen3_5EngineRequest,
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
        let serves_dense_model = self
            .model
            .as_ref()
            .is_some_and(|model| model.streams_dense_layers());
        if prefill_attempt_failure.is_layer_major_attempt && serves_dense_model {
            self.decline_dense_layer_major_prefill_after_failure(
                request_id,
                active_request,
                prefill_attempt_failure.prefill_start,
            )?;
            return Ok(true);
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
        active_request: &mut Qwen3_5EngineRequest,
        known_shortfall_bytes: u64,
    ) -> Result<bool, InferenceEngineError> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| fatal_engine_error("Qwen3.5 engine lost its loaded model"))?;
        if !model.govern_redo_after_memory_failure(MemoryPhase::Prefill, known_shortfall_bytes) {
            active_request.engine_extension.layer_at_a_time_prefill = None;
            return Ok(false);
        }
        model
            .runtime()
            .synchronize_gpu_stream_and_clear_allocator_cache()
            .map_err(qwen3_5_runtime_error)?;
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
    match prompt_prefill_chunk_attempt_error {
        PromptPrefillChunkAttemptError::ActiveMemoryLimitExceeded {
            active_memory_bytes,
            attempted_allocation_bytes,
            allowed_active_memory_bytes,
        } => (*active_memory_bytes as u64)
            .saturating_add(*attempted_allocation_bytes as u64)
            .saturating_sub(*allowed_active_memory_bytes as u64),
        PromptPrefillChunkAttemptError::Engine(_)
        | PromptPrefillChunkAttemptError::GraphicsProcessorMemoryExhausted { .. } => 0,
    }
}

/// Whether a failed attempt ran out of memory, as opposed to an engine
/// failure that ends the request whatever path it took.
pub(super) const fn ran_out_of_memory(
    prompt_prefill_chunk_attempt_error: &PromptPrefillChunkAttemptError,
) -> bool {
    match prompt_prefill_chunk_attempt_error {
        PromptPrefillChunkAttemptError::Engine(_) => false,
        PromptPrefillChunkAttemptError::ActiveMemoryLimitExceeded { .. }
        | PromptPrefillChunkAttemptError::GraphicsProcessorMemoryExhausted { .. } => true,
    }
}

impl Qwen3_5EngineState {
    /// Sends a dense model's prompt chunk by chunk after its layer-major
    /// prefill ran out of memory (issue #1985).
    ///
    /// A dense model's kept layers are not on the memory governor yet, so a
    /// redo would free nothing. Holding every span's hidden states and
    /// checkpoints at once can need more than one chunk did, so the failure
    /// means the group did not fit, not that the prompt cannot run: the
    /// state is rebuilt from the prompt cache and the planned layers are
    /// kept again. If that cannot fit beside this request's state, the
    /// layers stream until the request ends, which is still correct.
    pub(super) fn decline_dense_layer_major_prefill_after_failure(
        &mut self,
        request_id: RequestId,
        active_request: &mut Qwen3_5EngineRequest,
        prefill_start: usize,
    ) -> Result<(), InferenceEngineError> {
        tracing::warn!(
            request_id = request_id.value(),
            prefill_start,
            "layer-major prefill did not fit; continuing chunk by chunk"
        );
        active_request.engine_extension.layer_major_prefill_declined = true;
        self.reseed_prefill_from_prompt_cache(request_id, active_request, prefill_start)?;
        self.keep_planned_dense_layers_after_decline(request_id);
        Ok(())
    }

    fn keep_planned_dense_layers_after_decline(&self, request_id: RequestId) {
        let Some(model) = self.model.as_ref() else {
            return;
        };
        if let Err(replan_error) = model.replan_kept_dense_layers_under_installed_limit() {
            tracing::warn!(
                request_id = request_id.value(),
                error = %replan_error,
                "could not keep the planned dense layers beside this request; they stream until it ends"
            );
        }
    }
}

#[cfg(test)]
#[path = "prefill_memory_recovery_tests.rs"]
mod tests;
