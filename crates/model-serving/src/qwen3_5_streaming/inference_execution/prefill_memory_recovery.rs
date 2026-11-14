//! Recovers one prompt-processing chunk from memory exhaustion instead of
//! failing the request.
//!
//! A memory failure during a prefill chunk almost always means retained expert
//! residency, not the chunk itself, consumed the headroom: a mid-flight ceiling
//! change or a cancelled request can leave more seated experts than the
//! current budget allows, and the planner deliberately never shrinks existing
//! occupancy on its own. Dying mid-request is the worst user outcome, so the
//! engine sheds that residency once and resumes the same prompt.
//!
//! Recovery deliberately rebuilds the decoder state instead of rewinding it.
//! A failed command buffer can leave some layers' K/V and recurrent state
//! updated and others not, and checkpointing live state before every chunk
//! would keep old K/V handles alive, defeating MLX buffer donation and copying
//! the whole cache per chunk. Every completed chunk already published its
//! durable prompt-cache blocks before the cursor advanced, so re-seeding from
//! the persistent prompt cache reproduces exactly the verified frontier.
//!
//! The retry does not halve the chunk: the retired ladder showed that smaller
//! chunks re-pay complete-layer storage reads without changing the outcome
//! when residency is the real consumer.

use astronomical_ipc_protocol::RequestId;

use crate::InferenceEngineError;
use crate::qwen3_5_core::decoder::RequestDecoderStateStack;
use crate::qwen3_5_streaming::model::reclaim_retained_payload_for_request_memory_pressure;

use super::engine_request::Qwen3_5EngineRequest;
use super::prompt_prefill_errors::PromptPrefillChunkAttemptError;
use super::{Qwen3_5EngineState, fatal_engine_error, qwen3_5_runtime_error};

/// One recovery per request while retention is frozen: after it, retained
/// experts stay at zero, so a second failure means the prompt's own workspace
/// does not fit and retrying again could only loop.
pub(super) const MAXIMUM_PREFILL_MEMORY_RECOVERY_ATTEMPTS_PER_REQUEST: u8 = 1;

/// Decides whether a prefill memory failure may be recovered.
///
/// The allowed count comes from the request's refill state
/// (`PostRecoveryExpertRefill::allowed_recovery_attempt_count`): one
/// recovery, and one more while a refill is active.
///
/// Image requests stay terminal: their uploaded visual embeddings cover only
/// the prompt suffix planned at request start, so re-seeding to a different
/// cursor would misalign the consumed embedding rows.
pub(super) fn should_attempt_prefill_memory_recovery(
    completed_recovery_attempt_count: u8,
    has_visual_inputs: bool,
    allowed_recovery_attempt_count: u8,
) -> bool {
    !has_visual_inputs && completed_recovery_attempt_count < allowed_recovery_attempt_count
}

impl Qwen3_5EngineState {
    /// Sheds retained expert residency, rebuilds decoder state from the
    /// persistent prompt cache, and leaves the request ready to retry prefill.
    pub(super) fn recover_prefill_after_memory_exhaustion(
        &mut self,
        request_id: RequestId,
        active_request: &mut Qwen3_5EngineRequest,
        failed_prefill_start: usize,
    ) -> Result<(), InferenceEngineError> {
        active_request
            .engine_extension
            .completed_prefill_memory_recovery_attempt_count = active_request
            .engine_extension
            .completed_prefill_memory_recovery_attempt_count
            .saturating_add(1);
        active_request.engine_extension.post_recovery_expert_refill = active_request
            .engine_extension
            .post_recovery_expert_refill
            .after_recovery();
        active_request
            .engine_extension
            .planned_prefill_workspace_bytes_at_recovery = self.planned_prefill_workspace_bytes();
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
        let retained_expert_payload_bytes_before_shedding = model
            .expert_weight_memory_cache_statistics()
            .resident_payload_byte_count;
        // `usize::MAX` targets every retained byte; the freeze is re-applied
        // after the prompt-cache restore below.
        let memory_snapshot_after_shedding =
            match reclaim_retained_payload_for_request_memory_pressure(model, usize::MAX)? {
                Some(memory_snapshot_after_shedding) => Some(memory_snapshot_after_shedding),
                None => {
                    model
                        .runtime()
                        .synchronize_gpu_stream_and_clear_allocator_cache()
                        .map_err(qwen3_5_runtime_error)?;
                    model.runtime().memory_snapshot().ok()
                }
            };
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
        // Restore lifts any request-pressure freeze when its own load memory is
        // gone, so re-freeze retention at zero: the retry streams experts
        // from storage, which is slower but cannot repeat the failure that
        // residency just caused. The freeze lifts once a chunk has measured
        // the real workspace (`prefill_expert_refill`), or at the decode
        // handoff.
        self.model
            .as_ref()
            .ok_or_else(|| fatal_engine_error("Qwen3.5 engine lost its loaded model"))?
            .freeze_expert_retention_for_prefill_memory_recovery();
        // Demand evidence recorded by discarded chunks must not steer the
        // decode handoff, so the retry opens a fresh window.
        active_request
            .engine_extension
            .has_opened_prefill_demand_window = false;
        tracing::warn!(
            request_id = request_id.value(),
            failed_prefill_start,
            resumed_prefill_cursor = restored_prefill_cursor,
            retained_expert_payload_bytes_before_shedding,
            active_memory_bytes_after_shedding = memory_snapshot_after_shedding
                .as_ref()
                .map(|memory_snapshot| memory_snapshot.active_memory_bytes()),
            "recovered prefill memory exhaustion by shedding expert residency and re-seeding from the persistent prompt cache"
        );
        Ok(())
    }
}

/// How a layer-major prefill failed, which decides how the request goes on.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum LayerMajorDecline {
    /// Admission refused the group before any forward ran: the decoder
    /// state is intact, though admission may have released kept layers.
    NotAdmitted,
    /// The forward ran out of memory and may have half-updated the state.
    RanOutOfMemory,
}

/// Classifies a layer-major failure, or `None` for an engine failure that
/// ends the request whatever path it took.
pub(super) const fn layer_major_decline_for(
    prompt_prefill_chunk_attempt_error: &PromptPrefillChunkAttemptError,
) -> Option<LayerMajorDecline> {
    match prompt_prefill_chunk_attempt_error {
        PromptPrefillChunkAttemptError::Engine(_) => None,
        PromptPrefillChunkAttemptError::AdaptiveMemoryLimitExceeded { .. } => {
            Some(LayerMajorDecline::NotAdmitted)
        }
        PromptPrefillChunkAttemptError::ActiveMemoryLimitExceeded { .. }
        | PromptPrefillChunkAttemptError::GraphicsProcessorMemoryExhausted { .. } => {
            Some(LayerMajorDecline::RanOutOfMemory)
        }
    }
}

impl Qwen3_5EngineState {
    /// Sends a request's prompt chunk by chunk after its layer-major prefill
    /// failed (issue #1985).
    ///
    /// Holding every span's hidden states and checkpoints at once can need
    /// more than one chunk did, so a failure means the group did not fit, not
    /// that the prompt cannot run. A refusal ran nothing, so the decoder
    /// state stands. A failure inside the forward may have half-updated it,
    /// so that is rebuilt from the prompt cache like any recovery, without
    /// spending the chunk path's own one recovery. Either way admission or
    /// recovery may have given kept layers back, so the planned layers are
    /// kept again; if that cannot fit beside this request's state, the
    /// layers stream until the request ends, which is still correct.
    pub(super) fn decline_layer_major_prefill_after_failure(
        &mut self,
        request_id: RequestId,
        active_request: &mut Qwen3_5EngineRequest,
        prefill_start: usize,
        prompt_prefill_chunk_attempt_error: PromptPrefillChunkAttemptError,
    ) -> Result<(), InferenceEngineError> {
        let layer_major_decline = match prompt_prefill_chunk_attempt_error {
            PromptPrefillChunkAttemptError::Engine(generation_error) => {
                return Err(generation_error);
            }
            memory_failure => layer_major_decline_for(&memory_failure)
                .ok_or_else(|| fatal_engine_error("a layer-major memory failure had no decline"))?,
        };
        tracing::warn!(
            request_id = request_id.value(),
            prefill_start,
            ?layer_major_decline,
            "layer-major prefill did not fit; continuing chunk by chunk"
        );
        active_request.engine_extension.layer_major_prefill_declined = true;
        if layer_major_decline == LayerMajorDecline::RanOutOfMemory {
            let completed_recovery_attempt_count = active_request
                .engine_extension
                .completed_prefill_memory_recovery_attempt_count;
            self.recover_prefill_after_memory_exhaustion(
                request_id,
                active_request,
                prefill_start,
            )?;
            active_request
                .engine_extension
                .completed_prefill_memory_recovery_attempt_count = completed_recovery_attempt_count;
        }
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
