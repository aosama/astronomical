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
use crate::qwen3_5_streaming::model::reclaim_retained_experts_for_request_memory_pressure;

use super::engine_request::Qwen3_5EngineRequest;
use super::{Qwen3_5EngineState, fatal_engine_error, qwen3_5_runtime_error};

/// One recovery per request: after it, retained experts stay frozen at zero
/// until the decode handoff, so a second failure means the prompt's own
/// workspace does not fit and retrying again could only loop.
pub(super) const MAXIMUM_PREFILL_MEMORY_RECOVERY_ATTEMPTS_PER_REQUEST: u8 = 1;

/// Decides whether a prefill memory failure may be recovered.
///
/// Image requests stay terminal: their uploaded visual embeddings cover only
/// the prompt suffix planned at request start, so re-seeding to a different
/// cursor would misalign the consumed embedding rows.
pub(super) fn should_attempt_prefill_memory_recovery(
    completed_recovery_attempt_count: u8,
    has_visual_inputs: bool,
) -> bool {
    !has_visual_inputs
        && completed_recovery_attempt_count < MAXIMUM_PREFILL_MEMORY_RECOVERY_ATTEMPTS_PER_REQUEST
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
        active_request.completed_prefill_memory_recovery_attempt_count = active_request
            .completed_prefill_memory_recovery_attempt_count
            .saturating_add(1);
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
            match reclaim_retained_experts_for_request_memory_pressure(model, usize::MAX)? {
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
                        active_request.reserved_context_tokens,
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
        // gone, so re-freeze retention at zero: the rest of prefill streams
        // experts from storage, which is slower but cannot repeat the failure
        // that residency just caused. The decode handoff lifts this freeze.
        self.model
            .as_ref()
            .ok_or_else(|| fatal_engine_error("Qwen3.5 engine lost its loaded model"))?
            .limit_expert_retention_for_request_memory_pressure(usize::MAX)
            .map_err(InferenceEngineError::from)?;
        // Demand evidence recorded by discarded chunks must not steer the
        // decode handoff, so the retry opens a fresh window.
        active_request.has_opened_expert_demand_window = false;
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

#[cfg(test)]
#[path = "prefill_memory_recovery_tests.rs"]
mod tests;
