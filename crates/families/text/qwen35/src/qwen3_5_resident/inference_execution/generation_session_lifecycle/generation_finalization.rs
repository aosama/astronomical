//! One terminal cleanup path for success, rejection, cancellation, and failure.
//!
//! The live prefix is kept for the next reply before the request's arrays are
//! dropped; allocator cleanup follows, then the memory governor ends the
//! request, and the final telemetry reports what remains. The telemetry, the
//! attribution record and the after-error path are shared with the streaming
//! engine in `text_generation::generation_loop::generation_finalization_steps`.

use crate::astro_decoder_state_cache::AstroDecoderStateCacheState;
use crate::{
    AstroSsdPromptCacheBlockKey, GenerationFinalization, PerformanceAttributionOutcome,
    RetainedPrefixRetentionDecision, RetainedPrefixRetentionRequirements,
};
use engine_core::astro_ssd_prompt_cache::block_causal_input::AstroSsdPromptCacheBlockCausalInput;
use ipc_protocol::RequestDraftingStatistics;
use text_generation::RequestDecoderStateStack;
use text_generation::generation_loop::{generation_finalization_steps, request_end};

use super::Qwen3_5EngineState;
use super::engine_request::ResidentEngineRequest;
use super::retained_live_prefix::RetainedLiveDecoderPrefix;

impl generation_finalization_steps::TextFinalizationHooks for Qwen3_5EngineState {
    fn retained_live_context_state_payload_bytes(&self) -> u64 {
        Qwen3_5EngineState::live_retained_context_state_payload_bytes(self)
    }

    fn measured_prompt_mlx_working_buffer_bytes(&self) -> Option<u64> {
        self.measured_prompt_chunk_mlx_working_buffer_bytes()
    }

    fn finalize_generation_request(
        &mut self,
        active_request: ResidentEngineRequest,
        performance_attribution_outcome: PerformanceAttributionOutcome,
        failure_description: Option<&'static str>,
    ) -> GenerationFinalization {
        request_end::finalize_generation_request(
            self,
            active_request,
            performance_attribution_outcome,
            failure_description,
        )
    }
}

impl request_end::TextRequestEndHooks for Qwen3_5EngineState {
    type ReplyRecord = RequestDraftingStatistics;

    fn take_reply_record(
        &mut self,
        active_request: &mut ResidentEngineRequest,
    ) -> RequestDraftingStatistics {
        // Retain the completed generation's live state for in-memory prefix
        // adoption (issue #1167) before the request drops. The memory package
        // owns the decision; the engine measures facts and enacts it. The
        // retained KV stays live, so the finalized memory snapshot honestly
        // reports it as active.
        self.retained_live_prefix = self.retain_completed_live_prefix(active_request);
        let mut drafting_statistics = active_request.engine_extension.drafting_statistics;
        // A reply that ends partway through a kept chain leaves its drafts in
        // the emission queue; every entry but the tail (the round's
        // correction token, not a draft) was counted as kept but never sent.
        let unsent_accepted_token_count = active_request
            .engine_extension
            .speculative_emission_queue
            .len()
            .saturating_sub(1);
        drafting_statistics.withdraw_unsent_accepted_tokens(
            u32::try_from(unsent_accepted_token_count).unwrap_or(u32::MAX),
        );
        drafting_statistics
    }

    fn end_governed_request_after_memory_release(&mut self) {
        self.govern_request_end();
    }

    fn attach_reply_record(
        generation_finalization: GenerationFinalization,
        drafting_statistics: RequestDraftingStatistics,
    ) -> GenerationFinalization {
        generation_finalization.with_drafting_statistics(drafting_statistics)
    }
}

impl Qwen3_5EngineState {
    /// Extracts the completed generation's live decoder state for adoption by
    /// the next request, or `None` when the retention policy drops it.
    ///
    /// The adoptable boundary is the smaller of the prompt's complete
    /// published-block chain and the live state's verified KV offset — a
    /// failed or cancelled generation must not retain uninitialized slab
    /// regions behind its processed offset.
    fn retain_completed_live_prefix(
        &mut self,
        active_request: &mut ResidentEngineRequest,
    ) -> Option<RetainedLiveDecoderPrefix> {
        let model_contract = self.core.astro_ssd_prompt_cache_model_contract.as_ref()?;
        if !active_request.can_use_astro_ssd_prompt_cache {
            return None;
        }
        let block_token_count = model_contract.block_token_count();
        let verified_state_offset_tokens = (0..active_request.request_decoder_state.layer_count())
            .filter_map(|layer_index| {
                match active_request.request_decoder_state.layer(layer_index) {
                    Some(AstroDecoderStateCacheState::AppendOnlyAttention { attention }) => {
                        Some(attention.offset_tokens())
                    }
                    _ => None,
                }
            })
            .min()?;
        let verified_state_offset_count =
            usize::try_from(verified_state_offset_tokens.max(0)).unwrap_or(0);
        let prompt_complete_block_count = active_request.input_token_ids.len() / block_token_count;
        let adoptable_block_count = verified_state_offset_count / block_token_count;
        let adoptable_block_count = adoptable_block_count.min(prompt_complete_block_count);
        let retention_decision = RetainedPrefixRetentionRequirements {
            adoption_enabled: self
                .core
                .chunking
                .experimental_in_memory_prefix_adoption_enabled,
            adoptable_token_count: adoptable_block_count * block_token_count,
            retained_state_payload_bytes: active_request.request_decoder_state.payload_byte_count(),
            effective_mlx_memory_ceiling_bytes: u64::try_from(
                self.core.memory_limits.active_memory_limit_bytes(),
            )
            .unwrap_or(u64::MAX),
        }
        .decide();
        if retention_decision != RetainedPrefixRetentionDecision::Retain {
            return None;
        }
        let mut complete_block_keys = Vec::with_capacity(adoptable_block_count);
        let mut parent_block_key: Option<AstroSsdPromptCacheBlockKey> = None;
        let empty_block_causal_input = AstroSsdPromptCacheBlockCausalInput::empty();
        for block_index in 0..adoptable_block_count {
            let block_start = block_index * block_token_count;
            let block_end = block_start + block_token_count;
            let block_causal_input = active_request
                .astro_ssd_prompt_cache_block_causal_inputs
                .get(block_index)
                .unwrap_or(&empty_block_causal_input);
            let block_key = match parent_block_key.as_ref() {
                Some(parent_block_key) => parent_block_key.for_child_block_with_causal_input(
                    &active_request.input_token_ids[block_start..block_end],
                    block_causal_input,
                ),
                None => AstroSsdPromptCacheBlockKey::for_root_block_with_causal_input(
                    model_contract,
                    &active_request.input_token_ids[block_start..block_end],
                    block_causal_input,
                ),
            }
            .ok()?;
            parent_block_key = Some(block_key.clone());
            complete_block_keys.push(block_key);
        }
        let placeholder_decoder_state =
            RequestDecoderStateStack::empty_from_astro_decoder_state_cache_layout_with_full_attention_kv_state_growth_tokens(
                self.core.model.as_ref()?.astro_decoder_state_cache_layout(),
                self.core.full_attention_kv_state_growth_tokens,
            )
            .ok()?;
        let decoder_state = std::mem::replace(
            &mut active_request.request_decoder_state,
            placeholder_decoder_state,
        );
        Some(RetainedLiveDecoderPrefix {
            decoder_state,
            complete_block_keys,
        })
    }
}
