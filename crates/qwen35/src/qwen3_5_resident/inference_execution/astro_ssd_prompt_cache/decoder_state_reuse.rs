//! The resident engine's part of an `astro_ssd_prompt_cache` restore: before
//! the disk store is read, the live prefix kept from the last reply may take
//! the restore over (in-memory prefix adoption, issue #1167). The rest of the
//! restore is shared with the streaming engine in
//! `qwen3_5_core::shared_execution::astro_ssd_prompt_cache_restore`.

use ipc_protocol::WorkerInMemoryPrefixAdoptionOutcome;

use crate::astro_ssd_prompt_cache::prefix_lookup::AstroSsdPromptCachePrefixLookup;
use crate::qwen3_5_core::decoder::RequestDecoderStateStack;
use crate::qwen3_5_core::shared_execution::astro_ssd_prompt_cache_restore::{
    AstroSsdPromptCacheRestoreRequest, LivePrefixAdoption, Qwen3_5AstroSsdPromptCacheRestoreHooks,
};
use crate::{InferenceEngineError, PerformanceAttribution};

use super::Qwen3_5EngineState;

impl Qwen3_5AstroSsdPromptCacheRestoreHooks for Qwen3_5EngineState {
    /// Matches the prompt's chain against the retained live state and, when the
    /// retained state wins, installs it as the restore outcome directly.
    fn adopt_retained_live_prefix(
        &mut self,
        restore_request: &AstroSsdPromptCacheRestoreRequest<'_>,
        disk_restored_token_count: usize,
        request_decoder_state: &mut RequestDecoderStateStack,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<LivePrefixAdoption, InferenceEngineError> {
        let astro_ssd_prompt_cache = restore_request.astro_ssd_prompt_cache;
        if !self
            .core
            .chunking
            .experimental_in_memory_prefix_adoption_enabled
        {
            return Ok(LivePrefixAdoption::NotAdopted(None));
        }
        if self.retained_live_prefix.is_none() {
            return Ok(LivePrefixAdoption::NotAdopted(Some(
                WorkerInMemoryPrefixAdoptionOutcome::NoRetainedState,
            )));
        }
        let retained_live_prefix = match self.retained_live_prefix.take() {
            Some(retained_live_prefix) => retained_live_prefix,
            None => unreachable!("the holder was just proven present"),
        };
        let retained_lookup = AstroSsdPromptCachePrefixLookup::for_retained_complete_block_keys(
            &astro_ssd_prompt_cache.model_contract,
            restore_request.prompt_token_ids,
            restore_request.block_causal_inputs,
            &retained_live_prefix.complete_block_keys,
            |block_hash| astro_ssd_prompt_cache.has_recurrent_snapshot(block_hash),
        );
        Ok(
            match super::in_memory_adoption_decision::decide_in_memory_prefix_adoption(
                retained_lookup.restored_token_count(),
                disk_restored_token_count,
            ) {
                super::in_memory_adoption_decision::InMemoryPrefixAdoptionDecision::AdoptRetainedState => {
                    let adopted_restore_outcome = self.install_adopted_live_prefix(
                        astro_ssd_prompt_cache,
                        retained_live_prefix,
                        retained_lookup,
                        request_decoder_state,
                        performance_attribution,
                    )?;
                    LivePrefixAdoption::Adopted(Box::new(adopted_restore_outcome))
                }
                // The retained state lost to the disk store or diverged from
                // this prompt's chain: drop it. Its arrays return to the
                // `mlx_buffer_cache` here rather than serving a conversation
                // they no longer match.
                super::in_memory_adoption_decision::InMemoryPrefixAdoptionDecision::ChainMismatch => {
                    LivePrefixAdoption::NotAdopted(Some(
                        WorkerInMemoryPrefixAdoptionOutcome::ChainMismatch,
                    ))
                }
                super::in_memory_adoption_decision::InMemoryPrefixAdoptionDecision::RetainedStateShorter => {
                    LivePrefixAdoption::NotAdopted(Some(
                        WorkerInMemoryPrefixAdoptionOutcome::RetainedStateShorter,
                    ))
                }
            },
        )
    }
}
