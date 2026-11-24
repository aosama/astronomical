//! A new request's decoder state, with whatever prompt prefix the
//! `astro_ssd_prompt_cache` holds already seated in it by the shared restore.

use ipc_protocol::RequestId;

use crate::InferenceEngineError;
use crate::PerformanceAttribution;
use crate::RequestDecoderStateStack;
use crate::qwen3_5_core::engine_errors::qwen3_5_runtime_error;
use crate::qwen3_5_core::shared_execution::astro_ssd_prompt_cache_restore::{
    AstroSsdPromptCacheRestoreOutcome, AstroSsdPromptCacheRestoreRequest,
    restore_astro_ssd_prompt_cache_prefix,
};

use super::PrismHadamardQwen35InferenceExecution;

/// What a request starts from: its decoder state and, when a store is open,
/// what the store restored into it.
pub(super) struct StartingDecoderState {
    pub(super) request_decoder_state: RequestDecoderStateStack,
    pub(super) restore_outcome: Option<AstroSsdPromptCacheRestoreOutcome>,
}

impl PrismHadamardQwen35InferenceExecution {
    /// Creates the request's empty decoder-state stack, its key-value slabs
    /// growing in the core's configured step like every Qwen 3.5 request,
    /// and, when an `astro_ssd_prompt_cache` store is open, restores the
    /// longest stored prefix of the prompt into it through the shared restore.
    pub(super) fn starting_request_decoder_state(
        &mut self,
        request_id: RequestId,
        prompt_token_ids: &[u32],
        reserved_context_tokens: usize,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<StartingDecoderState, InferenceEngineError> {
        let mut request_decoder_state = RequestDecoderStateStack::empty_from_astro_decoder_state_cache_layout_with_full_attention_kv_state_growth_tokens(
            &self.execution_core.loaded_model()?.astro_decoder_state_cache_layout,
            self.execution_core.full_attention_kv_state_growth_tokens,
        )
        .map_err(qwen3_5_runtime_error)?;
        // The store is lent out for the restore so the engine itself can be
        // borrowed mutably beside it, and is put back whatever the outcome:
        // a failed restore must not leave later requests without a store.
        let Some(astro_ssd_prompt_cache) = self.execution_core.astro_ssd_prompt_cache.take() else {
            return Ok(StartingDecoderState {
                request_decoder_state,
                restore_outcome: None,
            });
        };
        let restore_result = restore_astro_ssd_prompt_cache_prefix(
            self,
            &AstroSsdPromptCacheRestoreRequest {
                request_id,
                astro_ssd_prompt_cache: &astro_ssd_prompt_cache,
                prompt_token_ids,
                block_causal_inputs: &[],
                total_context_tokens: reserved_context_tokens,
            },
            &mut request_decoder_state,
            performance_attribution,
        );
        self.execution_core.astro_ssd_prompt_cache = Some(astro_ssd_prompt_cache);
        Ok(StartingDecoderState {
            request_decoder_state,
            restore_outcome: Some(restore_result?),
        })
    }
}

#[cfg(test)]
#[path = "starting_decoder_state_tests.rs"]
mod starting_decoder_state_tests;
