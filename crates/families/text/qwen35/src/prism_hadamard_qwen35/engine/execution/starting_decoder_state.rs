//! A new request's decoder state, with whatever prompt prefix the
//! `astro_ssd_prompt_cache` holds already seated in it.

use crate::AstroSsdPromptCacheDiskStore;
use crate::InferenceEngineError;
use crate::PerformanceAttribution;
use crate::RequestDecoderStateStack;
use crate::prism_hadamard_qwen35::engine::astro_ssd_prompt_cache::{
    PrismAstroSsdPromptCacheRestoreOutcome, lookup_and_restore_prompt_prefix,
};
use crate::prism_hadamard_qwen35::model::PrismHadamardTextModel;
use engine_core::astro_ssd_prompt_cache::AstroSsdPromptCacheCounters;

use super::engine_trait_impl::gpu_error;

/// Creates the request's empty decoder-state stack, its key-value slabs
/// growing in the core's configured step like every Qwen 3.5 request, and, when an
/// `astro_ssd_prompt_cache` store is open, restores the longest stored prompt
/// prefix into it. The outcome's restored token count is where the request's
/// first forward starts; without a store nothing is restored.
pub(super) fn starting_request_decoder_state(
    model: &PrismHadamardTextModel,
    astro_ssd_prompt_cache: Option<&AstroSsdPromptCacheDiskStore>,
    prompt_token_ids: &[u32],
    full_attention_kv_state_growth_tokens: i32,
    astro_ssd_prompt_cache_counters: &mut AstroSsdPromptCacheCounters,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<
    (
        RequestDecoderStateStack,
        PrismAstroSsdPromptCacheRestoreOutcome,
    ),
    InferenceEngineError,
> {
    let mut request_decoder_state = RequestDecoderStateStack::empty_from_astro_decoder_state_cache_layout_with_full_attention_kv_state_growth_tokens(
        &model.astro_decoder_state_cache_layout,
        full_attention_kv_state_growth_tokens,
    )
    .map_err(gpu_error)?;
    let Some(astro_ssd_prompt_cache) = astro_ssd_prompt_cache else {
        let nothing_restored = PrismAstroSsdPromptCacheRestoreOutcome {
            restored_token_count: 0,
            last_restored_astro_ssd_prompt_cache_block_key: None,
        };
        return Ok((request_decoder_state, nothing_restored));
    };
    let restore_outcome = lookup_and_restore_prompt_prefix(
        model,
        astro_ssd_prompt_cache,
        prompt_token_ids,
        &mut request_decoder_state,
        astro_ssd_prompt_cache_counters,
        performance_attribution,
    )?;
    Ok((request_decoder_state, restore_outcome))
}

#[cfg(test)]
#[path = "starting_decoder_state_tests.rs"]
mod starting_decoder_state_tests;
