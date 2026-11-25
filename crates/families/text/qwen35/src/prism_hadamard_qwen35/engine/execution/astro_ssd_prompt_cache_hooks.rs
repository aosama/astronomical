//! The folded-ternary engine's answers to the shared `astro_ssd_prompt_cache`
//! steps (issue #2473, Prism step 9): restore, block and tail capture, and
//! the contract the store is opened with.

use crate::prism_hadamard_qwen35::model::PrismHadamardTextModel;
use crate::{
    AstroDecoderStateCacheLayout, AstroSsdPromptCacheModelContract,
    AstroSsdPromptCacheModelContractError, InferenceEngineError,
};
use engine_core::astro_ssd_prompt_cache::quota_bounded_contract::resolve_quota_bounded_astro_ssd_prompt_cache_contract;
use text_generation::astro_ssd_prompt_cache_capture::TextAstroSsdPromptCacheCapture;
use text_generation::generation_loop::astro_ssd_prompt_cache_restore::TextAstroSsdPromptCacheRestoreHooks;

use super::{PrismHadamardQwen35ActiveGeneration, PrismHadamardQwen35InferenceExecution};

/// A folded pack keeps no prefix between replies and leases no room from a
/// memory governor, so every restore hook keeps its default.
impl TextAstroSsdPromptCacheRestoreHooks for PrismHadamardQwen35InferenceExecution {}

impl TextAstroSsdPromptCacheCapture for PrismHadamardQwen35InferenceExecution {
    /// The folded pack holds every weight and has nothing to evict, so the
    /// one retry after a publication blocked by the MLX active-memory limit
    /// repeats the publication unchanged, as the resident engine's does.
    fn reclaim_before_publication_retry(
        &self,
        _model: &PrismHadamardTextModel,
        _active_generation: &mut PrismHadamardQwen35ActiveGeneration,
        _active_memory_deficit_bytes: usize,
    ) -> Result<(), InferenceEngineError> {
        Ok(())
    }
}

/// The contract the folded-ternary store has always been opened with: the
/// shared resolution, retried at half the context until the block chain fits
/// the disk quota, so a pack with a long position range keeps a store
/// instead of failing to load. The common-prefix stride is floored at one
/// block, as the folded-ternary startup always floored it.
#[allow(clippy::too_many_arguments)]
pub(super) fn prism_quota_bounded_contract(
    model_id: String,
    model_revision: String,
    astro_decoder_state_cache_layout: AstroDecoderStateCacheLayout,
    maximum_context_token_count: usize,
    effective_mlx_memory_ceiling_bytes: u64,
    global_ssd_quota_bytes: u64,
    configured_block_token_count: Option<usize>,
    common_prefix_checkpoint_stride_blocks: u32,
) -> Result<AstroSsdPromptCacheModelContract, AstroSsdPromptCacheModelContractError> {
    resolve_quota_bounded_astro_ssd_prompt_cache_contract(
        &model_id,
        &model_revision,
        astro_decoder_state_cache_layout,
        maximum_context_token_count,
        effective_mlx_memory_ceiling_bytes,
        global_ssd_quota_bytes,
        configured_block_token_count,
        common_prefix_checkpoint_stride_blocks.max(1),
    )
}
