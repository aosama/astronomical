//! The `astro_ssd_prompt_cache` block length a model-derived contract is
//! rounded up to, so a prompt is never cut into uneven spans.
//!
//! The span cutter clamps every prompt span at a block boundary. A derived
//! block that is not a whole number of prompt chunks (Ornith 35B-A3B derives
//! 3,328 tokens against 2,048-token chunks) turns a long prompt's chunks into
//! 2,048 and 1,280-token fragments. The short fragments run the routed-expert
//! multiplies at fewer rows per expert, a worse operating point for the tiled
//! kernels, and add forwards: measured 2026-11-27 on an M5 Pro, a
//! 15,000-token prompt read in 6,753 ms with 3,328-token blocks and 6,626 ms
//! with 4,096-token blocks. Rounding up also writes fewer recurrent snapshots,
//! one a block, at the cost of coarser prefix reuse.

use ipc_protocol::WorkerChunkingConfiguration;

use super::model_loading_steps::AstroSsdPromptCacheContractResolver;

/// The prompt chunk lengths a block should hold a whole number of.
#[derive(Clone, Copy, Debug)]
pub struct PromptChunkSizes {
    pub resident_token_count: usize,
    pub ssd_streaming_token_count: usize,
}

impl PromptChunkSizes {
    /// The chunk sizes of a settled configuration. An in-memory chunk still
    /// Automatic (`None`, which a model's startup never leaves) counts as the
    /// streaming chunk, so blocks stay whole numbers of the chunk that is
    /// known.
    #[must_use]
    pub const fn of(chunking: &WorkerChunkingConfiguration) -> Self {
        let in_memory_chunk_tokens = match chunking.prompt_processing_chunk_size_tokens {
            Some(in_memory_chunk_tokens) => in_memory_chunk_tokens,
            None => chunking.fixed_ssd_streaming_prompt_processing_chunk_size_tokens,
        };
        Self {
            resident_token_count: in_memory_chunk_tokens as usize,
            ssd_streaming_token_count: chunking
                .fixed_ssd_streaming_prompt_processing_chunk_size_tokens
                as usize,
        }
    }
}
use crate::{
    AstroDecoderStateCacheLayout, AstroSsdPromptCacheModelContract,
    AstroSsdPromptCacheModelContractError,
};

/// Everything a contract resolution needs apart from its block length.
pub struct PromptCacheContractInputs {
    pub model_id: String,
    pub model_revision: String,
    pub astro_decoder_state_cache_layout: AstroDecoderStateCacheLayout,
    pub maximum_context_token_count: usize,
    pub active_memory_limit_bytes: u64,
    pub global_quota_bytes: u64,
    pub common_prefix_stride_blocks: u32,
}

impl PromptCacheContractInputs {
    fn resolve(
        &self,
        contract_resolver: AstroSsdPromptCacheContractResolver,
        block_token_count: Option<usize>,
    ) -> Result<AstroSsdPromptCacheModelContract, AstroSsdPromptCacheModelContractError> {
        contract_resolver(
            self.model_id.clone(),
            self.model_revision.clone(),
            self.astro_decoder_state_cache_layout.clone(),
            self.maximum_context_token_count,
            self.active_memory_limit_bytes,
            self.global_quota_bytes,
            block_token_count,
            self.common_prefix_stride_blocks,
        )
    }
}

/// The contract with the configured block length; with none configured, the
/// model-derived contract rounded up to whole prompt chunks, or the derived
/// one when the rounded length does not fit the context or the contract
/// refuses it (its SSD quota, memory or alignment), so alignment never costs
/// the prompt cache itself.
///
/// # Errors
/// Fails only when the configured or the derived contract fails.
pub fn resolve_chunk_aligned_contract(
    contract_resolver: AstroSsdPromptCacheContractResolver,
    contract_inputs: &PromptCacheContractInputs,
    configured_block_token_count: Option<usize>,
    prompt_chunk_sizes: PromptChunkSizes,
) -> Result<AstroSsdPromptCacheModelContract, AstroSsdPromptCacheModelContractError> {
    if configured_block_token_count.is_some() {
        return contract_inputs.resolve(contract_resolver, configured_block_token_count);
    }
    let derived_model_contract = contract_inputs.resolve(contract_resolver, None)?;
    let Some(aligned_block_token_count) = chunk_aligned_block_token_count(
        derived_model_contract.block_token_count(),
        prompt_chunk_sizes.resident_token_count,
        prompt_chunk_sizes.ssd_streaming_token_count,
        derived_model_contract.maximum_context_token_count(),
    ) else {
        return Ok(derived_model_contract);
    };
    Ok(contract_inputs
        .resolve(contract_resolver, Some(aligned_block_token_count))
        .unwrap_or(derived_model_contract))
}

/// The block length to try instead of `derived_block_token_count`, or `None`
/// when it already lines up with every prompt chunk size, or when no aligned
/// length fits in `maximum_context_token_count`.
///
/// The alignment is the larger chunk size when it is a whole number of the
/// smaller (the 4,096-token SSD-streaming chunk holds two 2,048-token resident
/// chunks), so the block lines up whichever way the model's experts are held;
/// otherwise the resident chunk size alone.
#[must_use]
pub fn chunk_aligned_block_token_count(
    derived_block_token_count: usize,
    resident_prompt_chunk_token_count: usize,
    ssd_streaming_prompt_chunk_token_count: usize,
    maximum_context_token_count: usize,
) -> Option<usize> {
    let chunk_alignment_token_count = prompt_chunk_alignment_token_count(
        resident_prompt_chunk_token_count,
        ssd_streaming_prompt_chunk_token_count,
    )?;
    if derived_block_token_count.is_multiple_of(chunk_alignment_token_count) {
        return None;
    }
    let aligned_block_token_count =
        derived_block_token_count.checked_next_multiple_of(chunk_alignment_token_count)?;
    (aligned_block_token_count <= maximum_context_token_count).then_some(aligned_block_token_count)
}

fn prompt_chunk_alignment_token_count(
    resident_prompt_chunk_token_count: usize,
    ssd_streaming_prompt_chunk_token_count: usize,
) -> Option<usize> {
    if resident_prompt_chunk_token_count == 0 {
        return None;
    }
    let larger_chunk_token_count =
        resident_prompt_chunk_token_count.max(ssd_streaming_prompt_chunk_token_count);
    let smaller_chunk_token_count = resident_prompt_chunk_token_count
        .min(ssd_streaming_prompt_chunk_token_count)
        .max(1);
    if larger_chunk_token_count.is_multiple_of(smaller_chunk_token_count) {
        Some(larger_chunk_token_count)
    } else {
        Some(resident_prompt_chunk_token_count)
    }
}

#[cfg(test)]
#[path = "prompt_cache_block_alignment_tests.rs"]
mod prompt_cache_block_alignment_tests;
