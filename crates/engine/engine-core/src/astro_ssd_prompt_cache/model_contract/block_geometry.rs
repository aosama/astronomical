//! The two block-sizing steps of `AstroSsdPromptCacheModelContract::resolve`,
//! kept as functions of plain numbers so each can be tested on its own.

use crate::AstroDecoderStateCacheLayout;
use crate::astro_ssd_prompt_cache::model_contract_storage_geometry::maximum_block_manifest_file_bytes;

use super::{
    AstroSsdPromptCacheModelContractError, BlockFileSizingInputs, BlockTokenCountSizingInputs,
    ResolvedBlockFileGeometry, block_file_geometry_within_ssd_quota,
    configured_or_derived_block_token_count,
};

/// The payload bytes one captured block carries.
pub(super) struct BlockPayloadByteCounts {
    pub(super) sequence_state_payload_bytes_per_block: usize,
    pub(super) capture_payload_bytes: usize,
}

/// Settles the block token count and the file sizes that follow from it,
/// returning them with the largest manifest a block of this model can need.
/// A configured count is honored when it fits; otherwise the count is derived
/// from the payload geometry, and either way the result is sized against the
/// global SSD quota.
pub(super) fn resolve_block_file_geometry(
    configured_block_token_count: Option<usize>,
    astro_decoder_state_cache_layout: &AstroDecoderStateCacheLayout,
    block_token_count_sizing_inputs: &BlockTokenCountSizingInputs,
) -> Result<(ResolvedBlockFileGeometry, u64), AstroSsdPromptCacheModelContractError> {
    // The block length belongs to the immutable artifact contract, not a deployment-wide
    // tuning constant. Matching append-only allocation growth avoids capture boundaries that
    // force incompatible state reshaping, while the quota-derived size prevents a laptop
    // with less SSD capacity from accepting a state block it can never retain.
    let block_token_count_is_user_configured = configured_block_token_count.is_some();
    let initial_block_token_count = configured_or_derived_block_token_count(
        configured_block_token_count,
        block_token_count_sizing_inputs,
    )?;
    let maximum_block_manifest_file_bytes = maximum_block_manifest_file_bytes(
        block_token_count_sizing_inputs.maximum_context_token_count,
    )?;
    let resolved_block_file_geometry = block_file_geometry_within_ssd_quota(
        initial_block_token_count,
        &BlockFileSizingInputs {
            astro_decoder_state_cache_layout,
            maximum_context_token_count: block_token_count_sizing_inputs
                .maximum_context_token_count,
            persistence_alignment_token_count: block_token_count_sizing_inputs
                .persistence_alignment_token_count,
            global_ssd_quota_bytes: block_token_count_sizing_inputs.global_ssd_quota_bytes,
            block_token_count_is_user_configured,
            maximum_block_manifest_file_bytes,
        },
    )?;
    Ok((
        resolved_block_file_geometry,
        maximum_block_manifest_file_bytes,
    ))
}

/// The sequence-state payload of a whole block, and the payload of one
/// capture, which is that block plus the boundary snapshot.
pub(super) fn capture_payload_byte_counts(
    sequence_state_payload_bytes_per_token: usize,
    block_token_count: usize,
    boundary_state_payload_bytes: usize,
) -> Result<BlockPayloadByteCounts, AstroSsdPromptCacheModelContractError> {
    let sequence_state_payload_bytes_per_block = sequence_state_payload_bytes_per_token
        .checked_mul(block_token_count)
        .ok_or(AstroSsdPromptCacheModelContractError::SequenceStateBlockPayloadByteCountOverflow)?;
    let capture_payload_bytes = sequence_state_payload_bytes_per_block
        .checked_add(boundary_state_payload_bytes)
        .ok_or(AstroSsdPromptCacheModelContractError::CapturePayloadByteCountOverflow)?;
    Ok(BlockPayloadByteCounts {
        sequence_state_payload_bytes_per_block,
        capture_payload_bytes,
    })
}
