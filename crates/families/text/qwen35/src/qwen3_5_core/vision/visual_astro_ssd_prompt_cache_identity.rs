//! Qwen-owned mapping from visual prompt rows to causal `astro_ssd_prompt_cache` block identity.
//!
//! Decoder-state keys must bind image content where projected visual rows enter
//! the token stream. Keeping this planner beside Qwen vision prevents the
//! architecture-neutral `astro_ssd_prompt_cache` from learning image-pad semantics.

use thiserror::Error;

#[cfg(test)]
mod tests;

use engine_core::astro_ssd_prompt_cache::block_causal_input::AstroSsdPromptCacheBlockCausalInput;

const QWEN_VISUAL_BLOCK_CAUSAL_INPUT_DOMAIN: &[u8] = b"astronomical-qwen3-5-visual-block-v1";

#[derive(Clone, Debug, Eq, PartialEq)]
struct Qwen3_5VisualBlockSegment {
    image_sha256_digest: [u8; 32],
    block_token_offset: usize,
    image_row_start: usize,
    row_count: usize,
}

/// Canonical causal input for every prompt block, including the trailing partial block.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen3_5VisualAstroSsdPromptCacheIdentityPlan {
    block_causal_inputs: Vec<AstroSsdPromptCacheBlockCausalInput>,
}

impl Qwen3_5VisualAstroSsdPromptCacheIdentityPlan {
    /// Returns block inputs in prompt order.
    #[must_use]
    pub fn block_causal_inputs(&self) -> &[AstroSsdPromptCacheBlockCausalInput] {
        &self.block_causal_inputs
    }
}

/// Binds each image digest and row range to the block positions that consume it.
pub fn plan_qwen3_5_visual_astro_ssd_prompt_cache_block_inputs(
    prompt_token_ids: &[u32],
    block_token_count: usize,
    ordered_image_sha256_digests: &[[u8; 32]],
    ordered_image_visual_embedding_row_counts: &[usize],
    image_pad_token_id: u32,
) -> Result<
    Qwen3_5VisualAstroSsdPromptCacheIdentityPlan,
    Qwen3_5VisualAstroSsdPromptCacheIdentityPlanError,
> {
    validate_visual_identity_inputs(
        block_token_count,
        ordered_image_sha256_digests,
        ordered_image_visual_embedding_row_counts,
    )?;

    let image_pad_token_positions =
        find_image_pad_token_positions(prompt_token_ids, image_pad_token_id);
    let total_image_row_count = sum_image_row_counts(ordered_image_visual_embedding_row_counts)?;
    if image_pad_token_positions.len() != total_image_row_count {
        return Err(
            Qwen3_5VisualAstroSsdPromptCacheIdentityPlanError::ImagePadCountMismatch {
                image_pad_token_count: image_pad_token_positions.len(),
                total_image_row_count,
            },
        );
    }

    let prompt_block_count = prompt_token_ids.len().div_ceil(block_token_count);
    let block_segments = assign_image_segments_to_blocks(
        ordered_image_sha256_digests,
        ordered_image_visual_embedding_row_counts,
        &image_pad_token_positions,
        prompt_block_count,
        block_token_count,
    )?;

    let block_causal_inputs = block_segments
        .into_iter()
        .map(canonical_block_causal_input)
        .collect();
    Ok(Qwen3_5VisualAstroSsdPromptCacheIdentityPlan {
        block_causal_inputs,
    })
}

/// Rejects the inputs no block identity can be built from: a zero block size,
/// digests and row counts that do not pair up, and an image with no rows.
fn validate_visual_identity_inputs(
    block_token_count: usize,
    ordered_image_sha256_digests: &[[u8; 32]],
    ordered_image_visual_embedding_row_counts: &[usize],
) -> Result<(), Qwen3_5VisualAstroSsdPromptCacheIdentityPlanError> {
    if block_token_count == 0 {
        return Err(Qwen3_5VisualAstroSsdPromptCacheIdentityPlanError::ZeroBlockTokenCount);
    }
    if ordered_image_sha256_digests.len() != ordered_image_visual_embedding_row_counts.len() {
        return Err(
            Qwen3_5VisualAstroSsdPromptCacheIdentityPlanError::ImageIdentityCountMismatch {
                image_digest_count: ordered_image_sha256_digests.len(),
                image_row_count_entry_count: ordered_image_visual_embedding_row_counts.len(),
            },
        );
    }
    if let Some((image_index, _)) = ordered_image_visual_embedding_row_counts
        .iter()
        .enumerate()
        .find(|(_, image_row_count)| **image_row_count == 0)
    {
        return Err(
            Qwen3_5VisualAstroSsdPromptCacheIdentityPlanError::ZeroImageRows { image_index },
        );
    }
    Ok(())
}

/// Prompt positions of every image-pad token, in prompt order.
fn find_image_pad_token_positions(prompt_token_ids: &[u32], image_pad_token_id: u32) -> Vec<usize> {
    prompt_token_ids
        .iter()
        .enumerate()
        .filter_map(|(token_index, token_id)| {
            (*token_id == image_pad_token_id).then_some(token_index)
        })
        .collect::<Vec<_>>()
}

/// Sum of every image's visual row count, refusing a sum that overflows.
fn sum_image_row_counts(
    ordered_image_visual_embedding_row_counts: &[usize],
) -> Result<usize, Qwen3_5VisualAstroSsdPromptCacheIdentityPlanError> {
    ordered_image_visual_embedding_row_counts
        .iter()
        .try_fold(0_usize, |accumulated_row_count, image_row_count| {
            accumulated_row_count.checked_add(*image_row_count)
        })
        .ok_or(Qwen3_5VisualAstroSsdPromptCacheIdentityPlanError::ImageRowCountOverflow)
}

/// Walks the images in prompt order, consuming each image's run of pad
/// positions, and files every image's rows under the blocks they land in.
fn assign_image_segments_to_blocks(
    ordered_image_sha256_digests: &[[u8; 32]],
    ordered_image_visual_embedding_row_counts: &[usize],
    image_pad_token_positions: &[usize],
    prompt_block_count: usize,
    block_token_count: usize,
) -> Result<Vec<Vec<Qwen3_5VisualBlockSegment>>, Qwen3_5VisualAstroSsdPromptCacheIdentityPlanError>
{
    let mut block_segments = vec![Vec::new(); prompt_block_count];
    let mut image_pad_position_cursor = 0_usize;
    for ((image_index, image_sha256_digest), image_row_count) in ordered_image_sha256_digests
        .iter()
        .copied()
        .enumerate()
        .zip(ordered_image_visual_embedding_row_counts.iter().copied())
    {
        let image_pad_position_end = image_pad_position_cursor
            .checked_add(image_row_count)
            .ok_or(Qwen3_5VisualAstroSsdPromptCacheIdentityPlanError::ImageRowCountOverflow)?;
        let image_pad_positions =
            &image_pad_token_positions[image_pad_position_cursor..image_pad_position_end];
        ensure_contiguous_image_pad_rows(image_pad_positions, image_index)?;
        push_image_block_segments(
            &mut block_segments,
            image_sha256_digest,
            image_pad_positions[0],
            image_row_count,
            block_token_count,
        );
        image_pad_position_cursor = image_pad_position_end;
    }
    Ok(block_segments)
}

/// An image's pad tokens must sit side by side in the prompt, because its rows
/// are bound to the blocks by position.
fn ensure_contiguous_image_pad_rows(
    image_pad_positions: &[usize],
    image_index: usize,
) -> Result<(), Qwen3_5VisualAstroSsdPromptCacheIdentityPlanError> {
    if image_pad_positions
        .windows(2)
        .any(|adjacent_positions| adjacent_positions[1] != adjacent_positions[0] + 1)
    {
        return Err(
            Qwen3_5VisualAstroSsdPromptCacheIdentityPlanError::NoncontiguousImagePadRows {
                image_index,
            },
        );
    }
    Ok(())
}

/// Cuts one image's rows at every block boundary they cross and appends each
/// piece to the block it falls in, with the piece's offset inside that block
/// and the image row it starts at.
fn push_image_block_segments(
    block_segments: &mut [Vec<Qwen3_5VisualBlockSegment>],
    image_sha256_digest: [u8; 32],
    image_prompt_start: usize,
    image_row_count: usize,
    block_token_count: usize,
) {
    let mut image_row_start = 0_usize;
    while image_row_start < image_row_count {
        let segment_prompt_start = image_prompt_start + image_row_start;
        let block_index = segment_prompt_start / block_token_count;
        let block_token_offset = segment_prompt_start % block_token_count;
        let row_count =
            (block_token_count - block_token_offset).min(image_row_count - image_row_start);
        block_segments[block_index].push(Qwen3_5VisualBlockSegment {
            image_sha256_digest,
            block_token_offset,
            image_row_start,
            row_count,
        });
        image_row_start += row_count;
    }
}

fn canonical_block_causal_input(
    visual_block_segments: Vec<Qwen3_5VisualBlockSegment>,
) -> AstroSsdPromptCacheBlockCausalInput {
    if visual_block_segments.is_empty() {
        return AstroSsdPromptCacheBlockCausalInput::empty();
    }
    let mut canonical_bytes = Vec::with_capacity(
        QWEN_VISUAL_BLOCK_CAUSAL_INPUT_DOMAIN.len() + 8 + visual_block_segments.len() * 56,
    );
    canonical_bytes.extend_from_slice(QWEN_VISUAL_BLOCK_CAUSAL_INPUT_DOMAIN);
    canonical_bytes.extend_from_slice(&(visual_block_segments.len() as u64).to_be_bytes());
    for visual_block_segment in visual_block_segments {
        canonical_bytes.extend_from_slice(&visual_block_segment.image_sha256_digest);
        canonical_bytes
            .extend_from_slice(&(visual_block_segment.block_token_offset as u64).to_be_bytes());
        canonical_bytes
            .extend_from_slice(&(visual_block_segment.image_row_start as u64).to_be_bytes());
        canonical_bytes.extend_from_slice(&(visual_block_segment.row_count as u64).to_be_bytes());
    }
    AstroSsdPromptCacheBlockCausalInput::from_canonical_bytes(&canonical_bytes)
}

/// Visual prompt geometry cannot produce safe per-block astro SSD prompt cache identity.
#[derive(Debug, Error, Eq, PartialEq)]
pub enum Qwen3_5VisualAstroSsdPromptCacheIdentityPlanError {
    #[error("astro SSD prompt cache block token count must be positive")]
    ZeroBlockTokenCount,
    #[error(
        "visual prompt has {image_digest_count} image digests but {image_row_count_entry_count} image row counts"
    )]
    ImageIdentityCountMismatch {
        image_digest_count: usize,
        image_row_count_entry_count: usize,
    },
    #[error("image {image_index} has zero visual rows")]
    ZeroImageRows { image_index: usize },
    #[error("total image visual row count overflowed")]
    ImageRowCountOverflow,
    #[error(
        "prompt has {image_pad_token_count} image-pad tokens but images provide {total_image_row_count} visual rows"
    )]
    ImagePadCountMismatch {
        image_pad_token_count: usize,
        total_image_row_count: usize,
    },
    #[error("image {image_index} has noncontiguous image-pad rows")]
    NoncontiguousImagePadRows { image_index: usize },
}
