//! Qwen-owned mapping from visual prompt rows to causal cache-block identity.
//!
//! Decoder-state keys must bind image content where projected visual rows enter
//! the token stream. Keeping this planner beside Qwen vision prevents the
//! architecture-neutral persistent cache from learning image-pad semantics.

use thiserror::Error;

use crate::PersistentPromptCacheBlockCausalInput;

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
pub struct Qwen3_5VisualPromptCacheIdentityPlan {
    block_causal_inputs: Vec<PersistentPromptCacheBlockCausalInput>,
}

impl Qwen3_5VisualPromptCacheIdentityPlan {
    /// Returns block inputs in prompt order.
    #[must_use]
    pub fn block_causal_inputs(&self) -> &[PersistentPromptCacheBlockCausalInput] {
        &self.block_causal_inputs
    }
}

/// Binds each image digest and row range to the block positions that consume it.
pub fn plan_qwen3_5_visual_prompt_cache_block_inputs(
    prompt_token_ids: &[u32],
    block_token_count: usize,
    ordered_image_sha256_digests: &[[u8; 32]],
    ordered_image_visual_embedding_row_counts: &[usize],
    image_pad_token_id: u32,
) -> Result<Qwen3_5VisualPromptCacheIdentityPlan, Qwen3_5VisualPromptCacheIdentityPlanError> {
    if block_token_count == 0 {
        return Err(Qwen3_5VisualPromptCacheIdentityPlanError::ZeroBlockTokenCount);
    }
    if ordered_image_sha256_digests.len() != ordered_image_visual_embedding_row_counts.len() {
        return Err(
            Qwen3_5VisualPromptCacheIdentityPlanError::ImageIdentityCountMismatch {
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
        return Err(Qwen3_5VisualPromptCacheIdentityPlanError::ZeroImageRows { image_index });
    }

    let image_pad_token_positions = prompt_token_ids
        .iter()
        .enumerate()
        .filter_map(|(token_index, token_id)| {
            (*token_id == image_pad_token_id).then_some(token_index)
        })
        .collect::<Vec<_>>();
    let total_image_row_count = ordered_image_visual_embedding_row_counts
        .iter()
        .try_fold(0_usize, |accumulated_row_count, image_row_count| {
            accumulated_row_count.checked_add(*image_row_count)
        })
        .ok_or(Qwen3_5VisualPromptCacheIdentityPlanError::ImageRowCountOverflow)?;
    if image_pad_token_positions.len() != total_image_row_count {
        return Err(
            Qwen3_5VisualPromptCacheIdentityPlanError::ImagePadCountMismatch {
                image_pad_token_count: image_pad_token_positions.len(),
                total_image_row_count,
            },
        );
    }

    let prompt_block_count = prompt_token_ids.len().div_ceil(block_token_count);
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
            .ok_or(Qwen3_5VisualPromptCacheIdentityPlanError::ImageRowCountOverflow)?;
        let image_pad_positions =
            &image_pad_token_positions[image_pad_position_cursor..image_pad_position_end];
        if image_pad_positions
            .windows(2)
            .any(|adjacent_positions| adjacent_positions[1] != adjacent_positions[0] + 1)
        {
            return Err(
                Qwen3_5VisualPromptCacheIdentityPlanError::NoncontiguousImagePadRows {
                    image_index,
                },
            );
        }
        let image_prompt_start = image_pad_positions[0];
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
        image_pad_position_cursor = image_pad_position_end;
    }

    let block_causal_inputs = block_segments
        .into_iter()
        .map(canonical_block_causal_input)
        .collect();
    Ok(Qwen3_5VisualPromptCacheIdentityPlan {
        block_causal_inputs,
    })
}

fn canonical_block_causal_input(
    visual_block_segments: Vec<Qwen3_5VisualBlockSegment>,
) -> PersistentPromptCacheBlockCausalInput {
    if visual_block_segments.is_empty() {
        return PersistentPromptCacheBlockCausalInput::empty();
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
    PersistentPromptCacheBlockCausalInput::from_canonical_bytes(&canonical_bytes)
}

/// Visual prompt geometry cannot produce safe per-block cache identity.
#[derive(Debug, Error, Eq, PartialEq)]
pub enum Qwen3_5VisualPromptCacheIdentityPlanError {
    #[error("prompt-cache block token count must be positive")]
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
