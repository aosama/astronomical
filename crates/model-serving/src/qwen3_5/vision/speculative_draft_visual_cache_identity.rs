//! Conservative visual identity for the independent SpecPrefill drafter cache.
//!
//! Draft scoring currently carries image digests but not image-pad row geometry.
//! It therefore retains the safe root binding while ordinary target caching uses
//! the position-aware planner in `visual_prompt_cache_identity`.

use crate::PersistentPromptCacheBlockCausalInput;

const QWEN_SPECULATIVE_DRAFT_VISUAL_ROOT_DOMAIN: &[u8] =
    b"astronomical-qwen3-5-speculative-draft-images-v1";

pub(crate) fn qwen3_5_speculative_draft_block_causal_inputs(
    prompt_token_count: usize,
    block_token_count: usize,
    ordered_image_sha256_digests: &[[u8; 32]],
) -> Vec<PersistentPromptCacheBlockCausalInput> {
    (0..prompt_token_count.div_ceil(block_token_count))
        .map(|block_index| {
            qwen3_5_speculative_draft_block_causal_input(ordered_image_sha256_digests, block_index)
        })
        .collect()
}

pub(crate) fn qwen3_5_speculative_draft_block_causal_input(
    ordered_image_sha256_digests: &[[u8; 32]],
    block_index: usize,
) -> PersistentPromptCacheBlockCausalInput {
    if block_index != 0 || ordered_image_sha256_digests.is_empty() {
        return PersistentPromptCacheBlockCausalInput::empty();
    }
    let mut canonical_bytes = QWEN_SPECULATIVE_DRAFT_VISUAL_ROOT_DOMAIN.to_vec();
    canonical_bytes.extend_from_slice(&(ordered_image_sha256_digests.len() as u64).to_be_bytes());
    for image_sha256_digest in ordered_image_sha256_digests {
        canonical_bytes.extend_from_slice(image_sha256_digest);
    }
    PersistentPromptCacheBlockCausalInput::from_canonical_bytes(&canonical_bytes)
}
