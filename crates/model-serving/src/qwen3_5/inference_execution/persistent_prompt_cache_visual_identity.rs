//! Request preparation for position-aware Qwen visual prompt-cache identity.
//!
//! This owner validates image row geometry once, builds the ordinary target
//! block plan once, and lets lookup and capture consume the same immutable identities.

use crate::{
    InferenceEngineError, PerformanceAttribution, PerformanceOperation,
    PersistentPromptCacheBlockCausalInput, PersistentPromptCacheDiskStore, Qwen3_5InferenceRequest,
    plan_qwen3_5_visual_prompt_cache_block_inputs,
};

use super::super::model::memory_admission::invalid_request_error;
use super::fatal_engine_error;

pub(super) struct Qwen3_5PersistentPromptCacheVisualIdentity {
    pub(super) ordered_image_sha256_digests: Vec<[u8; 32]>,
    pub(super) ordered_image_visual_embedding_row_counts: Vec<usize>,
    pub(super) block_causal_inputs: Vec<PersistentPromptCacheBlockCausalInput>,
}

/// Borrowed request context keeps prompt scanning and cache-policy decisions in their existing owners.
pub(super) struct Qwen3_5PersistentPromptCacheVisualIdentityInput<'a> {
    pub(super) prompt_token_ids: &'a [u32],
    pub(super) prompt_image_pad_token_count: usize,
    pub(super) image_pad_token_id: u32,
    pub(super) persistent_prompt_cache: Option<&'a PersistentPromptCacheDiskStore>,
    pub(super) can_use_persistent_prompt_cache: bool,
    pub(super) speculative_prefill_is_enabled: bool,
}

impl Qwen3_5PersistentPromptCacheVisualIdentity {
    pub(super) fn prepare(
        inference_request: &Qwen3_5InferenceRequest,
        visual_identity_input: Qwen3_5PersistentPromptCacheVisualIdentityInput<'_>,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<Self, InferenceEngineError> {
        let has_processed_visual_images = inference_request.has_processed_visual_images();
        let ordered_image_sha256_digests = if has_processed_visual_images
            && (visual_identity_input.persistent_prompt_cache.is_some()
                || visual_identity_input.speculative_prefill_is_enabled)
        {
            inference_request
                .processed_visual_images()
                .iter()
                .map(|processed_visual_image| processed_visual_image.encoded_image_sha256)
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        let ordered_image_visual_embedding_row_counts = if has_processed_visual_images {
            inference_request
                .processed_visual_images()
                .iter()
                .map(|processed_visual_image| {
                    processed_visual_image.image_token_count_after_spatial_merge
                })
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        let expected_visual_row_count = ordered_image_visual_embedding_row_counts
            .iter()
            .try_fold(0_usize, |accumulated_visual_row_count, image_row_count| {
                accumulated_visual_row_count.checked_add(*image_row_count)
            })
            .ok_or_else(|| fatal_engine_error("processed image token count overflowed"))?;
        if has_processed_visual_images
            && visual_identity_input.prompt_image_pad_token_count != expected_visual_row_count
        {
            return Err(invalid_request_error(
                "image pad token count does not match processed image token count",
            ));
        }
        let block_causal_inputs = if has_processed_visual_images
            && visual_identity_input.can_use_persistent_prompt_cache
        {
            let block_token_count = visual_identity_input
                .persistent_prompt_cache
                .ok_or_else(|| {
                    fatal_engine_error(
                        "persistent prompt cache disappeared during visual identity planning",
                    )
                })?
                .model_contract
                .block_token_count();
            performance_attribution
                .measure_operation(
                    PerformanceOperation::PersistentPromptCacheCausalInputPlanning,
                    |_performance_attribution| {
                        plan_qwen3_5_visual_prompt_cache_block_inputs(
                            visual_identity_input.prompt_token_ids,
                            block_token_count,
                            &ordered_image_sha256_digests,
                            &ordered_image_visual_embedding_row_counts,
                            visual_identity_input.image_pad_token_id,
                        )
                    },
                )
                .map_err(|visual_identity_plan_error| {
                    invalid_request_error(format!(
                        "visual prompt-cache identity planning failed: {visual_identity_plan_error}"
                    ))
                })?
                .block_causal_inputs()
                .to_vec()
        } else {
            Vec::new()
        };
        Ok(Self {
            ordered_image_sha256_digests,
            ordered_image_visual_embedding_row_counts,
            block_causal_inputs,
        })
    }
}
