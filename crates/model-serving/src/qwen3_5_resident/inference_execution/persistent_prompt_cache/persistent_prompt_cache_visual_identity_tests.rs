use astronomical_ipc_protocol::RequestId;

use crate::InferenceEngineError;
use crate::PerformanceAttribution;
use crate::qwen3_5_core::text::Qwen3_5InferenceRequest;
use crate::qwen3_5_core::vision::{Qwen3_5ImageGrid, Qwen3_5ProcessedImage};

use super::Qwen3_5PersistentPromptCacheVisualIdentity;
use super::Qwen3_5PersistentPromptCacheVisualIdentityInput;

const IMAGE_PAD_TOKEN_ID: u32 = 151_655;

fn processed_image_with_token_count_after_spatial_merge(
    image_token_count_after_spatial_merge: usize,
) -> Qwen3_5ProcessedImage {
    Qwen3_5ProcessedImage {
        encoded_image_sha256: [7; 32],
        pixel_values: Vec::new(),
        pixel_values_row_count: 0,
        pixel_values_column_count: 0,
        image_grid: Qwen3_5ImageGrid {
            temporal_patch_count: 2,
            height_patch_count: 4,
            width_patch_count: 4,
        },
        resized_height_pixels: 336,
        resized_width_pixels: 336,
        image_token_count_after_spatial_merge,
    }
}

#[test]
fn should_plan_an_empty_visual_identity_for_a_text_only_request() {
    let inference_request =
        Qwen3_5InferenceRequest::new(RequestId::new(900), vec![248_045, 846, 198], 512);

    let visual_identity = Qwen3_5PersistentPromptCacheVisualIdentity::prepare(
        &inference_request,
        Qwen3_5PersistentPromptCacheVisualIdentityInput {
            prompt_token_ids: &[248_045, 846, 198],
            prompt_image_pad_token_count: 0,
            image_pad_token_id: IMAGE_PAD_TOKEN_ID,
            persistent_prompt_cache: None,
            can_use_persistent_prompt_cache: false,
        },
        &mut PerformanceAttribution::disabled(),
    )
    .expect("a text-only request should prepare without a disk store");

    assert!(
        visual_identity
            .ordered_image_visual_embedding_row_counts
            .is_empty()
    );
    assert!(visual_identity.block_causal_inputs.is_empty());
}

#[test]
fn should_collect_ordered_row_counts_without_planning_blocks_when_reuse_is_disabled() {
    let inference_request = Qwen3_5InferenceRequest::new(
        RequestId::new(901),
        vec![248_045, IMAGE_PAD_TOKEN_ID, IMAGE_PAD_TOKEN_ID],
        512,
    )
    .with_processed_visual_images(vec![
        processed_image_with_token_count_after_spatial_merge(1),
        processed_image_with_token_count_after_spatial_merge(1),
    ]);

    let visual_identity = Qwen3_5PersistentPromptCacheVisualIdentity::prepare(
        &inference_request,
        Qwen3_5PersistentPromptCacheVisualIdentityInput {
            prompt_token_ids: &[248_045, IMAGE_PAD_TOKEN_ID, IMAGE_PAD_TOKEN_ID],
            prompt_image_pad_token_count: 2,
            image_pad_token_id: IMAGE_PAD_TOKEN_ID,
            persistent_prompt_cache: None,
            can_use_persistent_prompt_cache: false,
        },
        &mut PerformanceAttribution::disabled(),
    )
    .expect("an image request with matching pad count should prepare");

    assert_eq!(
        visual_identity.ordered_image_visual_embedding_row_counts,
        vec![1, 1]
    );
    assert!(visual_identity.block_causal_inputs.is_empty());
}

#[test]
fn should_reject_an_image_pad_count_that_differs_from_the_processed_image_tokens() {
    let inference_request = Qwen3_5InferenceRequest::new(
        RequestId::new(902),
        vec![
            248_045,
            IMAGE_PAD_TOKEN_ID,
            IMAGE_PAD_TOKEN_ID,
            IMAGE_PAD_TOKEN_ID,
        ],
        512,
    )
    .with_processed_visual_images(vec![
        processed_image_with_token_count_after_spatial_merge(1),
        processed_image_with_token_count_after_spatial_merge(1),
    ]);

    let prepare_result = Qwen3_5PersistentPromptCacheVisualIdentity::prepare(
        &inference_request,
        Qwen3_5PersistentPromptCacheVisualIdentityInput {
            prompt_token_ids: &[
                248_045,
                IMAGE_PAD_TOKEN_ID,
                IMAGE_PAD_TOKEN_ID,
                IMAGE_PAD_TOKEN_ID,
            ],
            prompt_image_pad_token_count: 3,
            image_pad_token_id: IMAGE_PAD_TOKEN_ID,
            persistent_prompt_cache: None,
            can_use_persistent_prompt_cache: false,
        },
        &mut PerformanceAttribution::disabled(),
    );

    assert!(matches!(
        prepare_result,
        Err(InferenceEngineError::InvalidRequest { .. })
    ));
}

#[test]
fn should_reject_processed_image_row_counts_that_overflow_the_expected_total() {
    let inference_request = Qwen3_5InferenceRequest::new(RequestId::new(903), vec![248_045], 512)
        .with_processed_visual_images(vec![
            processed_image_with_token_count_after_spatial_merge(usize::MAX),
            processed_image_with_token_count_after_spatial_merge(usize::MAX),
        ]);

    let prepare_result = Qwen3_5PersistentPromptCacheVisualIdentity::prepare(
        &inference_request,
        Qwen3_5PersistentPromptCacheVisualIdentityInput {
            prompt_token_ids: &[248_045],
            prompt_image_pad_token_count: usize::MAX,
            image_pad_token_id: IMAGE_PAD_TOKEN_ID,
            persistent_prompt_cache: None,
            can_use_persistent_prompt_cache: false,
        },
        &mut PerformanceAttribution::disabled(),
    );

    assert!(matches!(
        prepare_result,
        Err(InferenceEngineError::Fatal { .. })
    ));
}

#[test]
fn should_require_the_disk_store_when_visual_block_planning_is_enabled() {
    let inference_request =
        Qwen3_5InferenceRequest::new(RequestId::new(904), vec![248_045, IMAGE_PAD_TOKEN_ID], 512)
            .with_processed_visual_images(vec![
                processed_image_with_token_count_after_spatial_merge(1),
            ]);

    let prepare_result = Qwen3_5PersistentPromptCacheVisualIdentity::prepare(
        &inference_request,
        Qwen3_5PersistentPromptCacheVisualIdentityInput {
            prompt_token_ids: &[248_045, IMAGE_PAD_TOKEN_ID],
            prompt_image_pad_token_count: 1,
            image_pad_token_id: IMAGE_PAD_TOKEN_ID,
            persistent_prompt_cache: None,
            can_use_persistent_prompt_cache: true,
        },
        &mut PerformanceAttribution::disabled(),
    );

    assert!(matches!(
        prepare_result,
        Err(InferenceEngineError::Fatal { .. })
    ));
}
