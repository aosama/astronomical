use astronomical_model_serving::{
    Qwen3_5VisualEmbeddingRequiredImage, plan_qwen3_5_visual_embedding_suffix,
};

/// The image pad token ID for Qwen3.5-MoE family models.
/// Used in hermetic tests where no model directory is available.
const IMAGE_PAD_TOKEN_ID: u32 = 248_056;

#[test]
fn should_use_the_caller_supplied_image_pad_token_id() {
    let caller_image_pad_token_id = 777_u32;
    let prompt_token_ids = [caller_image_pad_token_id, 42, caller_image_pad_token_id];

    let suffix_plan =
        plan_qwen3_5_visual_embedding_suffix(&prompt_token_ids, 1, &[2], caller_image_pad_token_id)
            .expect("the visual suffix plan should use the caller-supplied image-pad token");

    assert_eq!(suffix_plan.restored_visual_embedding_row_count(), 1);
    assert_eq!(suffix_plan.remaining_visual_embedding_row_count(), 1);
    assert_eq!(
        suffix_plan.required_images(),
        &[Qwen3_5VisualEmbeddingRequiredImage::new(0, 2, 1, 1)]
    );
}

#[test]
fn should_require_every_image_when_no_prompt_tokens_were_restored() {
    let prompt_token_ids = [
        IMAGE_PAD_TOKEN_ID,
        IMAGE_PAD_TOKEN_ID,
        42,
        IMAGE_PAD_TOKEN_ID,
        IMAGE_PAD_TOKEN_ID,
        IMAGE_PAD_TOKEN_ID,
    ];

    let suffix_plan =
        plan_qwen3_5_visual_embedding_suffix(&prompt_token_ids, 0, &[2, 3], IMAGE_PAD_TOKEN_ID)
            .expect("the visual suffix plan should be valid");

    assert_eq!(suffix_plan.restored_visual_embedding_row_count(), 0);
    assert_eq!(suffix_plan.remaining_visual_embedding_row_count(), 5);
    assert_eq!(
        suffix_plan.required_images(),
        &[
            Qwen3_5VisualEmbeddingRequiredImage::new(0, 2, 0, 2),
            Qwen3_5VisualEmbeddingRequiredImage::new(1, 3, 0, 3),
        ]
    );
}

#[test]
fn should_skip_images_fully_covered_by_restored_prompt_state() {
    let prompt_token_ids = [
        IMAGE_PAD_TOKEN_ID,
        IMAGE_PAD_TOKEN_ID,
        42,
        IMAGE_PAD_TOKEN_ID,
        IMAGE_PAD_TOKEN_ID,
        IMAGE_PAD_TOKEN_ID,
    ];

    let suffix_plan =
        plan_qwen3_5_visual_embedding_suffix(&prompt_token_ids, 3, &[2, 3], IMAGE_PAD_TOKEN_ID)
            .expect("the visual suffix plan should be valid");

    assert_eq!(suffix_plan.restored_visual_embedding_row_count(), 2);
    assert_eq!(suffix_plan.remaining_visual_embedding_row_count(), 3);
    assert_eq!(
        suffix_plan.required_images(),
        &[Qwen3_5VisualEmbeddingRequiredImage::new(1, 3, 0, 3)]
    );
}

#[test]
fn should_start_inside_the_first_remaining_image_when_restore_splits_an_image_pad_run() {
    let prompt_token_ids = [
        IMAGE_PAD_TOKEN_ID,
        IMAGE_PAD_TOKEN_ID,
        IMAGE_PAD_TOKEN_ID,
        42,
        IMAGE_PAD_TOKEN_ID,
        IMAGE_PAD_TOKEN_ID,
    ];

    let suffix_plan =
        plan_qwen3_5_visual_embedding_suffix(&prompt_token_ids, 1, &[3, 2], IMAGE_PAD_TOKEN_ID)
            .expect("the visual suffix plan should be valid");

    assert_eq!(suffix_plan.restored_visual_embedding_row_count(), 1);
    assert_eq!(suffix_plan.remaining_visual_embedding_row_count(), 4);
    assert_eq!(
        suffix_plan.required_images(),
        &[
            Qwen3_5VisualEmbeddingRequiredImage::new(0, 3, 1, 2),
            Qwen3_5VisualEmbeddingRequiredImage::new(1, 2, 0, 2),
        ]
    );
}

#[test]
fn should_require_no_images_when_restored_prompt_state_covers_every_image_pad_token() {
    let prompt_token_ids = [
        IMAGE_PAD_TOKEN_ID,
        IMAGE_PAD_TOKEN_ID,
        42,
        IMAGE_PAD_TOKEN_ID,
        IMAGE_PAD_TOKEN_ID,
        99,
    ];

    let suffix_plan =
        plan_qwen3_5_visual_embedding_suffix(&prompt_token_ids, 5, &[2, 2], IMAGE_PAD_TOKEN_ID)
            .expect("the visual suffix plan should be valid");

    assert_eq!(suffix_plan.restored_visual_embedding_row_count(), 4);
    assert_eq!(suffix_plan.remaining_visual_embedding_row_count(), 0);
    assert!(suffix_plan.required_images().is_empty());
}

#[test]
fn should_reject_a_prompt_whose_image_pad_count_does_not_match_image_rows() {
    let prompt_token_ids = [IMAGE_PAD_TOKEN_ID, IMAGE_PAD_TOKEN_ID];

    let suffix_plan =
        plan_qwen3_5_visual_embedding_suffix(&prompt_token_ids, 0, &[1], IMAGE_PAD_TOKEN_ID);

    assert!(suffix_plan.is_err());
}
