use astronomical_model_serving::plan_qwen3_5_visual_prompt_cache_block_inputs;

const IMAGE_PAD_TOKEN_ID: u32 = 248_056;

#[test]
fn should_preserve_earlier_block_inputs_when_a_later_image_is_appended() {
    let first_image_digest = [1_u8; 32];
    let second_image_digest = [2_u8; 32];
    let original_prompt_tokens = [
        10,
        11,
        12,
        13,
        IMAGE_PAD_TOKEN_ID,
        IMAGE_PAD_TOKEN_ID,
        20,
        21,
    ];
    let appended_prompt_tokens = [
        10,
        11,
        12,
        13,
        IMAGE_PAD_TOKEN_ID,
        IMAGE_PAD_TOKEN_ID,
        20,
        21,
        30,
        31,
        IMAGE_PAD_TOKEN_ID,
        IMAGE_PAD_TOKEN_ID,
    ];

    let original_plan = plan_qwen3_5_visual_prompt_cache_block_inputs(
        &original_prompt_tokens,
        4,
        &[first_image_digest],
        &[2],
        IMAGE_PAD_TOKEN_ID,
    )
    .expect("the original visual block plan should be valid");
    let appended_plan = plan_qwen3_5_visual_prompt_cache_block_inputs(
        &appended_prompt_tokens,
        4,
        &[first_image_digest, second_image_digest],
        &[2, 2],
        IMAGE_PAD_TOKEN_ID,
    )
    .expect("the appended visual block plan should be valid");

    assert_eq!(
        &original_plan.block_causal_inputs()[..2],
        &appended_plan.block_causal_inputs()[..2]
    );
    assert!(appended_plan.block_causal_inputs()[0].is_empty());
    assert!(!appended_plan.block_causal_inputs()[1].is_empty());
    assert!(!appended_plan.block_causal_inputs()[2].is_empty());
}

#[test]
fn should_bind_split_image_rows_to_both_blocks_with_distinct_row_offsets() {
    let image_digest = [3_u8; 32];
    let prompt_tokens = [
        IMAGE_PAD_TOKEN_ID,
        IMAGE_PAD_TOKEN_ID,
        IMAGE_PAD_TOKEN_ID,
        IMAGE_PAD_TOKEN_ID,
        IMAGE_PAD_TOKEN_ID,
        IMAGE_PAD_TOKEN_ID,
        IMAGE_PAD_TOKEN_ID,
        IMAGE_PAD_TOKEN_ID,
    ];

    let plan = plan_qwen3_5_visual_prompt_cache_block_inputs(
        &prompt_tokens,
        4,
        &[image_digest],
        &[8],
        IMAGE_PAD_TOKEN_ID,
    )
    .expect("the split visual block plan should be valid");

    assert_eq!(plan.block_causal_inputs().len(), 2);
    assert!(!plan.block_causal_inputs()[0].is_empty());
    assert!(!plan.block_causal_inputs()[1].is_empty());
    assert_ne!(plan.block_causal_inputs()[0], plan.block_causal_inputs()[1]);
}

#[test]
fn should_bind_image_order_when_adjacent_images_have_the_same_row_geometry() {
    let first_image_digest = [1_u8; 32];
    let second_image_digest = [2_u8; 32];
    let prompt_tokens = [IMAGE_PAD_TOKEN_ID, IMAGE_PAD_TOKEN_ID, 10, 11];

    let first_order_plan = plan_qwen3_5_visual_prompt_cache_block_inputs(
        &prompt_tokens,
        4,
        &[first_image_digest, second_image_digest],
        &[1, 1],
        IMAGE_PAD_TOKEN_ID,
    )
    .expect("the first image order should produce a plan");
    let second_order_plan = plan_qwen3_5_visual_prompt_cache_block_inputs(
        &prompt_tokens,
        4,
        &[second_image_digest, first_image_digest],
        &[1, 1],
        IMAGE_PAD_TOKEN_ID,
    )
    .expect("the second image order should produce a plan");

    assert_ne!(
        first_order_plan.block_causal_inputs()[0],
        second_order_plan.block_causal_inputs()[0]
    );
}

#[test]
fn should_change_the_first_block_containing_replaced_visual_content() {
    let prompt_tokens = [
        10,
        11,
        12,
        13,
        IMAGE_PAD_TOKEN_ID,
        IMAGE_PAD_TOKEN_ID,
        20,
        21,
    ];
    let first_plan = plan_qwen3_5_visual_prompt_cache_block_inputs(
        &prompt_tokens,
        4,
        &[[1_u8; 32]],
        &[2],
        IMAGE_PAD_TOKEN_ID,
    )
    .expect("the first visual block plan should be valid");
    let replacement_plan = plan_qwen3_5_visual_prompt_cache_block_inputs(
        &prompt_tokens,
        4,
        &[[2_u8; 32]],
        &[2],
        IMAGE_PAD_TOKEN_ID,
    )
    .expect("the replacement visual block plan should be valid");

    assert_eq!(
        first_plan.block_causal_inputs()[0],
        replacement_plan.block_causal_inputs()[0]
    );
    assert_ne!(
        first_plan.block_causal_inputs()[1],
        replacement_plan.block_causal_inputs()[1]
    );
}

#[test]
fn should_reject_noncontiguous_rows_for_one_image() {
    let prompt_tokens = [IMAGE_PAD_TOKEN_ID, 10, IMAGE_PAD_TOKEN_ID, 11];

    let plan = plan_qwen3_5_visual_prompt_cache_block_inputs(
        &prompt_tokens,
        4,
        &[[1_u8; 32]],
        &[2],
        IMAGE_PAD_TOKEN_ID,
    );

    assert!(plan.is_err());
}
