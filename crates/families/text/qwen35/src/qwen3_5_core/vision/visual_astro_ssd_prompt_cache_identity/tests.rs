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

    let original_plan = super::plan_qwen3_5_visual_astro_ssd_prompt_cache_block_inputs(
        &original_prompt_tokens,
        4,
        &[first_image_digest],
        &[2],
        IMAGE_PAD_TOKEN_ID,
    )
    .expect("the original visual block plan should be valid");
    let appended_plan = super::plan_qwen3_5_visual_astro_ssd_prompt_cache_block_inputs(
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

    let plan = super::plan_qwen3_5_visual_astro_ssd_prompt_cache_block_inputs(
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

    let first_order_plan = super::plan_qwen3_5_visual_astro_ssd_prompt_cache_block_inputs(
        &prompt_tokens,
        4,
        &[first_image_digest, second_image_digest],
        &[1, 1],
        IMAGE_PAD_TOKEN_ID,
    )
    .expect("the first image order should produce a plan");
    let second_order_plan = super::plan_qwen3_5_visual_astro_ssd_prompt_cache_block_inputs(
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
    let first_plan = super::plan_qwen3_5_visual_astro_ssd_prompt_cache_block_inputs(
        &prompt_tokens,
        4,
        &[[1_u8; 32]],
        &[2],
        IMAGE_PAD_TOKEN_ID,
    )
    .expect("the first visual block plan should be valid");
    let replacement_plan = super::plan_qwen3_5_visual_astro_ssd_prompt_cache_block_inputs(
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

    let plan = super::plan_qwen3_5_visual_astro_ssd_prompt_cache_block_inputs(
        &prompt_tokens,
        4,
        &[[1_u8; 32]],
        &[2],
        IMAGE_PAD_TOKEN_ID,
    );

    assert!(plan.is_err());
}

#[test]
fn should_accept_inputs_that_pair_up_and_have_rows() {
    let validation = super::validate_visual_identity_inputs(4, &[[1_u8; 32], [2_u8; 32]], &[3, 1]);

    assert_eq!(validation, Ok(()));
}

#[test]
fn should_reject_a_zero_block_token_count() {
    let validation = super::validate_visual_identity_inputs(0, &[], &[]);

    assert_eq!(
        validation,
        Err(super::Qwen3_5VisualAstroSsdPromptCacheIdentityPlanError::ZeroBlockTokenCount)
    );
}

#[test]
fn should_reject_digests_that_do_not_pair_with_row_counts() {
    let validation = super::validate_visual_identity_inputs(4, &[[1_u8; 32]], &[2, 2]);

    assert_eq!(
        validation,
        Err(
            super::Qwen3_5VisualAstroSsdPromptCacheIdentityPlanError::ImageIdentityCountMismatch {
                image_digest_count: 1,
                image_row_count_entry_count: 2,
            }
        )
    );
}

#[test]
fn should_name_the_first_image_that_has_no_rows() {
    let validation = super::validate_visual_identity_inputs(
        4,
        &[[1_u8; 32], [2_u8; 32], [3_u8; 32]],
        &[2, 0, 0],
    );

    assert_eq!(
        validation,
        Err(
            super::Qwen3_5VisualAstroSsdPromptCacheIdentityPlanError::ZeroImageRows {
                image_index: 1
            }
        )
    );
}

#[test]
fn should_list_image_pad_positions_in_prompt_order() {
    let prompt_tokens = [
        10,
        IMAGE_PAD_TOKEN_ID,
        11,
        IMAGE_PAD_TOKEN_ID,
        IMAGE_PAD_TOKEN_ID,
    ];

    assert_eq!(
        super::find_image_pad_token_positions(&prompt_tokens, IMAGE_PAD_TOKEN_ID),
        [1, 3, 4]
    );
    assert!(super::find_image_pad_token_positions(&[10, 11], IMAGE_PAD_TOKEN_ID).is_empty());
    assert!(super::find_image_pad_token_positions(&[], IMAGE_PAD_TOKEN_ID).is_empty());
}

#[test]
fn should_sum_image_row_counts_and_refuse_an_overflowing_sum() {
    assert_eq!(super::sum_image_row_counts(&[3, 4, 5]), Ok(12));
    assert_eq!(super::sum_image_row_counts(&[]), Ok(0));
    assert_eq!(
        super::sum_image_row_counts(&[usize::MAX, 1]),
        Err(super::Qwen3_5VisualAstroSsdPromptCacheIdentityPlanError::ImageRowCountOverflow)
    );
}

#[test]
fn should_accept_adjacent_pad_positions_and_a_single_pad_row() {
    assert_eq!(
        super::ensure_contiguous_image_pad_rows(&[5, 6, 7], 0),
        Ok(())
    );
    assert_eq!(super::ensure_contiguous_image_pad_rows(&[5], 0), Ok(()));
}

#[test]
fn should_reject_pad_positions_with_a_gap_and_name_the_image() {
    assert_eq!(
        super::ensure_contiguous_image_pad_rows(&[5, 6, 8], 2),
        Err(
            super::Qwen3_5VisualAstroSsdPromptCacheIdentityPlanError::NoncontiguousImagePadRows {
                image_index: 2
            }
        )
    );
}

#[test]
fn should_keep_an_image_that_fits_inside_one_block_as_one_segment() {
    let mut block_segments = vec![Vec::new(); 2];

    super::push_image_block_segments(&mut block_segments, [4_u8; 32], 1, 2, 4);

    assert_eq!(
        block_segments[0],
        [super::Qwen3_5VisualBlockSegment {
            image_sha256_digest: [4_u8; 32],
            block_token_offset: 1,
            image_row_start: 0,
            row_count: 2,
        }]
    );
    assert!(block_segments[1].is_empty());
}

#[test]
fn should_cut_an_image_at_every_block_boundary_it_crosses() {
    let mut block_segments = vec![Vec::new(); 3];

    super::push_image_block_segments(&mut block_segments, [5_u8; 32], 3, 7, 4);

    let segment_geometry = block_segments
        .iter()
        .map(|segments| {
            segments
                .iter()
                .map(|segment| {
                    (
                        segment.block_token_offset,
                        segment.image_row_start,
                        segment.row_count,
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    assert_eq!(
        segment_geometry,
        [vec![(3, 0, 1)], vec![(0, 1, 4)], vec![(0, 5, 2)]]
    );
}

#[test]
fn should_start_a_segment_at_offset_zero_when_an_image_begins_on_a_block_boundary() {
    let mut block_segments = vec![Vec::new(); 2];

    super::push_image_block_segments(&mut block_segments, [6_u8; 32], 4, 4, 4);

    assert!(block_segments[0].is_empty());
    assert_eq!(block_segments[1].len(), 1);
    assert_eq!(block_segments[1][0].block_token_offset, 0);
    assert_eq!(block_segments[1][0].row_count, 4);
}
